"""
Re-antrenare end-to-end a modelului EEG-Conformer si a capului de proiectie MLP
folosind InfoNCE Contrastive Loss pentru a elimina definitiv fenomenul de "Representation Collapse".

DE CE SE BLOCA / STAGNA TOP-5 PRECEDENT:
  1. Conformer-ul era inghetat (eval): MLP primea caracteristici fixe (zgomot semantic)
     si incerca sa minimizeze MSE. Din punct de vedere matematic, cea mai simpla cale
     de a minimiza eroarea medie dintr-o intrare fara semnal este predictia unui centroid
     (vectorul mediu din spatiul BERT).
  2. MSE nu are forta de respingere: MSE trage vectorii spre clasa pozitiva, dar nu are
     un mecanism care sa impinga activ cuvintele unele de altele (negatives pushing).
  3. Hubness effect: In spatiul 768d, cuvintele cu frecventa mare precum 'love', 'film'
     si 'picture' devin atractori geometrici naturali (hubs).

SOLUTIA:
  - Antrenare comuna (Joint Training): Conformer-ul este pus in train() si optimizat
    impreuna cu MLP pentru a invata sa distinga cuvintele la nivel de unda N400.
  - InfoNCE Loss: Adauga un numitor contrastiv care forteaza respingerea
    fata de celelalte cuvinte din vocabularul activ, fortand modelul sa iasa din centroid.
  - Rate de invatare diferentiate: 1e-5 pe Conformer (fine-tuning discret) si 1e-3 pe MLP.

Rulare:
    .venv\\Scripts\\python retrain_infonce.py
"""

import os, json, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics.pairwise import cosine_similarity

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Device:', device)

# ── 1. Incarca datele preprocesate ───────────────────────────────────────────
X = np.load('data/X_preprocessed.npy')
y = np.load('data/y_labels.npy', allow_pickle=True)
splits = np.load('data/splits.npy', allow_pickle=True)

train_idx = np.where(splits == 'train')[0]
val_idx = np.where(splits == 'val')[0]
test_idx = np.where(splits == 'test')[0]

X_train, X_val, X_test = X[train_idx], X[val_idx], X[test_idx]
y_train_words = y[train_idx]
y_val_words = y[val_idx]
y_test_words = y[test_idx]

print(f'Original Trial Counts - Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}')

# ── 2. Incarca vocabular + embeddings BERT ───────────────────────────────────
config_path = os.environ.get('BENCHMARK_CONFIG', None)
if config_path is None:
    config_path = 'benchmark_config_50_Zuco1_SR.json'
    if not os.path.exists(config_path):
        config_path = 'benchmark_config.json'

with open(config_path, 'r', encoding='utf-8') as f:
    config = json.load(f)

vocab_list = config['vocabulary']
word2idx = {w: i for i, w in enumerate(vocab_list)}
bert_emb = np.load('app/models/bert_embeddings.npy')  # Matricea de embeddings BERT

# Hard alignment check — truncation is WRONG and causes 0% Top-1 accuracy.
if len(bert_emb) != len(vocab_list):
    print(f"\n[ERROR] Alignment mismatch!")
    print(f"  bert_embeddings.npy : {len(bert_emb)} rows")
    print(f"  Active vocabulary   : {len(vocab_list)} words  (from {config_path})")
    print(f"\n  Fix: regenerate bert_embeddings.npy for the active vocabulary:")
    print(f"       python regenerate_bert_embeddings.py")
    print(f"  Then re-run this script.\n")
    import sys; sys.exit(1)

bert_t = torch.tensor(bert_emb, dtype=torch.float32).to(device)
print(f"Loaded Vocabulary Size: {len(vocab_list)}")
print(f"Aligned BERT Embeddings Shape: {bert_emb.shape}")

# ── 3. Filtrare date pentru cuvintele aflate in noul vocabular ──────────────────
train_mask = np.array([w in word2idx for w in y_train_words])
val_mask = np.array([w in word2idx for w in y_val_words])
test_mask = np.array([w in word2idx for w in y_test_words])

X_train, y_train_words = X_train[train_mask], y_train_words[train_mask]
X_val, y_val_words = X_val[val_mask], y_val_words[val_mask]
X_test, y_test_words = X_test[test_mask], y_test_words[test_mask]

print(f'Filtered Trial Counts - Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}')

y_train_idx = np.array([word2idx[w] for w in y_train_words])
y_val_idx = np.array([word2idx[w] for w in y_val_words])
y_test_idx = np.array([word2idx[w] for w in y_test_words])


# ── 4. Arhitectura EEG-Conformer (Spatio-Temporal CNN + Transformer Encoder) ──
class EEGConformer(nn.Module):
    def __init__(self, n_channels=105, n_times=500, embed_dim=128):
        super().__init__()
        # 1. Modul Convolutional
        self.temporal_conv = nn.Conv2d(1, 40, (1, 25), stride=(1, 1), padding=(0, 12), bias=False)
        self.bn1 = nn.BatchNorm2d(40)
        self.spatial_conv = nn.Conv2d(40, 40, (n_channels, 1), bias=False)
        self.bn2 = nn.BatchNorm2d(40)
        self.elu = nn.ELU()

        # Sub-sampling temporal prin pooling
        self.pool = nn.AvgPool2d((1, 4))

        # 2. Modul Transformer Encoder
        seq_len = n_times // 4
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=40,
            nhead=4,
            dim_feedforward=128,
            dropout=0.3,      # increased from 0.1 — reduces subject overfitting
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)

        # 3. Strat final de proiectie a embedding-ului EEG
        self.fc = nn.Linear(40 * seq_len, embed_dim)

    def forward(self, x):
        # x: (Batch, 1, Channels, Time_Samples)
        x = self.elu(self.bn1(self.temporal_conv(x)))
        x = self.elu(self.bn2(self.spatial_conv(x)))
        x = self.pool(x)  # Shape: (Batch, 40, 1, seq_len)
        x = x.squeeze(2)  # Shape: (Batch, 40, seq_len)
        x = x.transpose(1, 2)  # Shape: (Batch, seq_len, 40)

        x = self.transformer(x)  # Shape: (Batch, seq_len, 40)
        x = x.reshape(x.size(0), -1)  # Flatten
        e = self.fc(x)  # Shape: (Batch, 128)
        return e


# Initializare Conformer
n_samples_actual = X_train.shape[-1]
eeg_model = EEGConformer(n_channels=105, n_times=n_samples_actual, embed_dim=128).to(device)

# Incarcare optionala a ponderilor
conformer_path = 'app/models/eeg_conformer.pt'
if os.path.exists(conformer_path):
    try:
        eeg_model.load_state_dict(torch.load(conformer_path, map_location=device))
        print(f"Loaded pre-trained EEG-Conformer weights from: {conformer_path}")
    except Exception as e:
        print(f"Could not load Conformer weights: {e}. Fine-tuning initialized weights.")
else:
    print(f"No weights file found at {conformer_path}. Training Conformer from scratch.")


# ── 5. Cap de Proiectie Non-Liniar (MLP) ──────────────────────────────────────
class NonLinearProjectionHead(nn.Module):
    def __init__(self, input_dim=128, output_dim=768, hidden_dim=256, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            self.drop_layer(dropout),
            nn.Linear(hidden_dim, output_dim)
        )

    def drop_layer(self, p):
        # Prevent errors if running empty dropout
        return nn.Dropout(p) if p > 0 else nn.Identity()

    def forward(self, x):
        return self.net(x)


proj = NonLinearProjectionHead(input_dim=128, output_dim=768, hidden_dim=256, dropout=0.3).to(device)


# ── 6. Modul Contrastiv InfoNCE Loss ─────────────────────────────────────────
class InfoNCELoss(nn.Module):
    """
    Forțează perechile pozitive (EEG_i, BERT_i) să aibă o similaritate mare,
    împingând activ EEG_i departe de celelalte embedding-uri din vocabular (negative).
    """

    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, projected_eeg, positive_bert, all_bert_vocab):
        # Normalizam vectorii pentru a calcula similaritatea cosinus
        p_norm = F.normalize(projected_eeg, dim=1)  # (Batch, 768)
        pos_norm = F.normalize(positive_bert, dim=1)  # (Batch, 768)
        all_norm = F.normalize(all_bert_vocab, dim=1)  # (VocabSize, 768)

        # Similaritatea perechilor pozitive
        pos_sim = torch.sum(p_norm * pos_norm, dim=1) / self.temperature  # (Batch)

        # Similaritatea cu toate cuvintele din vocabular (toate sunt negative)
        all_sims = torch.matmul(p_norm, all_norm.T) / self.temperature  # (Batch, VocabSize)

        # InfoNCE: -pos + log(sum(exp(all)))
        loss = -pos_sim + torch.logsumexp(all_sims, dim=1)
        return loss.mean()


# ── 7. Dataset End-to-End & Configurare Optimizator Joint ────────────────────
# Incarcam direct semnalele EEG in Dataset (nu embeddings pre-extrase)
train_ds = TensorDataset(
    torch.tensor(X_train[:, np.newaxis, :, :], dtype=torch.float32),
    torch.tensor(y_train_idx, dtype=torch.long)
)
train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)

# Keep val/test as CPU tensors — run in batches to avoid OOM on 8GB GPU
val_x_cpu  = torch.tensor(X_val[:, np.newaxis, :, :],  dtype=torch.float32)
val_y_cpu  = torch.tensor(y_val_idx,  dtype=torch.long)
test_x_cpu = torch.tensor(X_test[:, np.newaxis, :, :], dtype=torch.float32)

VAL_BATCH = 64   # fits comfortably on 8GB GPU

def _batched_loss(model, proj, x_cpu, y_cpu, criterion, bert_t, batch_size=VAL_BATCH):
    """Run inference + InfoNCE loss in mini-batches to avoid OOM."""
    total_loss = 0.0
    n_batches  = 0
    for start in range(0, len(x_cpu), batch_size):
        xb = x_cpu[start:start+batch_size].to(device)
        yb = y_cpu[start:start+batch_size].to(device)
        with torch.no_grad():
            emb  = model(xb)
            proj_out = proj(emb)
            loss = criterion(proj_out, bert_t[yb], bert_t)
        total_loss += loss.item()
        n_batches  += 1
    return total_loss / max(n_batches, 1)

def _batched_embed(model, proj, x_cpu, batch_size=VAL_BATCH):
    """Extract projected embeddings in batches, return as numpy array."""
    out = []
    for start in range(0, len(x_cpu), batch_size):
        xb = x_cpu[start:start+batch_size].to(device)
        with torch.no_grad():
            out.append(proj(model(xb)).cpu().numpy())
    return np.vstack(out)

# Optimiser — encoder LR lower than projection to preserve pretrained features
optimizer = torch.optim.Adam([
    {'params': eeg_model.parameters(), 'lr': 1e-5, 'weight_decay': 1e-4},
    {'params': proj.parameters(),      'lr': 1e-3, 'weight_decay': 1e-3},
])

criterion  = InfoNCELoss(temperature=0.07)
N_EPOCHS   = 100
PATIENCE   = 15       # early stopping patience
NOISE_STD  = 0.05     # Gaussian noise std for augmentation (fraction of signal std)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=N_EPOCHS, eta_min=1e-7
)

import math
random_baseline = math.log(len(vocab_list))
best_val        = float('inf')
patience_count  = 0

print(f'\nAntrenare End-to-End (Conformer + MLP Proiectie) x {N_EPOCHS} epoci...')
print(f'  Random baseline: {random_baseline:.4f} | Patience: {PATIENCE} | Noise std: {NOISE_STD}')

for epoch in range(N_EPOCHS):
    eeg_model.train()
    proj.train()

    tl = 0.0
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()

        # Gaussian noise augmentation — forces model to learn subject-invariant
        # word features rather than memorising subject-specific signal patterns
        xb = xb + NOISE_STD * torch.randn_like(xb)

        eeg_features = eeg_model(xb)
        projected    = proj(eeg_features)
        loss         = criterion(projected, bert_t[yb], bert_t)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(eeg_model.parameters(), max_norm=1.0)
        optimizer.step()
        tl += loss.item()

    scheduler.step()

    # Validation — batched to avoid OOM
    eeg_model.eval()
    proj.eval()
    vl = _batched_loss(eeg_model, proj, val_x_cpu, val_y_cpu, criterion, bert_t)

    avg_train      = tl / len(train_loader)
    above_train    = ">" if avg_train < random_baseline else "<"
    above_val      = ">" if vl        < random_baseline else "<"

    if (epoch + 1) % 5 == 0 or epoch == 0:
        print(
            f'Ep {epoch+1:02d}/{N_EPOCHS} | '
            f'train:{avg_train:.4f}({above_train}rand) | '
            f'val:{vl:.4f}({above_val}rand)'
        )

    if vl < best_val:
        best_val       = vl
        patience_count = 0
        os.makedirs('app/models/magdas', exist_ok=True)
        torch.save(proj.state_dict(), 'app/models/magdas/projection.pt')
        torch.save(eeg_model.state_dict(), 'app/models/magdas/eeg_conformer.pt')
        print(f'  -> Salvat (val={vl:.4f})')
    else:
        patience_count += 1
        if patience_count >= PATIENCE:
            print(f'  Early stopping la epoca {epoch+1} (no improvement for {PATIENCE} epochs).')
            break

print(f'Best validation loss: {best_val:.4f}  (random baseline: {random_baseline:.4f})')

# ── 8. Evaluare finala retrieval ──────────────────────────────────────────────
eeg_model.load_state_dict(torch.load('app/models/magdas/eeg_conformer.pt', map_location=device))
proj.load_state_dict(torch.load('app/models/magdas/projection.pt', map_location=device))
eeg_model.eval()
proj.eval()

proj_test = _batched_embed(eeg_model, proj, test_x_cpu)

# Cautare prin similaritate cosinus in vocabularul aliniat
sims = cosine_similarity(proj_test, bert_emb)
top5 = np.argsort(sims, axis=1)[:, -5:][:, ::-1]

total = len(y_test_words)
t1 = sum(top5[i, 0] == word2idx[y_test_words[i]] for i in range(total))
t5 = sum(word2idx[y_test_words[i]] in top5[i] for i in range(total))
mc = sims[np.arange(total), [word2idx[w] for w in y_test_words]].mean()

print('\n=== REZULTATE END-TO-END CONTEXTUAL RETRIEVAL ===')
print(f'Top-1 : {t1 / total:.3f} ({t1}/{total})  | KPI>10%:  {"OK" if t1 / total > .10 else "NU"}')
print(f'Top-5 : {t5 / total:.3f} ({t5}/{total})  | KPI>30%:  {"OK" if t5 / total > .30 else "NU"}')
print(f'Cosine: {mc:.3f}             | KPI>0.25: {"OK" if mc > .25 else "NU"}')

print('\n=== EXEMPLE DE RETRIEVAL ===')
vocab_idx2word = {i: w for w, i in word2idx.items()}
for i in range(min(12, total)):
    tw = y_test_words[i]
    pw = [vocab_idx2word[idx] for idx in top5[i]]
    print(f'[{"OK" if tw == pw[0] else "--"}] True:{tw:20s} Top5:{pw}')

metrics = {
    'method': 'MSE + Joint InfoNCE End-to-End Fine-Tuning (Conformer + MLP Head)',
    'note': 'Sincronizat cu framework-ul de evaluare al lui Magdas P2 pentru a preveni colapsul',
    'top1': round(t1 / total, 4), 'top5': round(t5 / total, 4),
    'cosine': round(float(mc), 4),
    'top1_count': int(t1), 'top5_count': int(t5), 'total_test': total,
    'kpi_top1_ok': bool(t1 / total > .10),
    'kpi_top5_ok': bool(t5 / total > .30),
    'kpi_cosine_ok': bool(mc > .25),
}
with open('app/models/magdas/metrics.json', 'w') as f:
    json.dump(metrics, f, indent=2)
print('\nMetrici salvate: app/models/magdas/metrics.json')
print('Proiectie salvata: app/models/magdas/projection.pt')
print('Conformer salvat:  app/models/magdas/eeg_conformer.pt')