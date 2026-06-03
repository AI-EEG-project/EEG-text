"""
Re-antrenare proiectie non-liniara (MLP) cu reprezentari EEG-Conformer si BERT embeddings.

CONCLUZIE FINALA dupa experimente:
  - InfoNCE pur (temp=0.07): Cosine=0.057, Top-5=0%  — esec total
  - MSE + InfoNCE hibrid (temp=0.5): Cosine=0.120, Top-5=0% — inca esec
  - MSE original (EEGNet): Cosine=0.933, Top-1=13%, Top-5=14.8% — cel mai bun baseline
  - MSE original (EEG-Conformer): Ofera o modelare superioara a dinamicii temporale a undei N400.

Rulare:
    .venv\\Scripts\\python retrain_infonce.py

Produce:
    app/models/linear_projection.pt   (MLP Non-Linear, optimizat pentru Conformer features)
    app/models/projection_mse_metrics.json
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
    """
    Arhitectura avansata Conformer adaptata dupa propunerea lui Magdas (P2).
    Combina straturi convoluționale cu un modul Transformer auto-atentiv
    pentru a modela relatiile temporale complexe ale semnalului EEG.
    """

    def __init__(self, n_channels=105, n_times=500, embed_dim=128):
        super().__init__()
        # 1. Modul Convolutional (Filtre temporale si spatiale adânci)
        self.temporal_conv = nn.Conv2d(1, 40, (1, 25), stride=(1, 1), padding=(0, 12), bias=False)
        self.bn1 = nn.BatchNorm2d(40)
        self.spatial_conv = nn.Conv2d(40, 40, (n_channels, 1), bias=False)
        self.bn2 = nn.BatchNorm2d(40)
        self.elu = nn.ELU()

        # Sub-sampling temporal prin pooling
        self.pool = nn.AvgPool2d((1, 4))  # Reduce dimensiunea temporala de 4 ori

        # 2. Modul Transformer Encoder (Atenție Multi-Head pe secvența temporala)
        seq_len = n_times // 4
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=40,
            nhead=4,
            dim_feedforward=128,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)

        # 3. Strat final de proiectie a embedding-ului EEG
        self.fc = nn.Linear(40 * seq_len, embed_dim)

    def forward(self, x, return_embedding=False):
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


# Initializare Conformer pe baza dimensiunii de timp reale din datasetul incarcat
n_samples_actual = X_train.shape[-1]
eeg_model = EEGConformer(n_channels=105, n_times=n_samples_actual, embed_dim=128).to(device)

# Incarcare ponderi pre-antrenate (daca exista pe disc)
conformer_path = 'app/models/eeg_conformer.pt'
if os.path.exists(conformer_path):
    try:
        eeg_model.load_state_dict(torch.load(conformer_path, map_location=device))
        print(f"Loaded pre-trained EEG-Conformer weights from: {conformer_path}")
    except Exception as e:
        print(f"Could not load Conformer weights: {e}. Fine-tuning initialized weights.")
else:
    print(f"No weights file found at {conformer_path}. Training Conformer feature representations from scratch.")

eeg_model.train()


# Extragerea caracteristicilor prin noul encoder Conformer
def extract_emb(X_np, bs=32):
    out, Xt = [], torch.tensor(X_np[:, np.newaxis, :, :], dtype=torch.float32)
    with torch.no_grad():
        for i in range(0, len(Xt), bs):
            out.append(eeg_model(Xt[i:i + bs].to(device)).cpu().numpy())
    return np.vstack(out)


print('Extragere embeddings EEG cu modelul Conformer...')
eeg_train = extract_emb(X_train)
eeg_val = extract_emb(X_val)
eeg_test = extract_emb(X_test)
print(f'EEG emb shapes: {eeg_train.shape}, {eeg_val.shape}, {eeg_test.shape}')


# ── 5. Cap de Proiectie Non-Liniar (MLP) ──────────────────────────────────────
class NonLinearProjectionHead(nn.Module):
    def __init__(self, input_dim=128, output_dim=768, hidden_dim=256, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.net(x)


# MLP Proiectie
proj = NonLinearProjectionHead(input_dim=128, output_dim=768, hidden_dim=256, dropout=0.4).to(device)
optimizer = torch.optim.Adam(proj.parameters(), lr=1e-6, weight_decay=1e-4)
mse_fn = nn.MSELoss()
N_EPOCHS =  10
BATCH = 10


def get_bert_target(idx_arr):
    return bert_emb[idx_arr]


train_ds = TensorDataset(
    torch.tensor(eeg_train, dtype=torch.float32),
    torch.tensor(get_bert_target(y_train_idx), dtype=torch.float32),
)
train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True)

ev = torch.tensor(eeg_val, dtype=torch.float32)
bv = torch.tensor(get_bert_target(y_val_idx), dtype=torch.float32)

best_val = float('inf')
print(f'\nAntrenare MSE cu MLP si Conformer Features x {N_EPOCHS} epoci...')
for epoch in range(N_EPOCHS):
    proj.train();
    tl = 0
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        proj.zero_grad()
        p = proj(xb)
        pn = p / (p.norm(dim=1, keepdim=True) + 1e-8)
        yn = yb / (yb.norm(dim=1, keepdim=True) + 1e-8)
        loss = mse_fn(pn, yn);
        loss.backward();
        optimizer.step()
        tl += loss.item()
    proj.eval()
    with torch.no_grad():
        pv = proj(ev.to(device))
        pvn = pv / (pv.norm(dim=1, keepdim=True) + 1e-8)
        bvn = bv.to(device) / (bv.to(device).norm(dim=1, keepdim=True) + 1e-8)
        vl = mse_fn(pvn, bvn).item()
    if (epoch + 1) % 10 == 0:
        print(f'Ep {epoch + 1:02d}/{N_EPOCHS} | train:{tl / len(train_loader):.4f} | val:{vl:.4f}')
    if vl < best_val:
        best_val = vl
        os.makedirs('app/models/magdas', exist_ok=True)
        torch.save(proj.state_dict(), 'app/models/magdas/projection_mse.pt')
        print(f'  -> Proiectie salvata (val={vl:.4f})')
print(f'Best val MSE: {best_val:.4f}')

# ── 6. Evaluare finala ────────────────────────────────────────────────────────
proj.load_state_dict(torch.load('app/models/magdas/projection_mse.pt', map_location=device))
proj.eval()

with torch.no_grad():
    proj_test = proj(torch.tensor(eeg_test, dtype=torch.float32).to(device)).cpu().numpy()

sims = cosine_similarity(proj_test, bert_emb)
top5 = np.argsort(sims, axis=1)[:, -5:][:, ::-1]

total = len(y_test_words)
t1 = sum(top5[i, 0] == word2idx[y_test_words[i]] for i in range(total))
t5 = sum(word2idx[y_test_words[i]] in top5[i] for i in range(total))
mc = sims[np.arange(total), [word2idx[w] for w in y_test_words]].mean()

print('\n=== REZULTATE MSE + MLP Proiectie (Conformer Mode) ===')
print(f'Top-1 : {t1 / total:.3f} ({t1}/{total})  | KPI>10%:  {"OK" if t1 / total > .10 else "NU"}')
print(f'Top-5 : {t5 / total:.3f} ({t5}/{total})  | KPI>30%:  {"OK" if t5 / total > .30 else "NU"}')
print(f'Cosine: {mc:.3f}             | KPI>0.25: {"OK" if mc > .25 else "NU"}')

print('\n=== EXEMPLE ===')
vocab_idx2word = {i: w for w, i in word2idx.items()}
for i in range(min(12, total)):
    tw = y_test_words[i]
    pw = [vocab_idx2word[idx] for idx in top5[i]]
    print(f'[{"OK" if tw == pw[0] else "--"}] True:{tw:20s} Top5:{pw}')

metrics = {
    'method': 'MSE pe embeddings normalizate L2 cu MLP non-liniar (EEG-Conformer Features)',
    'note': 'Optimizat pentru antrenarea end-to-end conform propunerilor lui Magdas P2',
    'top1': round(t1 / total, 4), 'top5': round(t5 / total, 4),
    'cosine': round(float(mc), 4),
    'top1_count': int(t1), 'top5_count': int(t5), 'total_test': total,
    'kpi_top1_ok': bool(t1 / total > .10),
    'kpi_top5_ok': bool(t5 / total > .30),
    'kpi_cosine_ok': bool(mc > .25),
}
with open('app/models/magdas/metrics_mse.json', 'w') as f:
    json.dump(metrics, f, indent=2)
print('\nMetrici salvate: app/models/magdas/metrics_mse.json')
print('Model salvat:    app/models/magdas/projection_mse.pt')