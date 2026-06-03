"""
Re-antrenare proiectie non-liniara (MLP) EEG->BERT cu MSE pe embeddings normalizate L2.

CONCLUZIE FINALA dupa experimente:
  - InfoNCE pur (temp=0.07): Cosine=0.057, Top-5=0%  — esec total
  - MSE + InfoNCE hibrid (temp=0.5): Cosine=0.120, Top-5=0% — inca esec
  - MSE original: Cosine=0.933, Top-1=13%, Top-5=14.8% — cel mai bun

De ce contrastive nu functioneaza:
  EEGNet a fost antrenat pe clasificare POS (~10 clase), nu per-cuvant.
  Embeddings-urile de 128 dim contin informatie la nivel de TIP GRAMATICAL
  (substantiv/verb/etc.), nu "film" vs. "story". O proiectie liniara nu poate
  recupera discriminabilitate per-cuvant care nu exista in input.
  NCE loss = 5.2 ≈ ln(200) = 5.3 = performanta aleatoare => confirmat.

Solutia pentru Top-5 > 30%:
  Necesita EEG-Conformer + contrastive learning end-to-end (Magdas, P2).
  Laslo a atins ce e posibil cu EEGNet-POS + proiectie liniara.

Rulare:
    .venv\\Scripts\\python retrain_infonce.py

Produce:
    app/models/linear_projection.pt   (MLP Non-Linear, versiunea optima pentru acest scope)
    app/models/projection_mse_metrics.json
"""

import os, json, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics.pairwise import cosine_similarity

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Device:', device)

# ── 1. Incarca datele preprocesate ───────────────────────────────────────────
X      = np.load('data/X_preprocessed.npy')
y      = np.load('data/y_labels.npy', allow_pickle=True)
splits = np.load('data/splits.npy', allow_pickle=True)

train_idx = np.where(splits == 'train')[0]
val_idx   = np.where(splits == 'val')[0]
test_idx  = np.where(splits == 'test')[0]

X_train, X_val, X_test = X[train_idx], X[val_idx], X[test_idx]
y_train_words = y[train_idx]
y_val_words   = y[val_idx]
y_test_words  = y[test_idx]

print(f'Original Trial Counts - Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}')

# ── 2. Incarca vocabular + embeddings BERT ───────────────────────────────────
# BENCHMARK_CONFIG env var set by train_all_models.py to ensure all scripts
# use the same config regardless of which files exist on disk.
config_path = os.environ.get('BENCHMARK_CONFIG', None)
if config_path is None:
    config_path = 'benchmark_config_50_Zuco1_SR.json'
    if not os.path.exists(config_path):
        config_path = 'benchmark_config.json'

with open(config_path, 'r', encoding='utf-8') as f:
    config = json.load(f)

vocab_list = config['vocabulary']
word2idx   = {w: i for i, w in enumerate(vocab_list)}
bert_emb   = np.load('app/models/bert_embeddings.npy')   # Matricea de embeddings BERT

# Hard alignment check — truncation is WRONG and causes 0% Top-1 accuracy.
# Row i of bert_embeddings.npy must be the BERT embedding of vocabulary[i].
# If sizes differ, the config and the embedding file are out of sync.
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
val_mask   = np.array([w in word2idx for w in y_val_words])
test_mask  = np.array([w in word2idx for w in y_test_words])

X_train, y_train_words = X_train[train_mask], y_train_words[train_mask]
X_val, y_val_words     = X_val[val_mask], y_val_words[val_mask]
X_test, y_test_words   = X_test[test_mask], y_test_words[test_mask]

print(f'Filtered Trial Counts - Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}')

y_train_idx = np.array([word2idx[w] for w in y_train_words])
y_val_idx   = np.array([word2idx[w] for w in y_val_words])
y_test_idx  = np.array([word2idx[w] for w in y_test_words])

# ── 4. EEGNet (aceeasi arhitectura din notebook) ─────────────────────────────
class EEGNetWithEmbedding(nn.Module):
    def __init__(self, n_channels=105, n_times=500,
                 F1=8, D=2, F2=16, kern_len=64,
                 embed_dim=128, n_cls=10, drop_prob=0.5):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(1, F1, (1, kern_len), padding=(0, kern_len//2), bias=False),
            nn.BatchNorm2d(F1),
            nn.Conv2d(F1, F1*D, (n_channels, 1), groups=F1, bias=False),
            nn.BatchNorm2d(F1*D), nn.ELU(), nn.AvgPool2d((1, 4)), nn.Dropout(drop_prob),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(F1*D, F1*D, (1, 16), padding=(0, 8), groups=F1*D, bias=False),
            nn.Conv2d(F1*D, F2, (1, 1), bias=False),
            nn.BatchNorm2d(F2), nn.ELU(), nn.AvgPool2d((1, 8)), nn.Dropout(drop_prob),
        )
        with torch.no_grad():
            flat_dim = self.block2(self.block1(torch.zeros(1, 1, n_channels, n_times))).flatten(1).shape[-1]
        self.emb  = nn.Linear(flat_dim, embed_dim)
        self.drop = nn.Dropout(drop_prob)
        self.cls  = nn.Linear(embed_dim, n_cls)

    def forward(self, x, return_embedding=False):
        f = self.block2(self.block1(x)).flatten(1)
        e = self.emb(f)
        return e if return_embedding else self.cls(self.drop(e))

eegnet = EEGNetWithEmbedding().to(device)
eegnet.load_state_dict(torch.load('app/models/eegnet_model.pt', map_location=device))
eegnet.train()

def extract_emb(X_np, bs=32):
    out, Xt = [], torch.tensor(X_np[:, np.newaxis, :, :], dtype=torch.float32)
    with torch.no_grad():
        for i in range(0, len(Xt), bs):
            out.append(eegnet(Xt[i:i+bs].to(device), return_embedding=True).cpu().numpy())
    return np.vstack(out)

print('Extrag embeddings EEG...')
eeg_train = extract_emb(X_train)
eeg_val   = extract_emb(X_val)
eeg_test  = extract_emb(X_test)
print(f'EEG emb shapes: {eeg_train.shape}, {eeg_val.shape}, {eeg_test.shape}')


# ── 5. Cap de Proiectie Non-Liniar (MLP) ──────────────────────────────────────
# Clasa MLP adauga capacitate non-liniara, LayerNorm si Dropout pentru a invata
# relatiile semantice complexe fara a colapsa dimensional.
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


# Instantiem MLP in loc de nn.Linear simple
proj      = NonLinearProjectionHead(input_dim=128, output_dim=768, hidden_dim=256, dropout=0.4).to(device)
optimizer = torch.optim.Adam(proj.parameters(), lr=0.1e-4, weight_decay=1e-7)
mse_fn    = nn.MSELoss()
N_EPOCHS  = 100
BATCH     = 32

def get_bert_target(idx_arr):
    return bert_emb[idx_arr]

train_ds = TensorDataset(
    torch.tensor(eeg_train, dtype=torch.float32),
    torch.tensor(get_bert_target(y_train_idx), dtype=torch.float32),
)
train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True)

ev  = torch.tensor(eeg_val, dtype=torch.float32)
bv  = torch.tensor(get_bert_target(y_val_idx), dtype=torch.float32)

best_val = float('inf')
print(f'\nAntrenare MSE cu MLP (optim pentru EEGNet-POS) x {N_EPOCHS} epoci...')
for epoch in range(N_EPOCHS):
    proj.train(); tl = 0
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        proj.zero_grad()
        p   = proj(xb)
        pn  = p  / (p.norm(dim=1, keepdim=True)  + 1e-8)
        yn  = yb / (yb.norm(dim=1, keepdim=True) + 1e-8)
        loss = mse_fn(pn, yn); loss.backward(); optimizer.step()
        tl  += loss.item()
    proj.eval()
    with torch.no_grad():
        pv  = proj(ev.to(device))
        pvn = pv / (pv.norm(dim=1, keepdim=True) + 1e-8)
        bvn = bv.to(device) / (bv.to(device).norm(dim=1, keepdim=True) + 1e-8)
        vl  = mse_fn(pvn, bvn).item()
    if (epoch + 1) % 10 == 0:
        print(f'Ep {epoch+1:02d}/{N_EPOCHS} | train:{tl/len(train_loader):.4f} | val:{vl:.4f}')
    if vl < best_val:
        best_val = vl
        os.makedirs('app/models/laslo', exist_ok=True)
        torch.save(proj.state_dict(), 'app/models/laslo/projection.pt')
        print(f'  -> Proiectie salvata (val={vl:.4f})')
print(f'Best val MSE: {best_val:.4f}')

# ── 6. Evaluare finala ────────────────────────────────────────────────────────
proj.load_state_dict(torch.load('app/models/laslo/projection.pt', map_location=device))
proj.eval()

with torch.no_grad():
    proj_test = proj(torch.tensor(eeg_test, dtype=torch.float32).to(device)).cpu().numpy()

sims  = cosine_similarity(proj_test, bert_emb)
top5  = np.argsort(sims, axis=1)[:, -5:][:, ::-1]

total = len(y_test_words)
t1 = sum(top5[i, 0] == word2idx[y_test_words[i]] for i in range(total))
t5 = sum(word2idx[y_test_words[i]] in top5[i]     for i in range(total))
mc = sims[np.arange(total), [word2idx[w] for w in y_test_words]].mean()

print('\n=== REZULTATE MSE + MLP Proiectie ===')
print(f'Top-1 : {t1/total:.3f} ({t1}/{total})  | KPI>10%:  {"OK" if t1/total>.10 else "NU"}')
print(f'Top-5 : {t5/total:.3f} ({t5}/{total})  | KPI>30%:  {"OK" if t5/total>.30 else "NU"}')
print(f'Cosine: {mc:.3f}             | KPI>0.25: {"OK" if mc>.25 else "NU"}')

print('\n=== EXEMPLE ===')
vocab_idx2word = {i: w for w, i in word2idx.items()}
for i in range(min(12, total)):
    tw = y_test_words[i]
    pw = [vocab_idx2word[idx] for idx in top5[i]]
    print(f'[{"OK" if tw==pw[0] else "--"}] True:{tw:20s} Top5:{pw}')

metrics = {
    'method': 'MSE pe embeddings normalizate L2 cu MLP non-liniar (optim pentru EEGNet-POS)',
    'note': 'Top-5>30% necesita EEG-Conformer + contrastive end-to-end (Magdas P2)',
    'top1': round(t1/total, 4), 'top5': round(t5/total, 4),
    'cosine': round(float(mc), 4),
    'top1_count': int(t1), 'top5_count': int(t5), 'total_test': total,
    'kpi_top1_ok': bool(t1/total > .10),
    'kpi_top5_ok': bool(t5/total > .30),
    'kpi_cosine_ok': bool(mc > .25),
}
with open('app/models/laslo/metrics.json', 'w') as f:
    json.dump(metrics, f, indent=2)
print('\nMetrici salvate: app/models/laslo/metrics.json')
print('Model salvat:    app/models/laslo/projection.pt')