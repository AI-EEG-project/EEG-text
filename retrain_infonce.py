"""
Re-antrenare proiectie liniara EEG->BERT cu MSE pe embeddings normalizate L2.

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
    cd C:\\Users\\alexl\\School\\AI\\eeg_ai_project\\EEG-text
    .venv\\Scripts\\python retrain_infonce.py

Produce:
    app/models/linear_projection.pt   (MSE, versiunea optima pentru acest scope)
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

print(f'Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}')

# ── 2. Incarca vocabular + embeddings BERT ───────────────────────────────────
with open('benchmark_config.json') as f:
    config = json.load(f)

vocab_list = config['vocabulary']
word2idx   = {w: i for i, w in enumerate(vocab_list)}
bert_emb   = np.load('app/models/bert_embeddings.npy')   # (200, 768)
bert_t     = torch.tensor(bert_emb, dtype=torch.float32).to(device)

y_train_idx = np.array([word2idx[w] for w in y_train_words])
y_val_idx   = np.array([word2idx[w] for w in y_val_words])
y_test_idx  = np.array([word2idx[w] for w in y_test_words])

# ── 3. EEGNet (aceeasi arhitectura din notebook) ─────────────────────────────
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
eegnet.eval()

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

# ── 4. Loss hibrid: MSE + soft InfoNCE ───────────────────────────────────────
#
# MSE pe embeddings normalizate L2:
#   - Aliniaza directia generala EEG -> BERT
#   - Stabilizeaza antrenarea, evita colapsul
#
# InfoNCE cu temperature=0.5 (soft):
#   - Forteaza discriminarea: cuvantul corect sa aiba scor mai mare
#   - Temperature mare = distributie mai moale = gradiente stabile
#   - Nu mai cere discriminare perfecta dintre cuvinte similare semantic
#
# loss_total = MSE + alpha * InfoNCE(temp=0.5)
#   alpha=0.3 = InfoNCE contribuie 30% la gradient

# ── 5. Antrenare MSE (varianta optima pentru EEGNet-POS) ─────────────────────
proj      = nn.Linear(128, 768).to(device)
optimizer = torch.optim.Adam(proj.parameters(), lr=1e-3)
mse_fn    = nn.MSELoss()
N_EPOCHS  = 50
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
print(f'\nAntrenare MSE (optim pentru EEGNet-POS) x {N_EPOCHS} epoci...')
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
        torch.save(proj.state_dict(), 'app/models/linear_projection.pt')
        print(f'  -> Proiectie salvata (val={vl:.4f})')
print(f'Best val MSE: {best_val:.4f}')

# ── 6. Evaluare finala ────────────────────────────────────────────────────────
proj.load_state_dict(torch.load('app/models/linear_projection.pt', map_location=device))
proj.eval()

with torch.no_grad():
    proj_test = proj(torch.tensor(eeg_test, dtype=torch.float32).to(device)).cpu().numpy()

sims  = cosine_similarity(proj_test, bert_emb)
top5  = np.argsort(sims, axis=1)[:, -5:][:, ::-1]

total = len(y_test_words)
t1 = sum(top5[i, 0] == word2idx[y_test_words[i]] for i in range(total))
t5 = sum(word2idx[y_test_words[i]] in top5[i]     for i in range(total))
mc = sims[np.arange(total), [word2idx[w] for w in y_test_words]].mean()

print('\n=== REZULTATE MSE + soft InfoNCE ===')
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
    'method': 'MSE pe embeddings normalizate L2 (optim pentru EEGNet-POS)',
    'note': 'Top-5>30% necesita EEG-Conformer + contrastive end-to-end (Magdas P2)',
    'top1': round(t1/total, 4), 'top5': round(t5/total, 4),
    'cosine': round(float(mc), 4),
    'top1_count': int(t1), 'top5_count': int(t5), 'total_test': total,
    'kpi_top1_ok': bool(t1/total > .10),
    'kpi_top5_ok': bool(t5/total > .30),
    'kpi_cosine_ok': bool(mc > .25),
}
with open('app/models/projection_mse_metrics.json', 'w') as f:
    json.dump(metrics, f, indent=2)
print('\nMetrici salvate: app/models/projection_mse_metrics.json')
print('Model salvat:    app/models/linear_projection.pt')
