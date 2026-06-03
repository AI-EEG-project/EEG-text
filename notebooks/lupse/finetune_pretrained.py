"""
finetune_pretrained.py — Fine-tune a pre-trained EEG-to-text encoder on the ZuCo vocabulary
Lupse Ioan Victor — Sapt. 14

What "pretrained" means here
-----------------------------
Unlike Laslo's EEGNet (POS classifier) and Magdas's fresh Conformer, this script starts
from a backbone already trained on semantic EEG-text alignment, then fine-tunes it on our
specific vocabulary.

Backbone priority order:
  1. --pretrained-path <file>  — any .pt checkpoint you supply manually
                                 (e.g. weights downloaded from a paper's GitHub release)
  2. Magdas's eeg_conformer.pt — trained on ZuCo 1.0 SR with InfoNCE; this IS the best
                                 semantics-trained EEG model available in this project
  3. Random-init EEGConformer  — fallback if nothing else is found

Note on HuggingFace
-------------------
There are no widely released pretrained EEG-to-text checkpoints on HuggingFace as of 2024.
The two repos that looked promising:
  - MikeWangWZHL/EEG-To-Text  (Wang et al. 2022) — weights must be requested from the
    authors via their GitHub: https://github.com/MikeWangWZHL/EEG-To-Text
  - NZAE/EEG-Conformer         — motor imagery, not semantic reading

To use a manually downloaded checkpoint:
    python notebooks/lupse/finetune_pretrained.py --pretrained-path path/to/weights.pt

Fine-tuning strategy (corrected from v1)
-----------------------------------------
  - Encoder LR : 1e-5  (same as Magdas — conservative but not frozen)
  - Projection LR: 1e-4 (lower than before to prevent overfitting on small val set)
  - InfoNCE temperature: 0.07
  - CosineAnnealingLR + early stopping (patience 20)
  - Random baseline printed at start so you immediately see if training is above chance

Outputs  (never overwrites Laslo or Magdas files)
  app/models/lupse/encoder.pt
  app/models/lupse/projection.pt
  app/models/lupse/metrics.json

Run from repo root:
    .venv\\Scripts\\python notebooks/lupse/finetune_pretrained.py
    .venv\\Scripts\\python notebooks/lupse/finetune_pretrained.py --pretrained-path weights.pt
"""

import argparse
import os
import sys
import json
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics.pairwise import cosine_similarity

# ── CLI ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Fine-tune pretrained EEG-to-text encoder.")
parser.add_argument(
    "--pretrained-path", default=None,
    help="Path to a .pt checkpoint to use as the encoder backbone (optional).",
)
parser.add_argument(
    "--epochs", type=int, default=80,
    help="Maximum training epochs (default: 80).",
)
parser.add_argument(
    "--batch", type=int, default=32,
    help="Batch size (default: 32).",
)
args = parser.parse_args()

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT        = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR    = os.path.join(ROOT, "data")
MODELS_ROOT = os.path.join(ROOT, "app", "models")
OUT_DIR     = os.path.join(MODELS_ROOT, "lupse")
os.makedirs(OUT_DIR, exist_ok=True)

OUT_ENCODER = os.path.join(OUT_DIR, "encoder.pt")
OUT_PROJ    = os.path.join(OUT_DIR, "projection.pt")
OUT_METRICS = os.path.join(OUT_DIR, "metrics.json")

BERT_EMB_PATH = os.path.join(MODELS_ROOT, "bert_embeddings.npy")

# Magdas's conformer — best available semantics-trained backbone
MAGDAS_CONFORMER = os.path.join(MODELS_ROOT, "magdas", "eeg_conformer.pt")
if not os.path.exists(MAGDAS_CONFORMER):
    MAGDAS_CONFORMER = os.path.join(MODELS_ROOT, "eeg_conformer.pt")

# ── Device ────────────────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# ── 1. Load preprocessed data ─────────────────────────────────────────────────
print("\n[1] Loading preprocessed data...")
X      = np.load(os.path.join(DATA_DIR, "X_preprocessed.npy"))
y      = np.load(os.path.join(DATA_DIR, "y_labels.npy"), allow_pickle=True)
splits = np.load(os.path.join(DATA_DIR, "splits.npy"),   allow_pickle=True)

train_idx = np.where(splits == "train")[0]
val_idx   = np.where(splits == "val")[0]
test_idx  = np.where(splits == "test")[0]

X_train, X_val, X_test           = X[train_idx], X[val_idx], X[test_idx]
y_train_words, y_val_words, y_test_words = y[train_idx], y[val_idx], y[test_idx]

n_channels = X_train.shape[1]
n_times    = X_train.shape[2]
print(f"  Trials — train: {len(X_train)}, val: {len(X_val)}, test: {len(X_test)}")
print(f"  EEG shape per trial: ({n_channels}, {n_times})")

# ── 2. Vocabulary and BERT embeddings ─────────────────────────────────────────
print("\n[2] Loading vocabulary and BERT embeddings...")
config_path = os.environ.get('BENCHMARK_CONFIG', None)
if config_path is None:
    config_path = os.path.join(ROOT, "benchmark_config_50_Zuco1_SR.json")
    if not os.path.exists(config_path):
        config_path = os.path.join(ROOT, "benchmark_config.json")

with open(config_path, "r", encoding="utf-8") as f:
    cfg = json.load(f)

vocab_list = cfg["vocabulary"]
word2idx   = {w: i for i, w in enumerate(vocab_list)}
bert_emb   = np.load(BERT_EMB_PATH)

if len(bert_emb) != len(vocab_list):
    print(f"\n[ERROR] Alignment mismatch!")
    print(f"  bert_embeddings.npy : {len(bert_emb)} rows")
    print(f"  Active vocabulary   : {len(vocab_list)} words  (from {config_path})")
    print(f"\n  Fix: python regenerate_bert_embeddings.py")
    sys.exit(1)

bert_t = torch.tensor(bert_emb, dtype=torch.float32).to(device)

# Random baseline: -log(1/N) = log(N)
random_baseline = math.log(len(vocab_list))
print(f"  Vocabulary: {len(vocab_list)} words  |  BERT matrix: {bert_emb.shape}")
print(f"  InfoNCE random baseline loss = ln({len(vocab_list)}) = {random_baseline:.4f}")
print(f"  Training must go BELOW {random_baseline:.4f} to be above chance.")

def _filter(X_np, words):
    mask    = np.array([w in word2idx for w in words])
    words_f = words[mask]
    return X_np[mask], words_f, np.array([word2idx[w] for w in words_f])

X_train, y_train_words, y_train_idx = _filter(X_train, y_train_words)
X_val,   y_val_words,   y_val_idx   = _filter(X_val,   y_val_words)
X_test,  y_test_words,  y_test_idx  = _filter(X_test,  y_test_words)
print(f"  Filtered — train: {len(X_train)}, val: {len(X_val)}, test: {len(X_test)}")

# ── 3. Model architecture ─────────────────────────────────────────────────────
EMBED_DIM = 128

class EEGConformer(nn.Module):
    """Identical to Magdas's architecture so pretrained weights load cleanly."""
    def __init__(self, n_channels=105, n_times=500, embed_dim=128):
        super().__init__()
        self.temporal_conv = nn.Conv2d(1, 40, (1, 25), padding=(0, 12), bias=False)
        self.bn1            = nn.BatchNorm2d(40)
        self.spatial_conv   = nn.Conv2d(40, 40, (n_channels, 1), bias=False)
        self.bn2            = nn.BatchNorm2d(40)
        self.elu            = nn.ELU()
        self.pool           = nn.AvgPool2d((1, 4))
        seq_len             = n_times // 4
        encoder_layer       = nn.TransformerEncoderLayer(
            d_model=40, nhead=4, dim_feedforward=128, dropout=0.1, batch_first=True
        )
        self.transformer    = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.fc             = nn.Linear(40 * seq_len, embed_dim)

    def forward(self, x):
        x = self.elu(self.bn1(self.temporal_conv(x)))
        x = self.elu(self.bn2(self.spatial_conv(x)))
        x = self.pool(x).squeeze(2).transpose(1, 2)
        x = self.transformer(x)
        return self.fc(x.reshape(x.size(0), -1))


class NonLinearProjectionHead(nn.Module):
    """128d -> 256d -> 768d. Matches all other retrain scripts."""
    def __init__(self, input_dim=128, output_dim=768, hidden_dim=150, dropout=0.8):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )
    def forward(self, x):
        return self.net(x)


# ── 4. Load pretrained backbone ───────────────────────────────────────────────
print("\n[3] Loading pretrained EEG-to-text backbone...")

encoder       = EEGConformer(n_channels=n_channels, n_times=n_times, embed_dim=EMBED_DIM).to(device)
backbone_name = None

def _try_load(path: str, strict: bool = True) -> bool:
    """Attempt to load weights into encoder. Returns True on success."""
    try:
        state = torch.load(path, map_location="cpu", weights_only=False)
        encoder.load_state_dict(state, strict=strict)
        return True
    except Exception as e:
        print(f"  [WARN] Could not load {path}: {e}")
        return False

# --- Priority 1: manually supplied checkpoint ─────────────────────────────────
if args.pretrained_path:
    path = args.pretrained_path
    if not os.path.isabs(path):
        path = os.path.join(ROOT, path)
    if os.path.exists(path):
        if _try_load(path, strict=False):
            backbone_name = f"Manual checkpoint: {path}"
            print(f"  Loaded manual checkpoint: {path}")
        else:
            print(f"  Manual checkpoint failed to load — continuing with fallback.")
    else:
        print(f"  [ERROR] --pretrained-path not found: {path}")
        print(f"  Continuing with fallback backbone.")

# --- Priority 2: Magdas's EEG-Conformer ──────────────────────────────────────
# This is trained on ZuCo 1.0 SR with InfoNCE — the most appropriate
# semantics-trained EEG backbone available in this project.
if backbone_name is None:
    if os.path.exists(MAGDAS_CONFORMER):
        if _try_load(MAGDAS_CONFORMER, strict=True):
            backbone_name = f"Magdas EEG-Conformer (ZuCo 1.0 SR, InfoNCE)"
            print(f"  Loaded Magdas conformer from: {MAGDAS_CONFORMER}")
            print(f"  Semantics-trained on ZuCo reading — best available pretrained backbone.")
        else:
            print(f"  Magdas weights incompatible (architecture mismatch?) — using random init.")
    else:
        print(f"  Magdas conformer not found at: {MAGDAS_CONFORMER}")
        print(f"  Run Magdas training first (train_all_models.py --skip-laslo --skip-lupse).")

# --- Priority 3: random init ──────────────────────────────────────────────────
if backbone_name is None:
    backbone_name = "EEGConformer (random init — no pretrained weights found)"
    print(f"  Using random init. Results will likely be below Magdas's model.")

print(f"\n  Backbone: {backbone_name}")

# ── 5. Projection head ────────────────────────────────────────────────────────
proj = NonLinearProjectionHead(input_dim=EMBED_DIM, output_dim=768).to(device)

# ── 6. InfoNCE loss ───────────────────────────────────────────────────────────
class InfoNCELoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, projected_eeg, positive_bert, all_bert_vocab):
        p_norm   = F.normalize(projected_eeg,  dim=1)
        pos_norm = F.normalize(positive_bert,  dim=1)
        all_norm = F.normalize(all_bert_vocab, dim=1)
        pos_sim  = torch.sum(p_norm * pos_norm, dim=1) / self.temperature
        all_sims = torch.matmul(p_norm, all_norm.T) / self.temperature
        return (-pos_sim + torch.logsumexp(all_sims, dim=1)).mean()

criterion = InfoNCELoss(temperature=0.07)

# ── 7. Optimiser ─────────────────────────────────────────────────────────────
# Encoder LR 1e-5: same as Magdas — conservative enough to preserve features,
#                  high enough to actually adapt (5e-7 was too low, encoder stayed frozen).
# Projection LR 1e-4: lower than before (5e-4 caused projection to overfit on small val set).
ENCODER_LR = 5e-6
PROJ_LR    = 5e-4
NOISE_STD  = 0.1   # Gaussian noise augmentation — same as Magdas

optimizer = torch.optim.Adam([
    {"params": encoder.parameters(), "lr": ENCODER_LR, "weight_decay": 1e-4},
    {"params": proj.parameters(),    "lr": PROJ_LR,    "weight_decay": 5e-2},
])
N_EPOCHS  = args.epochs
BATCH     = args.batch
PATIENCE  = 15
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=N_EPOCHS, eta_min=1e-7)

# ── 8. DataLoaders ────────────────────────────────────────────────────────────
train_ds = TensorDataset(
    torch.tensor(X_train[:, np.newaxis, :, :], dtype=torch.float32),
    torch.tensor(y_train_idx, dtype=torch.long),
)
train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True, drop_last=True)

# Keep on CPU — infer in batches to avoid OOM on 8GB GPU
val_x_cpu = torch.tensor(X_val[:, np.newaxis, :, :], dtype=torch.float32)
val_y_cpu = torch.tensor(y_val_idx, dtype=torch.long)

VAL_BATCH = 32

def _val_loss(encoder, proj, x_cpu, y_cpu, criterion, bert_t):
    total, n = 0.0, 0
    for s in range(0, len(x_cpu), VAL_BATCH):
        xb = x_cpu[s:s+VAL_BATCH].to(device)
        yb = y_cpu[s:s+VAL_BATCH].to(device)
        with torch.no_grad():
            loss = criterion(proj(encoder(xb)), bert_t[yb], bert_t)
        total += loss.item(); n += 1
    return total / max(n, 1)

# ── 9. Training loop ──────────────────────────────────────────────────────────
print(f"\n[4] Fine-tuning")
print(f"  Epochs   : {N_EPOCHS}  |  Batch: {BATCH}  |  Patience: {PATIENCE}")
print(f"  Encoder LR: {ENCODER_LR}  |  Projection LR: {PROJ_LR}  |  Noise std: {NOISE_STD}")
print(f"  Random baseline: {random_baseline:.4f}  — train loss must go below this\n")

best_val_loss  = float("inf")
patience_count = 0

for epoch in range(N_EPOCHS):
    encoder.eval(); proj.train()
    total_loss = 0.0

    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        # Gaussian noise augmentation — same strategy as Magdas training
        xb = xb + NOISE_STD * torch.randn_like(xb)
        loss = criterion(proj(encoder(xb)), bert_t[yb], bert_t)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(encoder.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()

    scheduler.step()

    encoder.eval(); proj.train()
    val_loss = _val_loss(encoder, proj, val_x_cpu, val_y_cpu, criterion, bert_t)

    avg_train = total_loss / len(train_loader)
    above_chance_train = "above-chance" if avg_train < random_baseline else "at/below-chance"
    above_chance_val   = "above-chance" if val_loss  < random_baseline else "at/below-chance"

    if (epoch + 1) % 10 == 0 or epoch == 0:
        print(
            f"  Ep {epoch+1:02d}/{N_EPOCHS} | "
            f"train: {avg_train:.4f} ({above_chance_train}) | "
            f"val: {val_loss:.4f} ({above_chance_val})"
        )

    if val_loss < best_val_loss:
        best_val_loss  = val_loss
        patience_count = 0
        torch.save(encoder.state_dict(), OUT_ENCODER)
        torch.save(proj.state_dict(), OUT_PROJ)
        print(f"  -> Saved (val={val_loss:.4f})")
    else:
        patience_count += 1
        if patience_count >= PATIENCE:
            print(f"  Early stopping at epoch {epoch+1} (no improvement for {PATIENCE} epochs).")
            break

print(f"\nBest validation loss : {best_val_loss:.4f}  (random baseline: {random_baseline:.4f})")
if best_val_loss < random_baseline:
    print("  Model learned above chance on validation set.")
else:
    print("  WARNING: model did not exceed random baseline on validation set.")
    print("  Consider running Magdas training first, then re-running this script.")

# ── 10. Evaluation ────────────────────────────────────────────────────────────
print("\n[5] Evaluating on test set...")
encoder.load_state_dict(torch.load(OUT_ENCODER, map_location=device, weights_only=False))
proj.load_state_dict(torch.load(OUT_PROJ, map_location=device, weights_only=False))
encoder.eval(); proj.eval()

test_x_cpu = torch.tensor(X_test[:, np.newaxis, :, :], dtype=torch.float32)
proj_test_parts = []
for s in range(0, len(test_x_cpu), VAL_BATCH):
    xb = test_x_cpu[s:s+VAL_BATCH].to(device)
    with torch.no_grad():
        proj_test_parts.append(proj(encoder(xb)).cpu().numpy())
proj_test = np.vstack(proj_test_parts)

sims  = cosine_similarity(proj_test, bert_emb)
top5  = np.argsort(sims, axis=1)[:, -5:][:, ::-1]
total = len(y_test_words)

t1 = sum(top5[i, 0] == word2idx[y_test_words[i]] for i in range(total))
t5 = sum(word2idx[y_test_words[i]] in top5[i]    for i in range(total))
mc = sims[np.arange(total), [word2idx[w] for w in y_test_words]].mean()

vocab_idx2word = {i: w for w, i in word2idx.items()}

print(f"\n=== RESULTS — Pretrained Fine-Tuned EEG-to-Text ===")
print(f"  Backbone  : {backbone_name}")
print(f"  Top-1     : {t1/total:.3f} ({t1}/{total})  | KPI >10%:  {'OK' if t1/total > .10 else 'FAIL'}")
print(f"  Top-5     : {t5/total:.3f} ({t5}/{total})  | KPI >30%:  {'OK' if t5/total > .30 else 'FAIL'}")
print(f"  Cosine    : {mc:.3f}              | KPI >0.25: {'OK' if mc > .25 else 'FAIL'}")

print("\n  Sample predictions:")
for i in range(min(10, total)):
    tw  = y_test_words[i]
    pw  = [vocab_idx2word[idx] for idx in top5[i]]
    hit = "OK" if tw == pw[0] else "--"
    print(f"    [{hit}] True: {tw:<20s}  Top-5: {pw}")

# ── 11. Save metrics ──────────────────────────────────────────────────────────
metrics = {
    "backbone":           backbone_name,
    "loss":               "InfoNCE (temp=0.07)",
    "encoder_lr":         ENCODER_LR,
    "proj_lr":            PROJ_LR,
    "best_val_loss":      round(best_val_loss, 4),
    "random_baseline":    round(random_baseline, 4),
    "above_chance":       bool(best_val_loss < random_baseline),
    "top1":               round(t1 / total, 4),
    "top5":               round(t5 / total, 4),
    "cosine":             round(float(mc), 4),
    "top1_count":         int(t1),
    "top5_count":         int(t5),
    "total_test":         total,
    "kpi_top1_ok":        bool(t1 / total > .10),
    "kpi_top5_ok":        bool(t5 / total > .30),
    "kpi_cosine_ok":      bool(mc > .25),
    "output_encoder":     OUT_ENCODER,
    "output_projection":  OUT_PROJ,
}
with open(OUT_METRICS, "w") as f:
    json.dump(metrics, f, indent=2)

print(f"\nMetrics : {OUT_METRICS}")
print(f"Encoder : {OUT_ENCODER}")
print(f"Proj    : {OUT_PROJ}")
print("\nTo use: select 'Pretrained fine-tuned (Lupse)' in the model dropdown.")
print("\nTo use a custom checkpoint next time:")
print(f"  python notebooks/lupse/finetune_pretrained.py --pretrained-path <path/to/weights.pt>")
