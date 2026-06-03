"""
regenerate_bert_embeddings.py — Rebuild bert_embeddings.npy and vocab_index.json
Lupse Ioan Victor — Sapt. 14

WHY THIS MUST BE RUN WHEN THE VOCABULARY CHANGES
-------------------------------------------------
bert_embeddings.npy stores one BERT vector per vocabulary word, in vocabulary order.
Row i must be the BERT embedding of vocabulary[i].

If you switch benchmark configs (e.g. from 200-word to 50-word), the old
bert_embeddings.npy has the wrong rows for the new word list.
Truncating the old file silently maps words to incorrect embeddings — the model
trains against wrong targets and Top-1 accuracy collapses to 0%.

This script re-derives both files from scratch for whatever config is currently active.

Outputs (overwrites existing files)
    app/models/bert_embeddings.npy    shape (vocab_size, 768)
    app/models/vocab_index.json       {"0": "word0", "1": "word1", ...}

Run from repo root:
    .venv\\Scripts\\python regenerate_bert_embeddings.py
    .venv\\Scripts\\python regenerate_bert_embeddings.py --config benchmark_config.json
    .venv\\Scripts\\python regenerate_bert_embeddings.py --verify-only
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT       = Path(__file__).resolve().parent
MODELS_DIR = ROOT / "app" / "models"

# ── CLI ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(
    description="Regenerate bert_embeddings.npy and vocab_index.json for the active vocabulary."
)
parser.add_argument(
    "--config",
    default=None,
    help=(
        "Path to benchmark config JSON. "
        "Defaults to benchmark_config_50_Zuco1_SR.json if it exists, "
        "otherwise benchmark_config.json."
    ),
)
parser.add_argument(
    "--verify-only",
    action="store_true",
    help="Check alignment without regenerating. Exits with code 1 if misaligned.",
)
parser.add_argument(
    "--model",
    default="bert-base-uncased",
    help="HuggingFace model name for embeddings (default: bert-base-uncased).",
)
args = parser.parse_args()

# ── Resolve config ────────────────────────────────────────────────────────────
if args.config:
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = ROOT / config_path
else:
    config_path = ROOT / "benchmark_config_50_Zuco1_SR.json"
    if not config_path.exists():
        config_path = ROOT / "benchmark_config.json"

if not config_path.exists():
    print(f"[ERROR] Config not found: {config_path}")
    sys.exit(1)

with open(config_path, "r", encoding="utf-8") as f:
    cfg = json.load(f)

vocabulary: list[str] = cfg["vocabulary"]
vocab_size = len(vocabulary)

print(f"\nConfig      : {config_path.name}")
print(f"Vocabulary  : {vocab_size} words")
print(f"BERT model  : {args.model}")

out_npy  = MODELS_DIR / "bert_embeddings.npy"
out_json = MODELS_DIR / "vocab_index.json"

# ── Verify-only mode ──────────────────────────────────────────────────────────
if args.verify_only:
    ok = True
    if not out_npy.exists():
        print(f"\n[FAIL] {out_npy} does not exist.")
        ok = False
    else:
        emb = np.load(str(out_npy))
        if emb.shape[0] != vocab_size:
            print(f"\n[FAIL] bert_embeddings.npy has {emb.shape[0]} rows but vocabulary has {vocab_size} words.")
            ok = False
        else:
            print(f"\n[OK]   bert_embeddings.npy shape {emb.shape} matches vocabulary size {vocab_size}.")

    if not out_json.exists():
        print(f"[FAIL] {out_json} does not exist.")
        ok = False
    else:
        with open(str(out_json)) as f:
            idx = json.load(f)
        if len(idx) != vocab_size:
            print(f"[FAIL] vocab_index.json has {len(idx)} entries but vocabulary has {vocab_size} words.")
            ok = False
        else:
            # Check that every vocabulary word appears
            idx_words = set(idx.values())
            missing = [w for w in vocabulary if w not in idx_words]
            if missing:
                print(f"[FAIL] {len(missing)} vocabulary words missing from vocab_index.json: {missing[:5]}...")
                ok = False
            else:
                print(f"[OK]   vocab_index.json has {len(idx)} entries, all vocabulary words present.")

    if ok:
        print("\nAlignment verified — no regeneration needed.")
        sys.exit(0)
    else:
        print("\nRun without --verify-only to regenerate.")
        sys.exit(1)

# ── Load BERT ─────────────────────────────────────────────────────────────────
print("\nLoading BERT tokenizer and model...")
try:
    from transformers import BertModel, BertTokenizer
except ImportError:
    print("[ERROR] transformers not installed. Run: pip install transformers")
    sys.exit(1)

try:
    import torch
except ImportError:
    print("[ERROR] torch not installed. Run: pip install torch")
    sys.exit(1)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

tokenizer = BertTokenizer.from_pretrained(args.model)
model     = BertModel.from_pretrained(args.model).to(device)
model.eval()

# ── Compute embeddings ────────────────────────────────────────────────────────
print(f"\nComputing CLS embeddings for {vocab_size} words...")
embeddings = np.zeros((vocab_size, 768), dtype=np.float32)

BATCH_SIZE = 32
with torch.no_grad():
    for start in range(0, vocab_size, BATCH_SIZE):
        batch_words = vocabulary[start : start + BATCH_SIZE]
        encoded = tokenizer(
            batch_words,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=8,   # single words — short context is fine
        ).to(device)
        outputs = model(**encoded)
        # CLS token (position 0) — consistent with how the original bert_embeddings.npy was made
        cls_vecs = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        embeddings[start : start + len(batch_words)] = cls_vecs

        end = min(start + BATCH_SIZE, vocab_size)
        print(f"  [{end:>3}/{vocab_size}]  {batch_words[0]} ... {batch_words[-1]}")

# ── L2 normalise (optional but consistent with training scripts) ──────────────
norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
norms = np.where(norms < 1e-8, 1.0, norms)
embeddings_normed = (embeddings / norms).astype(np.float32)

# ── Save ──────────────────────────────────────────────────────────────────────
MODELS_DIR.mkdir(parents=True, exist_ok=True)
np.save(str(out_npy), embeddings_normed)
print(f"\nSaved: {out_npy}  shape {embeddings_normed.shape}")

vocab_index = {str(i): word for i, word in enumerate(vocabulary)}
with open(str(out_json), "w", encoding="utf-8") as f:
    json.dump(vocab_index, f, indent=2, ensure_ascii=False)
print(f"Saved: {out_json}  ({vocab_size} entries)")

# ── Sanity check ──────────────────────────────────────────────────────────────
print("\nSanity check (cosine similarity between similar words):")
word2idx = {w: i for i, w in enumerate(vocabulary)}
pairs = [
    ("film",  "movie"),
    ("story", "life"),
    ("war",   "army"),
]
for w1, w2 in pairs:
    if w1 in word2idx and w2 in word2idx:
        v1 = embeddings_normed[word2idx[w1]]
        v2 = embeddings_normed[word2idx[w2]]
        sim = float(np.dot(v1, v2))
        print(f"  cos({w1}, {w2}) = {sim:.4f}")
    else:
        missing = w1 if w1 not in word2idx else w2
        print(f"  '{missing}' not in active vocabulary — skipped.")

print(f"\n[OK] bert_embeddings.npy and vocab_index.json are now aligned to {config_path.name}")
print("     You can now run the model training scripts safely.")
