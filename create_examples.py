"""
create_examples.py — Generate predefined example .npy files for the Streamlit demo
Lupse Ioan Victor — Sapt. 14

Reads X_preprocessed.npy / y_labels.npy and picks one representative EEG epoch
per test word, saves each as an individual .npy file, and writes manifest.json.

Run from repo root:
    python create_examples.py
"""

import json
import os
import numpy as np
from pathlib import Path

ROOT         = Path(__file__).resolve().parent
DATA_DIR     = ROOT / "data"
EXAMPLES_DIR = ROOT / "data" / "examples"
CONFIG_PATH  = ROOT / "benchmark_config.json"

EXAMPLES_DIR.mkdir(parents=True, exist_ok=True)

# ── Load preprocessed data ────────────────────────────────────────────────────
print("Loading preprocessed data...")
X      = np.load(str(DATA_DIR / "X_preprocessed.npy"))
y      = np.load(str(DATA_DIR / "y_labels.npy"), allow_pickle=True)
splits = np.load(str(DATA_DIR / "splits.npy"),   allow_pickle=True)

# ── Load vocabulary ───────────────────────────────────────────────────────────
with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    cfg = json.load(f)

vocab = cfg["vocabulary"]

# ── Pick test examples ─────────────────────────────────────────────────────────
# Take one epoch per word from the test split.
# Pick words that are semantically diverse and common enough to appear in test.
TARGET_WORDS = [
    "film", "story", "war", "family", "born",
    "love", "director", "school", "comedy", "life",
]

# Filter to words that actually exist in test split
test_mask  = splits == "test"
X_test     = X[test_mask]
y_test     = y[test_mask]

# Find first available epoch for each target word
examples   = []
used_words = set()

# First pass: target words
for word in TARGET_WORDS:
    if word not in vocab:
        continue
    idxs = np.where(y_test == word)[0]
    if len(idxs) == 0:
        continue
    epoch = X_test[idxs[0]]
    fname = f"{word}.npy"
    np.save(str(EXAMPLES_DIR / fname), epoch)
    examples.append({
        "id":          len(examples) + 1,
        "description": f"Word: '{word}'",
        "filename":    fname,
        "shape":       list(epoch.shape),
        "true_word":   word,
    })
    used_words.add(word)
    print(f"  Saved example #{len(examples)}: '{word}' — shape {epoch.shape}")

# Second pass: fill up to 10 with any available test words
if len(examples) < 10:
    for word in vocab:
        if word in used_words:
            continue
        idxs = np.where(y_test == word)[0]
        if len(idxs) == 0:
            continue
        epoch = X_test[idxs[0]]
        fname = f"{word}.npy"
        np.save(str(EXAMPLES_DIR / fname), epoch)
        examples.append({
            "id":          len(examples) + 1,
            "description": f"Word: '{word}'",
            "filename":    fname,
            "shape":       list(epoch.shape),
            "true_word":   word,
        })
        used_words.add(word)
        print(f"  Saved example #{len(examples)}: '{word}' — shape {epoch.shape}")
        if len(examples) >= 10:
            break

# ── Write manifest ─────────────────────────────────────────────────────────────
manifest = {"examples": examples}
with open(str(EXAMPLES_DIR / "manifest.json"), "w", encoding="utf-8") as f:
    json.dump(manifest, f, indent=2)

print(f"\nCreated {len(examples)} examples in {EXAMPLES_DIR}")
print(f"Manifest: {EXAMPLES_DIR / 'manifest.json'}")
