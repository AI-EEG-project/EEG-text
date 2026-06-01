"""
config.py — Configuratie centralizata EEG-to-Text demo
Lupse Ioan Victor — Sapt. 14

Singura modificare necesara dupa ce Laslo livreaza modelele:
    USE_MOCK = False
"""

import json
import os
from pathlib import Path

# ── Root repo ─────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent   # radacina repo-ului

# ── Switch mock / real ────────────────────────────────────────────────────────
# True  → MockPredictor (nu necesita fisierele lui Laslo)
# False → RealPredictor (necesita fisierele din app/models/)
USE_MOCK: bool = False

# ── Cai fisiere model (completate de Laslo) ───────────────────────────────────
MODELS_DIR          = ROOT / "app" / "models"
EEGNET_MODEL_PATH   = MODELS_DIR / "eegnet_model.pt"
PROJECTION_PATH     = MODELS_DIR / "linear_projection.pt"
BERT_EMBEDDINGS_PATH= MODELS_DIR / "bert_embeddings.npy"
VOCAB_INDEX_PATH    = MODELS_DIR / "vocab_index.json"

# ── Exemple predefinite ───────────────────────────────────────────────────────
EXAMPLES_DIR        = ROOT / "data" / "examples"
EXAMPLES_MANIFEST   = EXAMPLES_DIR / "manifest.json"

# ── Parametri EEG (din benchmark_config.json) ─────────────────────────────────
N_CHANNELS: int     = 105
SAMPLING_RATE: int  = 500        # Hz
EPOCH_START_MS: int = -200       # ms fata de onset cuvant
EPOCH_END_MS: int   = 800        # ms
T_SAMPLES: int      = int((EPOCH_END_MS - EPOCH_START_MS) / 1000 * SAMPLING_RATE)  # 500
ARTIFACT_THRESH: float = 100.0   # microV

# ── Vocabular (din benchmark_config.json) ─────────────────────────────────────
_CONFIG_PATH = ROOT / "benchmark_config.json"

with open(_CONFIG_PATH, "r", encoding="utf-8") as _f:
    _cfg = json.load(_f)

VOCABULARY: list[str]       = _cfg["vocabulary"]          # 200 cuvinte ordonate
VOCAB_SIZE: int             = len(VOCABULARY)             # 200
WORD2IDX: dict[str, int]    = {w: i for i, w in enumerate(VOCABULARY)}
IDX2WORD: dict[int, str]    = {i: w for i, w in enumerate(VOCABULARY)}
VOCABULARY_POS: dict[str, str] = _cfg["vocabulary_pos"]  # cuvant → POS tag

# ── Versiune model ─────────────────────────────────────────────────────────────
MODEL_VERSION: str = "mock-v1.0" if USE_MOCK else "eegnet-v1.0"
