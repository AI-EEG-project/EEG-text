"""
config.py — Central configuration for EEG-to-Text demo
Lupse Ioan Victor — Sapt. 14
"""

import json
from pathlib import Path

# ── Repo root ─────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent

# ── Mock switch ───────────────────────────────────────────────────────────────
# True  -> MockPredictor (no model files needed)
# False -> real model inference
USE_MOCK: bool = False

# ── Model files ───────────────────────────────────────────────────────────────
MODELS_DIR              = ROOT / "app" / "models"

# Shared assets (root — used by all three models)
BERT_EMBEDDINGS_PATH    = MODELS_DIR / "bert_embeddings.npy"
VOCAB_INDEX_PATH        = MODELS_DIR / "vocab_index.json"

# Laslo (P1): EEGNet backbone + MSE NonLinear projection
LASLO_DIR               = MODELS_DIR / "laslo"
EEGNET_MODEL_PATH       = MODELS_DIR / "eegnet_model.pt"   # backbone shared at root
PROJECTION_PATH         = LASLO_DIR  / "projection.pt"

# Magdas (P2): EEG-Conformer + end-to-end InfoNCE
MAGDAS_DIR              = MODELS_DIR / "magdas"
CONFORMER_MODEL_PATH    = MAGDAS_DIR / "eeg_conformer.pt"
MAGDAS_PROJ_PATH        = MAGDAS_DIR / "projection.pt"

# Lupse (P3): Pretrained EEG-to-text fine-tuned model
LUPSE_DIR               = MODELS_DIR / "lupse"
PRETRAINED_MODEL_PATH   = LUPSE_DIR  / "encoder.pt"
PRETRAINED_PROJ_PATH    = LUPSE_DIR  / "projection.pt"

# ── Available model backends ──────────────────────────────────────────────────
# Maps display name -> internal key used by get_predictor()
AVAILABLE_MODELS: dict[str, str] = {
    "EEGNet  (Laslo — MSE projection)":           "eegnet",
    "EEG-Conformer  (Magdas — end-to-end InfoNCE)": "conformer",
    "Pretrained fine-tuned  (Lupse)":              "pretrained",
}
DEFAULT_MODEL: str = "eegnet"

# ── Predefined examples ───────────────────────────────────────────────────────
EXAMPLES_DIR        = ROOT / "data" / "examples"
EXAMPLES_MANIFEST   = EXAMPLES_DIR / "manifest.json"

# ── EEG parameters (from benchmark_config.json) ──────────────────────────────
N_CHANNELS: int         = 105
SAMPLING_RATE: int      = 500        # Hz
EPOCH_START_MS: int     = -200
EPOCH_END_MS: int       = 800
T_SAMPLES: int          = int((EPOCH_END_MS - EPOCH_START_MS) / 1000 * SAMPLING_RATE)  # 500
ARTIFACT_THRESH: float  = 100.0      # microV

# ── Vocabulary (from benchmark_config.json) ───────────────────────────────────
_CONFIG_PATH = ROOT / "benchmark_config.json"

with open(_CONFIG_PATH, "r", encoding="utf-8") as _f:
    _cfg = json.load(_f)

VOCABULARY: list[str]           = _cfg["vocabulary"]
VOCAB_SIZE: int                 = len(VOCABULARY)
WORD2IDX: dict[str, int]        = {w: i for i, w in enumerate(VOCABULARY)}
IDX2WORD: dict[int, str]        = {i: w for i, w in enumerate(VOCABULARY)}
VOCABULARY_POS: dict[str, str]  = _cfg.get("vocabulary_pos", {})

# ── Model version string ──────────────────────────────────────────────────────
MODEL_VERSION: str = "mock-v1.0" if USE_MOCK else "eegnet-v1.0"
