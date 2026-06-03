"""
predictor.py — BasePredictor + MockPredictor + RealPredictor + ConformerPredictor
Lupse Ioan Victor — Sapt. 14

Three real model backends:
  eegnet     — Laslo's frozen EEGNet + NonLinearProjectionHead (MSE-trained)
  conformer  — Magdas's EEG-Conformer + NonLinearProjectionHead (end-to-end InfoNCE)
  pretrained — Lupse's fine-tuned pretrained encoder
"""

import json
import time
import hashlib
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np

from app import config
from app.backend.schemas import PredictionResponse, WordScore


# ── Shared architecture definitions ───────────────────────────────────────────
# These must exactly match the architectures used in the retrain scripts
# so state_dict keys align on load.

def _make_eegnet(device):
    """EEGNet as trained by Laslo (P1). Output: 128-dim embedding."""
    import torch.nn as nn
    class EEGNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.block1 = nn.Sequential(
                nn.Conv2d(1, 8, (1, 64), padding=(0, 32), bias=False),
                nn.BatchNorm2d(8),
                nn.Conv2d(8, 16, (105, 1), groups=8, bias=False),
                nn.BatchNorm2d(16),
                nn.ELU(),
                nn.AvgPool2d((1, 4)),
                nn.Dropout(0.5),
            )
            self.block2 = nn.Sequential(
                nn.Conv2d(16, 16, (1, 16), padding=(0, 8), groups=16, bias=False),
                nn.Conv2d(16, 16, (1, 1), bias=False),
                nn.BatchNorm2d(16),
                nn.ELU(),
                nn.AvgPool2d((1, 8)),
                nn.Dropout(0.5),
            )
            self.emb = nn.Linear(240, 128)
            self.cls = nn.Linear(128, 10)

        def forward(self, x, return_embedding=False):
            x = self.block1(x)
            x = self.block2(x)
            x = x.flatten(1)
            emb = self.emb(x)
            if return_embedding:
                return emb
            return self.cls(emb)

    return EEGNet().to(device)


def _make_conformer(device, n_times=500):
    """
    EEG-Conformer as implemented by Magdas (P2) and used in retrain scripts.
    n_times must match what the saved weights were trained on (500 by default).
    """
    import torch.nn as nn
    class EEGConformer(nn.Module):
        def __init__(self, n_channels=105, n_times=500, embed_dim=128):
            super().__init__()
            self.temporal_conv = nn.Conv2d(1, 40, (1, 25), stride=(1, 1), padding=(0, 12), bias=False)
            self.bn1 = nn.BatchNorm2d(40)
            self.spatial_conv = nn.Conv2d(40, 40, (n_channels, 1), bias=False)
            self.bn2 = nn.BatchNorm2d(40)
            self.elu = nn.ELU()
            self.pool = nn.AvgPool2d((1, 4))
            seq_len = n_times // 4
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=40, nhead=4, dim_feedforward=128, dropout=0.1, batch_first=True
            )
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
            self.fc = nn.Linear(40 * seq_len, embed_dim)

        def forward(self, x):
            x = self.elu(self.bn1(self.temporal_conv(x)))
            x = self.elu(self.bn2(self.spatial_conv(x)))
            x = self.pool(x)
            x = x.squeeze(2).transpose(1, 2)
            x = self.transformer(x)
            x = x.reshape(x.size(0), -1)
            return self.fc(x)

    return EEGConformer(n_channels=105, n_times=n_times, embed_dim=128).to(device)


def _make_projection(device):
    """
    NonLinearProjectionHead (128 -> 256 -> 768).
    Matches the architecture saved by all three retrain scripts.
    """
    import torch.nn as nn
    class NonLinearProjectionHead(nn.Module):
        def __init__(self, input_dim=128, output_dim=768, hidden_dim=256, dropout=0.4):
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

    return NonLinearProjectionHead().to(device)


def _load_pt(path, device):
    """Load a .pt state dict with weights_only=False for broad compatibility."""
    import torch
    path_str = str(path)
    if not Path(path_str).exists():
        raise FileNotFoundError(f"Model file not found: {path_str}")
    try:
        return torch.load(path_str, map_location=device, weights_only=False)
    except Exception as e:
        raise RuntimeError(f"Cannot load '{path_str}': {e}")


def _load_vocab_and_bert():
    """Load BERT embedding matrix and index->word map."""
    bert_matrix = np.load(str(config.BERT_EMBEDDINGS_PATH))
    with open(str(config.VOCAB_INDEX_PATH), "r", encoding="utf-8") as f:
        raw = json.load(f)
    idx2word = {int(k): v for k, v in raw.items()}
    return bert_matrix, idx2word


def _top5_from_embedding(emb_768: np.ndarray, bert_matrix: np.ndarray,
                          idx2word: dict, model_version: str,
                          elapsed_ms: float) -> PredictionResponse:
    """Cosine similarity search -> top-5 WordScore list."""
    from sklearn.metrics.pairwise import cosine_similarity
    sims = cosine_similarity(emb_768, bert_matrix)[0]
    top5_idx = np.argsort(sims)[-5:][::-1]
    top5 = [
        WordScore(
            word=idx2word[int(i)],
            score=round(float(np.clip(sims[i], 0.0, 1.0)), 4),
        )
        for i in top5_idx
    ]
    reconstructed = " ".join(ws.word for ws in top5)
    return PredictionResponse(
        top_5_words=top5,
        reconstructed_sentence=reconstructed,
        inference_time_ms=round(elapsed_ms, 2),
        model_version=model_version,
    )


# ── Abstract base ─────────────────────────────────────────────────────────────

class BasePredictor(ABC):
    @abstractmethod
    def predict(self, eeg: np.ndarray) -> PredictionResponse: ...


# ── Mock predictor ────────────────────────────────────────────────────────────

class MockPredictor(BasePredictor):
    _BASE_SCORES = [0.71, 0.63, 0.57, 0.51, 0.44]

    def predict(self, eeg: np.ndarray) -> PredictionResponse:
        t0 = time.perf_counter()
        fingerprint = hashlib.md5(eeg.tobytes()).digest()
        seed = int.from_bytes(fingerprint[:4], "little") % (2 ** 31)
        rng = np.random.default_rng(seed)
        indices = rng.choice(config.VOCAB_SIZE, size=5, replace=False)
        jitter = rng.uniform(-0.03, 0.03, size=5)
        scores = np.clip(np.array(self._BASE_SCORES) + jitter, 0.0, 1.0)
        top5 = [
            WordScore(word=config.IDX2WORD[int(idx)], score=round(float(sc), 4))
            for idx, sc in zip(indices, scores)
        ]
        reconstructed = " ".join(ws.word for ws in top5)
        elapsed_ms = round((time.perf_counter() - t0) * 1000 + rng.uniform(15, 80), 2)
        return PredictionResponse(
            top_5_words=top5,
            reconstructed_sentence=reconstructed,
            inference_time_ms=elapsed_ms,
            model_version="mock-v1.0",
        )


# ── EEGNet predictor (Laslo) ──────────────────────────────────────────────────

class EEGNetPredictor(BasePredictor):
    """
    Laslo's frozen EEGNet (128d) + NonLinearProjectionHead (128->768).
    Projection trained with MSE on L2-normalised BERT embeddings.
    Best result: Cosine=0.933, Top-1=13%, Top-5=14.8%
    """

    def __init__(self):
        import torch
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        self.torch = torch

        self.eegnet = _make_eegnet(device)
        self.eegnet.load_state_dict(_load_pt(config.EEGNET_MODEL_PATH, device))
        self.eegnet.eval()

        self.projection = _make_projection(device)
        try:
            self.projection.load_state_dict(_load_pt(config.PROJECTION_PATH, device))
        except Exception:
            # Fallback: linear projection saved by an older run
            import torch.nn as nn
            linear = nn.Linear(128, 768).to(device)
            linear.load_state_dict(_load_pt(config.PROJECTION_PATH, device))
            self.projection = linear
        self.projection.eval()

        self.bert_matrix, self.idx2word = _load_vocab_and_bert()

    def predict(self, eeg: np.ndarray) -> PredictionResponse:
        t0 = time.perf_counter()
        from app.backend.preprocess import preprocess
        eeg_norm = preprocess(eeg, fs=config.SAMPLING_RATE)

        x = self.torch.tensor(eeg_norm[np.newaxis, np.newaxis]).to(self.device)
        with self.torch.no_grad():
            emb_128 = self.eegnet(x, return_embedding=True)
            emb_768 = self.projection(emb_128).cpu().numpy()

        return _top5_from_embedding(
            emb_768, self.bert_matrix, self.idx2word,
            "eegnet-mse-v1.0", (time.perf_counter() - t0) * 1000
        )


# ── EEG-Conformer predictor (Magdas) ─────────────────────────────────────────

class ConformerPredictor(BasePredictor):
    """
    Magdas's EEG-Conformer (128d) + NonLinearProjectionHead.
    Joint end-to-end InfoNCE training.
    Result: Top-1=12.3%, Top-5=31.5%, Cosine=0.41
    """

    def __init__(self):
        import torch
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        self.torch = torch

        self.conformer = _make_conformer(device, n_times=config.T_SAMPLES)
        self.conformer.load_state_dict(_load_pt(config.CONFORMER_MODEL_PATH, device))
        self.conformer.eval()

        self.projection = _make_projection(device)
        # Use Magdas-specific projection if present, otherwise fall back to root
        proj_path = config.MAGDAS_PROJ_PATH if config.MAGDAS_PROJ_PATH.exists() else config.PROJECTION_PATH
        self.projection.load_state_dict(_load_pt(proj_path, device))
        self.projection.eval()

        self.bert_matrix, self.idx2word = _load_vocab_and_bert()

    def predict(self, eeg: np.ndarray) -> PredictionResponse:
        t0 = time.perf_counter()
        from app.backend.preprocess import preprocess
        eeg_norm = preprocess(eeg, fs=config.SAMPLING_RATE)

        x = self.torch.tensor(eeg_norm[np.newaxis, np.newaxis]).to(self.device)
        with self.torch.no_grad():
            emb_128 = self.conformer(x)
            emb_768 = self.projection(emb_128).cpu().numpy()

        return _top5_from_embedding(
            emb_768, self.bert_matrix, self.idx2word,
            "conformer-infonce-v1.0", (time.perf_counter() - t0) * 1000
        )


# ── Pretrained fine-tuned predictor (Lupse) ───────────────────────────────────

class PretrainedPredictor(BasePredictor):
    """
    Lupse's fine-tuned pretrained model (finetune_pretrained.py).
    Falls back to EEGNetPredictor if the fine-tuned weights are not yet generated.
    """

    def __init__(self):
        if not config.PRETRAINED_MODEL_PATH.exists():
            print("[PretrainedPredictor] pretrained_finetune.pt not found — falling back to EEGNet.")
            self._delegate = EEGNetPredictor()
            self._using_delegate = True
            return

        import torch
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        self.torch = torch
        self._using_delegate = False

        # Lupse fine-tunes an EEGConformer (same architecture as Magdas),
        # NOT EEGNet. Use _make_conformer to match the saved state dict keys.
        self.encoder = _make_conformer(device, n_times=config.T_SAMPLES)
        self.encoder.load_state_dict(_load_pt(config.PRETRAINED_MODEL_PATH, device))
        self.encoder.eval()

        self.projection = _make_projection(device)
        proj_path = config.PRETRAINED_PROJ_PATH
        if proj_path.exists():
            self.projection.load_state_dict(_load_pt(proj_path, device))
        else:
            self.projection.load_state_dict(_load_pt(config.MAGDAS_PROJ_PATH, device))
        self.projection.eval()

        self.bert_matrix, self.idx2word = _load_vocab_and_bert()

    def predict(self, eeg: np.ndarray) -> PredictionResponse:
        if self._using_delegate:
            return self._delegate.predict(eeg)

        t0 = time.perf_counter()
        from app.backend.preprocess import preprocess
        eeg_norm = preprocess(eeg, fs=config.SAMPLING_RATE)

        x = self.torch.tensor(eeg_norm[np.newaxis, np.newaxis]).to(self.device)
        with self.torch.no_grad():
            emb_128 = self.encoder(x)   # EEGConformer.forward takes only x
            emb_768 = self.projection(emb_128).cpu().numpy()

        return _top5_from_embedding(
            emb_768, self.bert_matrix, self.idx2word,
            "pretrained-finetune-v1.0", (time.perf_counter() - t0) * 1000
        )


# ── Factory ───────────────────────────────────────────────────────────────────

def get_predictor(model_name: str | None = None) -> BasePredictor:
    if config.USE_MOCK:
        return MockPredictor()
    name = (model_name or config.DEFAULT_MODEL).lower()
    if name == "conformer":
        return ConformerPredictor()
    if name == "pretrained":
        return PretrainedPredictor()
    return EEGNetPredictor()  # default
