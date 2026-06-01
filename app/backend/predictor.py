"""
predictor.py — Interfata BasePredictor + MockPredictor + RealPredictor
Lupse Ioan Victor — Sapt. 14
"""

import json
import time
import hashlib
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np

from app import config
from app.backend.schemas import PredictionResponse, WordScore


# ── Interfata abstracta ────────────────────────────────────────────────────────

class BasePredictor(ABC):
    @abstractmethod
    def predict(self, eeg: np.ndarray) -> PredictionResponse: ...


# ── MockPredictor ──────────────────────────────────────────────────────────────

class MockPredictor(BasePredictor):
    _BASE_SCORES = [0.71, 0.63, 0.57, 0.51, 0.44]

    def predict(self, eeg: np.ndarray) -> PredictionResponse:
        t0 = time.perf_counter()
        fingerprint = hashlib.md5(eeg.tobytes()).digest()
        seed = int.from_bytes(fingerprint[:4], "little") % (2**31)
        rng = np.random.default_rng(seed)
        indices = rng.choice(config.VOCAB_SIZE, size=5, replace=False)
        jitter = rng.uniform(-0.03, 0.03, size=5)
        scores = np.clip(np.array(self._BASE_SCORES) + jitter, 0.0, 1.0)
        top5 = [
            WordScore(word=config.IDX2WORD[int(idx)], score=round(float(sc), 4))
            for idx, sc in zip(indices, scores)
        ]
        indices_sent = rng.choice(config.VOCAB_SIZE, size=5, replace=False)
        reconstructed = " ".join(config.IDX2WORD[int(i)] for i in indices_sent)
        elapsed_ms = round((time.perf_counter() - t0) * 1000 + rng.uniform(15, 80), 2)
        return PredictionResponse(
            top_5_words=top5,
            reconstructed_sentence=reconstructed,
            inference_time_ms=elapsed_ms,
            model_version=config.MODEL_VERSION,
        )


# ── Loader pentru fisiere standard PyTorch (.pt) ────────────────────────────────

def _load_pt(path, device):
    """
    Incarca state_dict dintr-un fisier .pt standard (format zip sau legacy).
    """
    import torch
    path_str = str(path)

    if not Path(path_str).exists():
        raise FileNotFoundError(f"Nu s-a gasit fisierul modelului la calea: {path_str}")

    try:
        # Folosim weights_only=False pentru compatibilitate cu save-urile noastre
        return torch.load(path_str, map_location=device, weights_only=False)
    except Exception as e:
        raise RuntimeError(
            f"Eroare la incarcarea fisierului de ponderi '{path_str}'. "
            f"Asigura-te ca fisierul este o arhiva .pt valida si nu un director. Detalii: {e}"
        )


# ── RealPredictor ──────────────────────────────────────────────────────────────

class RealPredictor(BasePredictor):
    """
    Predictor real — incarca EEGNet + proiectia liniara + matricea BERT.

    Arhitectura EEGNet (dedusa din state_dict):
      block1: Conv2d(1,8,(1,64),pad=32,bias=F) -> BN(8)
              -> Conv2d(8,16,(105,1),groups=8,bias=F) -> BN(16) -> ELU -> AvgPool(1,4) -> Drop
      block2: Conv2d(16,16,(1,16),pad=8,groups=16,bias=F)
              -> Conv2d(16,16,1,bias=F) -> BN(16) -> ELU -> AvgPool(1,8) -> Drop
      emb:    Linear(240, 128)   [240 = 16 filtre * 15 pasi temporali]
      cls:    Linear(128, 10)    [clasificator, neutilizat la inferenta]

    Proiectie: Linear(128, 768)
    BERT:      (200, 768) float32
    """

    def __init__(self):
        import torch
        import torch.nn as nn
        from sklearn.metrics.pairwise import cosine_similarity

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        self.torch = torch

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

        self.eegnet = EEGNet().to(device)
        self.eegnet.load_state_dict(_load_pt(config.EEGNET_MODEL_PATH, device))
        self.eegnet.eval()

        self.projection = nn.Linear(128, 768).to(device)
        self.projection.load_state_dict(_load_pt(config.PROJECTION_PATH, device))
        self.projection.eval()

        self.bert_matrix = np.load(str(config.BERT_EMBEDDINGS_PATH))

        with open(str(config.VOCAB_INDEX_PATH), "r", encoding="utf-8") as f:
            vocab_raw = json.load(f)
        self.idx2word = {int(k): v for k, v in vocab_raw.items()}

        self.cos_sim = cosine_similarity

    def predict(self, eeg: np.ndarray) -> PredictionResponse:
        t0 = time.perf_counter()
        torch = self.torch

        from app.backend.preprocess import preprocess
        eeg_norm = preprocess(eeg, fs=config.SAMPLING_RATE)   # bandpass + z-score

        x = torch.tensor(eeg_norm[np.newaxis, np.newaxis]).to(self.device)

        with torch.no_grad():
            emb_128 = self.eegnet(x, return_embedding=True)
            emb_768 = self.projection(emb_128).cpu().numpy()

        sims = self.cos_sim(emb_768, self.bert_matrix)[0]
        top5_idx = np.argsort(sims)[-5:][::-1]

        top5 = [
            WordScore(
                word=self.idx2word[int(i)],
                score=round(float(np.clip(sims[i], 0.0, 1.0)), 4),
            )
            for i in top5_idx
        ]

        reconstructed = " ".join(ws.word for ws in top5)
        elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)

        return PredictionResponse(
            top_5_words=top5,
            reconstructed_sentence=reconstructed,
            inference_time_ms=elapsed_ms,
            model_version=config.MODEL_VERSION,
        )


# ── Factory ───────────────────────────────────────────────────────────────────

def get_predictor() -> BasePredictor:
    if config.USE_MOCK:
        return MockPredictor()
    return RealPredictor()