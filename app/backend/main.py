"""
main.py — Backend FastAPI pentru demo EEG-to-Text
Lupse Ioan Victor — Sapt. 14

Pornire:
    uvicorn app.backend.main:app --reload --port 8000

Swagger UI:
    http://localhost:8000/docs
"""

import io
import json
from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from app import config
from app.backend.predictor import BasePredictor, get_predictor
from app.backend.schemas import (
    ErrorDetail,
    EvaluationResponse,
    ExampleItem,
    ExamplesResponse,
    HealthResponse,
    PredictionResponse,
)

# ── State global ──────────────────────────────────────────────────────────────

_predictor: BasePredictor | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _predictor
    _predictor = get_predictor()
    mode = "MOCK" if config.USE_MOCK else "REAL"
    print(f"[startup] Predictor incarcat ({mode}) — vocab {config.VOCAB_SIZE} cuvinte")
    yield
    _predictor = None
    print("[shutdown] Predictor eliberat")


# ── Aplicatia FastAPI ─────────────────────────────────────────────────────────

app = FastAPI(
    title="EEG-to-Text API",
    description=(
        "Backend pentru reconstructia textului din semnale EEG. "
        "Proiect AI — Echipa Laslo / Magdas / Lupse — Sapt. 14."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501", "http://127.0.0.1:8501"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# ── Utilitare ─────────────────────────────────────────────────────────────────

def _validate_eeg(arr: np.ndarray) -> None:
    if arr.ndim != 2:
        raise HTTPException(422, f"EEG trebuie 2D (n_ch, T), primit {arr.ndim}D.")
    n_ch, n_t = arr.shape
    if n_ch != config.N_CHANNELS:
        raise HTTPException(422, f"Primul dim trebuie {config.N_CHANNELS} canale, primit {n_ch}.")
    T_MIN, T_MAX = 50, 700
    if not (T_MIN <= n_t <= T_MAX):
        raise HTTPException(422, f"n_times trebuie in [{T_MIN},{T_MAX}], primit {n_t}.")
    if not np.issubdtype(arr.dtype, np.floating) and not np.issubdtype(arr.dtype, np.integer):
        raise HTTPException(422, f"Dtype trebuie numeric, primit {arr.dtype}.")
    if np.any(~np.isfinite(arr)):
        raise HTTPException(422, "Array-ul contine NaN sau Inf.")
    if np.abs(arr).max() > 2000:
        raise HTTPException(422, f"Amplitudine prea mare ({np.abs(arr).max():.1f} uV).")


def _load_npy_from_bytes(data: bytes) -> np.ndarray:
    try:
        return np.load(io.BytesIO(data), allow_pickle=False)
    except Exception as exc:
        raise HTTPException(422, f"Fisierul nu poate fi citit ca numpy array: {exc}")


def _load_examples_manifest() -> list[dict]:
    p = config.EXAMPLES_MANIFEST
    if not p.exists():
        return []
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f).get("examples", [])


# ── Endpoint-uri ──────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["Utilitare"])
def health() -> HealthResponse:
    """Status server + versiune model."""
    return HealthResponse(
        status="ok",
        model_version=config.MODEL_VERSION,
        using_mock=config.USE_MOCK,
        vocab_size=config.VOCAB_SIZE,
        n_channels=config.N_CHANNELS,
    )


@app.get("/examples", response_model=ExamplesResponse, tags=["Utilitare"])
def get_examples() -> ExamplesResponse:
    """Returneaza lista cu 5 exemple predefinite din ZuCo."""
    raw = _load_examples_manifest()
    items = [
        ExampleItem(
            id=ex["id"],
            description=ex["description"],
            filename=ex["filename"],
            shape=ex["shape"],
            true_word=ex["true_word"],
        )
        for ex in raw
    ]
    return ExamplesResponse(examples=items)


@app.post("/predict", response_model=PredictionResponse, tags=["Predictie"])
async def predict(
    file: UploadFile = File(..., description="Fisier .npy cu epoch EEG, shape (105, T)")
) -> PredictionResponse:
    """
    Primeste fisier .npy (shape 105 x T) si returneaza top-5 cuvinte candidate.
    """
    content = await file.read()
    arr = _load_npy_from_bytes(content).astype(np.float32)
    _validate_eeg(arr)
    if _predictor is None:
        raise HTTPException(503, "Predictor-ul nu e initializat.")
    return _predictor.predict(arr)


@app.post("/predict/example/{example_id}", response_model=PredictionResponse, tags=["Predictie"])
def predict_example(example_id: int) -> PredictionResponse:
    """Ruleaza predictia pe unul din cele 5 exemple predefinite."""
    examples = _load_examples_manifest()
    matching = [ex for ex in examples if ex["id"] == example_id]
    if not matching:
        raise HTTPException(404, f"Exemplu id={example_id} inexistent.")
    ex = matching[0]
    npy_path = config.EXAMPLES_DIR / ex["filename"]
    if not npy_path.exists():
        raise HTTPException(404, f"Fisierul exemplului lipseste: {npy_path}")
    arr = np.load(npy_path).astype(np.float32)
    _validate_eeg(arr)
    if _predictor is None:
        raise HTTPException(503, "Predictor-ul nu e initializat.")
    return _predictor.predict(arr)


@app.get("/evaluate", response_model=EvaluationResponse, tags=["Evaluare"])
def evaluate() -> EvaluationResponse:
    """
    Evalueaza modelul pe toate exemplele predefinite si returneaza KPI-urile:
    Top-1 accuracy, Top-5 accuracy, avg cosine similarity, avg semantic similarity.

    Similaritatea semantica = cosine intre embeddings BERT ale cuvantului prezis vs. real.
    """
    if _predictor is None:
        raise HTTPException(503, "Predictor-ul nu e initializat.")
    examples = _load_examples_manifest()
    if not examples:
        raise HTTPException(404, "Niciun exemplu predefinit gasit.")

    # Incarca BERT matrix + vocab Laslo pentru similaritate semantica
    bert_matrix = np.load(str(config.BERT_EMBEDDINGS_PATH))   # (200, 768)
    with open(str(config.VOCAB_INDEX_PATH), "r", encoding="utf-8") as f:
        vocab_raw = json.load(f)
    word2idx = {v: int(k) for k, v in vocab_raw.items()}      # word -> row

    top1_hits, top5_hits = 0, 0
    cosine_scores: list[float] = []
    semantic_scores: list[float] = []
    per_example: list[dict] = []

    for ex in examples:
        npy_path = config.EXAMPLES_DIR / ex["filename"]
        if not npy_path.exists():
            continue
        arr = np.load(npy_path).astype(np.float32)
        try:
            _validate_eeg(arr)
        except HTTPException:
            continue

        result  = _predictor.predict(arr)
        words   = [ws.word  for ws in result.top_5_words]
        scores  = [ws.score for ws in result.top_5_words]
        true_w  = ex["true_word"]
        pred_w  = words[0]

        top1_hit = (pred_w == true_w)
        top5_hit = (true_w in words)
        if top1_hit:
            top1_hits += 1
        if top5_hit:
            top5_hits += 1
        cosine_scores.append(scores[0])

        # Similaritate semantica via BERT embeddings
        pi = word2idx.get(pred_w)
        ti = word2idx.get(true_w)
        if pi is not None and ti is not None:
            vp = bert_matrix[pi]
            vt = bert_matrix[ti]
            sem_sim = float(np.dot(vp, vt) / (np.linalg.norm(vp) * np.linalg.norm(vt) + 1e-8))
        else:
            sem_sim = 0.0
        semantic_scores.append(sem_sim)

        per_example.append({
            "id":           ex["id"],
            "description":  ex["description"],
            "true_word":    true_w,
            "pred_top1":    pred_w,
            "top5":         words,
            "top1_hit":     top1_hit,
            "top5_hit":     top5_hit,
            "cosine_score": round(scores[0], 4),
            "semantic_sim": round(sem_sim, 4),
            "inference_ms": result.inference_time_ms,
        })

    n = len(per_example)
    if n == 0:
        raise HTTPException(500, "Niciun exemplu evaluat cu succes.")

    return EvaluationResponse(
        n_examples=n,
        top1_accuracy=round(top1_hits / n, 3),
        top5_accuracy=round(top5_hits / n, 3),
        avg_cosine_similarity=round(float(np.mean(cosine_scores)), 4),
        avg_semantic_similarity=round(float(np.mean(semantic_scores)), 4),
        per_example=per_example,
    )
