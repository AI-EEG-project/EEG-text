"""
main.py — FastAPI backend for EEG-to-Text demo
Lupse Ioan Victor — Sapt. 14

Start:
    uvicorn app.backend.main:app --reload --port 8000

Swagger UI:
    http://localhost:8000/docs
"""

import io
import json
from contextlib import asynccontextmanager
from typing import Annotated

import numpy as np
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from app import config
from app.backend.predictor import BasePredictor, get_predictor
from app.backend.schemas import (
    EvaluationResponse,
    ExampleItem,
    ExamplesResponse,
    HealthResponse,
    PredictionResponse,
)

# ── Global predictor cache (one instance per model_name) ─────────────────────

_predictors: dict[str, BasePredictor] = {}


def _get_or_load(model_name: str) -> BasePredictor:
    """Lazy-load and cache predictor instances."""
    if model_name not in _predictors:
        _predictors[model_name] = get_predictor(model_name)
    return _predictors[model_name]


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Pre-load the default model at startup
    default = config.DEFAULT_MODEL
    _predictors[default] = get_predictor(default)
    mode = "MOCK" if config.USE_MOCK else "REAL"
    print(f"[startup] Default predictor loaded ({mode}) — vocab {config.VOCAB_SIZE} words")
    yield
    _predictors.clear()
    print("[shutdown] Predictors released")


# ── FastAPI app ───────────────────────────────────────────────────────────────

app = FastAPI(
    title="EEG-to-Text API",
    description="Backend for EEG-to-text word retrieval. Team: Laslo / Magdas / Lupse — Sapt. 14.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501", "http://127.0.0.1:8501"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# ── Validation helpers ────────────────────────────────────────────────────────

def _validate_eeg(arr: np.ndarray) -> None:
    if arr.ndim != 2:
        raise HTTPException(422, f"EEG must be 2D (n_channels, T), got {arr.ndim}D.")
    n_ch, n_t = arr.shape
    if n_ch != config.N_CHANNELS:
        raise HTTPException(422, f"Expected {config.N_CHANNELS} channels, got {n_ch}.")
    if not (50 <= n_t <= 700):
        raise HTTPException(422, f"Time samples must be in [50, 700], got {n_t}.")
    if not np.issubdtype(arr.dtype, np.floating) and not np.issubdtype(arr.dtype, np.integer):
        raise HTTPException(422, f"Array dtype must be numeric, got {arr.dtype}.")
    if np.any(~np.isfinite(arr)):
        raise HTTPException(422, "Array contains NaN or Inf values.")
    if np.abs(arr).max() > 2000:
        raise HTTPException(422, f"Amplitude too large ({np.abs(arr).max():.1f} uV). Max allowed: 2000 uV.")


def _load_npy(data: bytes) -> np.ndarray:
    try:
        return np.load(io.BytesIO(data), allow_pickle=False)
    except Exception as exc:
        raise HTTPException(422, f"Cannot read as numpy array: {exc}")


def _load_mat(data: bytes) -> np.ndarray:
    """
    Extract a word-level EEG epoch from a ZuCo .mat file.
    Returns the first valid 105-channel epoch found.
    Uses the same parsing logic as mat_to_npy.py.
    """
    import tempfile, os

    # Write bytes to a temp file so h5py / scipy can read it
    with tempfile.NamedTemporaryFile(suffix=".mat", delete=False) as tmp:
        tmp.write(data)
        tmp_path = tmp.name

    try:
        epoch = _parse_mat_first_epoch(tmp_path)
    finally:
        os.unlink(tmp_path)

    if epoch is None:
        raise HTTPException(
            422,
            "No valid 105-channel EEG epoch found in the .mat file. "
            "Make sure it is a ZuCo-format subject file."
        )
    return epoch


def _parse_mat_first_epoch(mat_path: str) -> np.ndarray | None:
    """Return first valid (105, T) EEG array from a ZuCo .mat file."""
    N_CH = config.N_CHANNELS
    EPOCH_SAMPLES = config.T_SAMPLES  # 500

    def _fix_shape(arr: np.ndarray) -> np.ndarray | None:
        if arr.ndim < 2:
            return None
        if arr.shape[0] == N_CH:
            return arr.astype(np.float32)
        if arr.shape[1] == N_CH:
            return arr.T.astype(np.float32)
        if arr.ndim == 3 and arr.shape[1] == N_CH:
            return arr.transpose(1, 0, 2).reshape(N_CH, -1).astype(np.float32)
        return None

    def _pad_crop(arr: np.ndarray) -> np.ndarray:
        T = arr.shape[1]
        if T >= EPOCH_SAMPLES:
            return arr[:, :EPOCH_SAMPLES]
        return np.pad(arr, ((0, 0), (0, EPOCH_SAMPLES - T)), mode="constant", constant_values=0.0)

    # Try HDF5 / v7.3
    try:
        import h5py
        with h5py.File(mat_path, "r") as f:
            root = f.get("sentenceData") or f.get("data")
            if root is None and list(f.keys()):
                root = f[list(f.keys())[0]]
            if root is None:
                raise ValueError("no root")
            for sent_ref in np.array(root).flat:
                try:
                    sent = f[sent_ref] if isinstance(sent_ref, h5py.Reference) else sent_ref
                except Exception:
                    continue
                word_group = sent.get("word") if hasattr(sent, "get") else None
                if word_group is None:
                    continue
                for word_ref in np.array(word_group).flat:
                    try:
                        w_obj = f[word_ref] if isinstance(word_ref, h5py.Reference) else word_ref
                    except Exception:
                        continue
                    for key in ("rawEEG", "rawData", "meanFixation"):
                        ds = w_obj.get(key) if hasattr(w_obj, "get") else None
                        if ds is None:
                            continue
                        try:
                            arr = np.array(ds, dtype=np.float32)
                            arr = _fix_shape(arr)
                            if arr is None:
                                continue
                            if not np.isfinite(arr).all() or np.abs(arr).max() > 5000:
                                continue
                            return _pad_crop(arr)
                        except Exception:
                            continue
        return None
    except OSError:
        pass

    # Fallback: legacy MATLAB v5/v7
    try:
        import scipy.io
        data = scipy.io.loadmat(mat_path, struct_as_record=False, squeeze_me=True)
        sentence_data = data.get("sentenceData") or data.get("data")
        if sentence_data is None:
            for k, v in data.items():
                if not k.startswith("__"):
                    sentence_data = v
                    break
        if sentence_data is None:
            return None
        if not isinstance(sentence_data, (list, np.ndarray)):
            sentence_data = [sentence_data]
        for sent in sentence_data:
            if not hasattr(sent, "word"):
                continue
            words = sent.word
            if not isinstance(words, (list, np.ndarray)):
                words = [words]
            for w_obj in words:
                for key in ("rawEEG", "rawData", "meanFixation"):
                    if not hasattr(w_obj, key):
                        continue
                    try:
                        arr = np.array(getattr(w_obj, key), dtype=np.float32)
                        arr = _fix_shape(arr)
                        if arr is None:
                            continue
                        if not np.isfinite(arr).all() or np.abs(arr).max() > 5000:
                            continue
                        return _pad_crop(arr)
                    except Exception:
                        continue
    except Exception:
        pass
    return None


def _load_examples_manifest() -> list[dict]:
    p = config.EXAMPLES_MANIFEST
    if not p.exists():
        return []
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f).get("examples", [])


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["Utility"])
def health() -> HealthResponse:
    """Server status and model version."""
    return HealthResponse(
        status="ok",
        model_version=config.MODEL_VERSION,
        using_mock=config.USE_MOCK,
        vocab_size=config.VOCAB_SIZE,
        n_channels=config.N_CHANNELS,
    )


@app.get("/examples", response_model=ExamplesResponse, tags=["Utility"])
def get_examples() -> ExamplesResponse:
    """Return the list of predefined ZuCo test examples."""
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


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict(
    file: UploadFile = File(..., description="EEG epoch as .npy (105, T) or .mat ZuCo subject file"),
    model: Annotated[str, Query(description="Model backend: eegnet | conformer | pretrained")] = "eegnet",
) -> PredictionResponse:
    """
    Accepts a .npy (shape 105 x T) or .mat ZuCo file and returns top-5 word candidates.
    Use the `model` query parameter to select the backend.
    """
    content = await file.read()
    filename = (file.filename or "").lower()

    if filename.endswith(".mat"):
        arr = _load_mat(content)
    else:
        arr = _load_npy(content).astype(np.float32)

    _validate_eeg(arr)
    predictor = _get_or_load(model)
    return predictor.predict(arr)


@app.post("/predict/example/{example_id}", response_model=PredictionResponse, tags=["Prediction"])
def predict_example(
    example_id: int,
    model: Annotated[str, Query(description="Model backend: eegnet | conformer | pretrained")] = "eegnet",
) -> PredictionResponse:
    """Run prediction on one of the predefined test examples."""
    examples = _load_examples_manifest()
    matching = [ex for ex in examples if ex["id"] == example_id]
    if not matching:
        raise HTTPException(404, f"Example id={example_id} not found.")
    ex = matching[0]
    npy_path = config.EXAMPLES_DIR / ex["filename"]
    if not npy_path.exists():
        raise HTTPException(404, f"Example file missing: {npy_path}")
    arr = np.load(npy_path).astype(np.float32)
    _validate_eeg(arr)
    predictor = _get_or_load(model)
    return predictor.predict(arr)


@app.get("/evaluate", response_model=EvaluationResponse, tags=["Evaluation"])
def evaluate(
    model: Annotated[str, Query(description="Model backend: eegnet | conformer | pretrained")] = "eegnet",
) -> EvaluationResponse:
    """
    Evaluate the selected model on all predefined examples.
    Returns Top-1 / Top-5 accuracy, average cosine similarity, and semantic similarity.
    """
    predictor = _get_or_load(model)
    examples = _load_examples_manifest()
    if not examples:
        raise HTTPException(404, "No predefined examples found.")

    bert_matrix = np.load(str(config.BERT_EMBEDDINGS_PATH))
    with open(str(config.VOCAB_INDEX_PATH), "r", encoding="utf-8") as f:
        vocab_raw = json.load(f)
    word2idx = {v: int(k) for k, v in vocab_raw.items()}

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

        result = predictor.predict(arr)
        words = [ws.word for ws in result.top_5_words]
        scores = [ws.score for ws in result.top_5_words]
        true_w = ex["true_word"]
        pred_w = words[0]

        top1_hit = pred_w == true_w
        top5_hit = true_w in words
        if top1_hit:
            top1_hits += 1
        if top5_hit:
            top5_hits += 1
        cosine_scores.append(scores[0])

        pi = word2idx.get(pred_w)
        ti = word2idx.get(true_w)
        if pi is not None and ti is not None:
            vp, vt = bert_matrix[pi], bert_matrix[ti]
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
        raise HTTPException(500, "No examples evaluated successfully.")

    return EvaluationResponse(
        n_examples=n,
        top1_accuracy=round(top1_hits / n, 3),
        top5_accuracy=round(top5_hits / n, 3),
        avg_cosine_similarity=round(float(np.mean(cosine_scores)), 4),
        avg_semantic_similarity=round(float(np.mean(semantic_scores)), 4),
        per_example=per_example,
    )
