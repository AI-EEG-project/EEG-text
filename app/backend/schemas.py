"""
schemas.py — Modele Pydantic pentru API EEG-to-Text
Lupse Ioan Victor — Sapt. 14
"""

from pydantic import BaseModel, Field


class WordScore(BaseModel):
    word: str = Field(..., description="Cuvantul candidat din vocabularul de 200")
    score: float = Field(..., ge=0.0, le=1.0, description="Scor cosine similarity (0-1)")


class PredictionResponse(BaseModel):
    top_5_words: list[WordScore] = Field(
        ..., description="Top-5 cuvinte candidate cu scoruri, ordonate descrescator"
    )
    reconstructed_sentence: str = Field(
        ..., description="Propozitie reconstruita din top-1 predictii consecutive"
    )
    inference_time_ms: float = Field(
        ..., description="Timp de inferenta in milisecunde"
    )
    model_version: str = Field(
        ..., description="Versiunea modelului folosit"
    )


class HealthResponse(BaseModel):
    status: str = Field(default="ok")
    model_version: str
    using_mock: bool
    vocab_size: int
    n_channels: int


class ExampleItem(BaseModel):
    id: int = Field(..., description="Index unic al exemplului (0-4)")
    description: str = Field(..., description="Descriere afisata in UI")
    filename: str = Field(..., description="Numele fisierului .npy")
    shape: list[int] = Field(..., description="Shape array EEG: [n_channels, n_times]")
    true_word: str = Field(..., description="Cuvantul real (revelat la cerere in UI)")


class ExamplesResponse(BaseModel):
    examples: list[ExampleItem]


class EvaluationResponse(BaseModel):
    n_examples: int = Field(..., description="Numar exemple evaluate")
    top1_accuracy: float = Field(..., description="Procentul in care top-1 = cuvant real")
    top5_accuracy: float = Field(..., description="Procentul in care cuvantul real e in top-5")
    avg_cosine_similarity: float = Field(..., description="Medie scoruri cosine top-1")
    avg_semantic_similarity: float = Field(..., description="Cosine similarity semantica medie (BERT word embeddings)")
    per_example: list[dict] = Field(default_factory=list, description="Detalii per exemplu")


class ErrorDetail(BaseModel):
    detail: str
