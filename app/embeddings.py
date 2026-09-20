"""
Text -> normalized vectors, via sentence-transformers (BGE small by default).
Runs fine on CPU for a small-to-medium course; normalize so dot-product == cosine.
"""
from typing import List
import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from app.config import EMBEDDING_MODEL_NAME

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
_embedder: SentenceTransformer | None = None


def load_embedder() -> SentenceTransformer:
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer(EMBEDDING_MODEL_NAME, device=DEVICE)
        print(f"Embedder ready ({EMBEDDING_MODEL_NAME}) on {DEVICE}")
    return _embedder


def embed_texts(texts: List[str], batch_size: int = 64) -> np.ndarray:
    embedder = load_embedder()
    vecs = embedder.encode(
        texts, batch_size=batch_size,
        normalize_embeddings=True, show_progress_bar=len(texts) > 200,
    )
    return np.asarray(vecs, dtype="float32")


def embed_query(text: str) -> np.ndarray:
    return embed_texts([text])[0]
