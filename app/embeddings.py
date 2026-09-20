# """
# Text -> normalized vectors, via sentence-transformers (BGE small by default).
# Runs fine on CPU for a small-to-medium course; normalize so dot-product == cosine.
# """
# from typing import List
# import numpy as np
# import torch
# from sentence_transformers import SentenceTransformer

# from app.config import EMBEDDING_MODEL_NAME

# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# _embedder: SentenceTransformer | None = None


# def load_embedder() -> SentenceTransformer:
#     global _embedder
#     if _embedder is None:
#         _embedder = SentenceTransformer(EMBEDDING_MODEL_NAME, device=DEVICE)
#         print(f"Embedder ready ({EMBEDDING_MODEL_NAME}) on {DEVICE}")
#     return _embedder


# def embed_texts(texts: List[str], batch_size: int = 64) -> np.ndarray:
#     embedder = load_embedder()
#     vecs = embedder.encode(
#         texts, batch_size=batch_size,
#         normalize_embeddings=True, show_progress_bar=len(texts) > 200,
#     )
#     return np.asarray(vecs, dtype="float32")


# def embed_query(text: str) -> np.ndarray:
#     return embed_texts([text])[0]

"""
Text -> normalized vectors, via fastembed (ONNX Runtime), NOT sentence-transformers/torch.

Why the switch: torch + sentence-transformers has a large baseline memory
footprint (200-400MB+ just for the runtime, before the model itself), which
blows past free-tier hosting limits (e.g. Render's free 512MB instances).
fastembed uses ONNX Runtime with quantized weights instead - same BGE model,
same output vectors, a fraction of the memory and no GPU/CUDA anywhere in
the dependency tree.
"""
from typing import List
import numpy as np
from fastembed import TextEmbedding

from app.config import EMBEDDING_MODEL_NAME

_embedder: TextEmbedding | None = None


def load_embedder() -> TextEmbedding:
    global _embedder
    if _embedder is None:
        # threads=1 keeps memory/CPU predictable on small/free instances
        # (e.g. Render free tier gives you 0.1 CPU) rather than letting
        # ONNX Runtime spin up threads sized for a full machine.
        _embedder = TextEmbedding(model_name=EMBEDDING_MODEL_NAME, threads=1)
        print(f"Embedder ready ({EMBEDDING_MODEL_NAME}, ONNX Runtime, CPU)")
    return _embedder


def embed_texts(texts: List[str]) -> np.ndarray:
    """Embed a batch of document/passage chunks."""
    embedder = load_embedder()
    vecs = list(embedder.embed(texts))
    return np.asarray(vecs, dtype="float32")


def embed_query(text: str) -> np.ndarray:
    """Embed a single search query (BGE uses a different prefix internally
    for queries vs. documents - query_embed handles that automatically)."""
    embedder = load_embedder()
    return np.asarray(next(embedder.query_embed(text)), dtype="float32")
