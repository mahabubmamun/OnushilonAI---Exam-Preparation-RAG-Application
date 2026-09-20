"""
FAISS-backed vector store with save/load, so an index survives a server
restart instead of living only in notebook memory.
"""
from pathlib import Path
from typing import List, Tuple

import numpy as np
import faiss


class VectorStore:
    """Stores normalized vectors; searches by cosine similarity (inner product)."""

    def __init__(self, dim: int):
        self.dim = dim
        self.index = faiss.IndexFlatIP(dim)

    def add(self, vectors: np.ndarray) -> List[int]:
        start = self.size
        self.index.add(vectors)
        return list(range(start, start + len(vectors)))

    def search(self, query_vec: np.ndarray, top_k: int) -> Tuple[List[float], List[int]]:
        if self.size == 0:
            return [], []
        k = min(top_k, self.size)
        scores, ids = self.index.search(query_vec.reshape(1, -1), k)
        return scores[0].tolist(), ids[0].tolist()

    @property
    def size(self) -> int:
        return self.index.ntotal

    def save(self, path: Path):
        faiss.write_index(self.index, str(path))

    @classmethod
    def load(cls, path: Path, dim: int) -> "VectorStore":
        store = cls(dim)
        store.index = faiss.read_index(str(path))
        return store
