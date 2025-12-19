import faiss
import numpy as np
from typing import List, Any

from core.vector_store.base import VectorStore

class FaissVectorStore(VectorStore):
    def __init__(self, dimension: int):
        self.dimension = dimension
        self.index = faiss.IndexFlatIP(dimension)
        self.metadatas: List[Any] = []

    def add(self, vectors: List[List[float]], metadatas: List[Any]) -> None:
        if len(vectors) != len(metadatas):
            raise ValueError("Vectors and metadata length mismatch")

        vectors_np = np.array(vectors).astype("float32")

        faiss.normalize_L2(vectors_np)

        self.index.add(vectors_np)
        self.metadatas.extend(metadatas)

    def search(self, query_vector: List[float], top_k: int = 5) -> List[Any]:
        if self.index.ntotal == 0:
            return []

        query_np = np.array([query_vector]).astype("float32")
        faiss.normalize_L2(query_np)

        scores, indices = self.index.search(query_np, top_k)
        return [self.metadatas[idx] for idx in indices[0]]

    def clear(self) -> None:
        self.index.reset()
        self.metadatas.clear()