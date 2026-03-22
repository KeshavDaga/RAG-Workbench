from abc import ABC, abstractmethod
from typing import List, Any


class VectorStore(ABC):
    @abstractmethod
    def add(
        self,
        vectors: List[List[float]],
        metadatas: List[Any],
    ) -> None:
        """
        Store vectors with associated metadata (text, timestamps, file info, etc.)
        """
        pass

    @abstractmethod
    def search(
        self,
        query_vector: List[float],
        top_k: int = 5,
    ) -> List[Any]:
        """
        Return top_k most similar metadata entries
        """
        pass

    @abstractmethod
    def clear(self) -> None:
        """
        Clear all vectors and metadata from the vector store
        """
        pass