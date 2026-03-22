import logging

from core.retriever.base import Retriever
from core.llm.base import Embedder
from core.vector_store.base import VectorStore

logger = logging.getLogger(__name__)


class SimpleRetriever(Retriever):
    def __init__(self, embedder: Embedder, vector_store: VectorStore):
        self.embedder = embedder
        self.vector_store = vector_store

    def retrieve(self, query: str):
        query_vec = self.embedder.embed([query])[0]
        results = self.vector_store.search(query_vec)
        logger.info(
            "SimpleRetriever search returned %d hit(s) query_len=%d",
            len(results),
            len(query),
        )
        return results
