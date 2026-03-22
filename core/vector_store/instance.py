from core.vector_store.faiss_store import FaissVectorStore

# Must match the embedding vector length from your embedder.
VECTOR_DIM = 4096

vector_store = FaissVectorStore(dimension=VECTOR_DIM)

