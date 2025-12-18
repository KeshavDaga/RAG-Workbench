from fastapi import APIRouter

from api.schemas import QueryRequest, QueryResponse
from core.engine import RAGEngine
from core.llm.registry import get_generator, get_embedder
from core.retriever.simple import SimpleRetriever
from core.vector_store.faiss_store import FaissVectorStore


# TEMP setup (will improve later)
VECTOR_DIM = 1024
vector_store = FaissVectorStore(dimension=VECTOR_DIM)

router = APIRouter()

@router.get("/health")
def health():
    return {"status": "ok"}

@router.post("/rag/query", response_model=QueryResponse)
def query_rag(request: QueryRequest):
    generator = get_generator(request.generator)
    embedder = get_embedder(request.embedder)

    retriever = SimpleRetriever(
        embedder=embedder,
        vector_store=vector_store,
    )

    engine = RAGEngine(
        generator=generator,
        retriever=retriever,
    )

    answer = engine.query(request.question)

    return QueryResponse(answer=answer)
