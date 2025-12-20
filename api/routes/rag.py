from fastapi import APIRouter

from api.schemas import QueryRequest, QueryResponse
from core.engine import RAGEngine
from core.llm.registry import get_generator, get_embedder
from core.retriever.simple import SimpleRetriever
from core.vector_store.instance import vector_store

router = APIRouter(tags=["RAG"])


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

