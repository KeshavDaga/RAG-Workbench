import logging

from fastapi import APIRouter

from api.schemas import QueryRequest, QueryResponse
from core.engine import RAGEngine
from core.llm.registry import get_generator, get_embedder
from core.retriever.simple import SimpleRetriever
from core.vector_store.instance import vector_store

logger = logging.getLogger(__name__)

router = APIRouter(tags=["RAG"])


@router.post("/rag/query", response_model=QueryResponse)
def query_rag(request: QueryRequest):
    logger.info(
        "rag/query generator=%s embedder=%s question_len=%d",
        request.generator,
        request.embedder,
        len(request.question),
    )
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
    logger.info(
        "rag/query done answer_len=%d preview=%s",
        len(answer),
        (answer[:120] + "…") if len(answer) > 120 else answer,
    )
    return QueryResponse(answer=answer)

