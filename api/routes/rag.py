import logging

from fastapi import APIRouter, HTTPException

from api.schemas import QueryRequest, QueryResponse
from core.engine import RAGEngine
from core.llm.registry import EMBEDDERS, GENERATORS, get_embedder, get_generator
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
    try:
        generator = get_generator(request.generator)
    except KeyError:
        logger.warning("rag/query unknown generator key=%r", request.generator)
        raise HTTPException(
            status_code=400,
            detail=f"Unknown generator {request.generator!r}. Allowed: {list(GENERATORS)}.",
        ) from None
    try:
        embedder = get_embedder(request.embedder)
    except KeyError:
        logger.warning("rag/query unknown embedder key=%r", request.embedder)
        raise HTTPException(
            status_code=400,
            detail=f"Unknown embedder {request.embedder!r}. Allowed: {list(EMBEDDERS)}.",
        ) from None

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

