import logging

from fastapi import APIRouter, HTTPException

from api.schemas import SummaryRequest, SummaryResponse
from core.llm.registry import get_generator
from core.summarization import hierarchical_summarize
from core.vector_store.instance import vector_store

logger = logging.getLogger(__name__)

router = APIRouter(tags=["RAG"])


def _http_detail(exc: BaseException) -> str:
    msg = str(exc).strip()
    if msg:
        return msg
    return f"{type(exc).__name__}: {exc!r}"


@router.post("/rag/summary", response_model=SummaryResponse)
def rag_summary(request: SummaryRequest):
    chunk_texts = vector_store.list_ordered_chunk_texts()
    if not chunk_texts:
        logger.warning("rag/summary rejected: no chunks in vector store")
        raise HTTPException(
            status_code=400,
            detail="No ingested chunks in the vector store. Ingest a video first.",
        )
    logger.info(
        "rag/summary start generator=%s chunk_count=%d",
        request.generator,
        len(chunk_texts),
    )
    try:
        generator = get_generator(request.generator)
        summary = hierarchical_summarize(generator, chunk_texts)
        logger.info("rag/summary done summary_len=%d", len(summary))
        return SummaryResponse(summary=summary)
    except ValueError as e:
        logger.warning("rag/summary validation error: %s", e)
        raise HTTPException(status_code=400, detail=str(e)) from e
    except KeyError as e:
        logger.warning("rag/summary unknown generator: %s", e)
        raise HTTPException(
            status_code=400,
            detail=f"Unknown generator key: {e}",
        ) from e
    except Exception as e:
        logger.exception("rag/summary failed")
        raise HTTPException(status_code=500, detail=_http_detail(e)) from e
