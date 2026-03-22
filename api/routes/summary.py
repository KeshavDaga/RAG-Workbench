from fastapi import APIRouter, HTTPException

from api.schemas import SummaryRequest, SummaryResponse
from core.llm.registry import get_generator
from core.summarization import hierarchical_summarize
from core.vector_store.instance import vector_store

router = APIRouter(tags=["RAG"])


@router.post("/rag/summary", response_model=SummaryResponse)
def rag_summary(request: SummaryRequest):
    chunk_texts = vector_store.list_ordered_chunk_texts()
    if not chunk_texts:
        raise HTTPException(
            status_code=400,
            detail="No ingested chunks in the vector store. Ingest a video first.",
        )
    try:
        generator = get_generator(request.generator)
        summary = hierarchical_summarize(generator, chunk_texts)
        return SummaryResponse(summary=summary)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except KeyError as e:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown generator key: {e}",
        ) from e
