import logging

from fastapi import APIRouter, HTTPException

from api.schemas import IngestRequest, IngestResponse
from core.llm.registry import get_embedder
from core.vector_store.instance import vector_store
from ingestion.youtube.fetch import YoutubeTranscriptError
from ingestion.youtube.ingest import ingest_youtube_video, InvalidVideoIdError

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Ingestion"])


def _http_detail(exc: BaseException) -> str:
    """Avoid empty API error bodies when str(exc) is blank (common with some HTTP/SDK errors)."""
    msg = str(exc).strip()
    if msg:
        return msg
    return f"{type(exc).__name__}: {exc!r}"


@router.post("/ingest/youtube", response_model=IngestResponse)
def ingest_youtube(request: IngestRequest):
    try:
        embedder = get_embedder(request.embedder)

        ingest_youtube_video(
            video_id_or_url=request.video_id,
            embedder=embedder,
            vector_store=vector_store,
            max_chars=request.max_chars,
            overlap_ratio=request.overlap_ratio,
        )

        return IngestResponse(
            success=True,
            message=f"Successfully ingested video: {request.video_id}",
        )
    except InvalidVideoIdError as e:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid video ID or URL: {_http_detail(e)}",
        ) from e
    except KeyError as e:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown embedder key: {e!s}. Use a key from the registry (e.g. qwen3).",
        ) from e
    except YoutubeTranscriptError as e:
        raise HTTPException(status_code=400, detail=_http_detail(e)) from e
    except Exception as e:
        logger.exception("Ingest failed for video_id=%s", request.video_id)
        raise HTTPException(status_code=400, detail=_http_detail(e)) from e


@router.post("/ingest/clear")
def clear_vector_store():
    vector_store.clear()
    return {
        "success": True,
        "message": "Vector store cleared"
    }

