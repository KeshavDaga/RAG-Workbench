from fastapi import APIRouter, HTTPException

from api.schemas import IngestRequest, IngestResponse
from core.llm.registry import get_embedder
from core.vector_store.instance import vector_store
from ingestion.youtube.ingest import ingest_youtube_video, InvalidVideoIdError

router = APIRouter(tags=["Ingestion"])


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
        raise HTTPException(status_code=400, detail=f"Invalid video ID or URL: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/ingest/clear")
def clear_vector_store():
    vector_store.clear()
    return {
        "success": True,
        "message": "Vector store cleared"
    }

