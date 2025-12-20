from fastapi import APIRouter

from api.routes import chat, rag, ingest

router = APIRouter()

router.include_router(chat.router)
router.include_router(rag.router)
router.include_router(ingest.router)

