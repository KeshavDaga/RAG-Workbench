from fastapi import APIRouter

from api.routes import chat, rag

router = APIRouter()

router.include_router(chat.router)
router.include_router(rag.router)

