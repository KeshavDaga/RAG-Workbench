import logging

from fastapi import APIRouter

from api.schemas import ChatRequest, ChatResponse
from core.llm.registry import get_generator

logger = logging.getLogger(__name__)

router = APIRouter(tags=["LLM"])


@router.post("/llm/chat", response_model=ChatResponse)
def chat_llm(request: ChatRequest):
    logger.info(
        "llm/chat generator=%s prompt_len=%d",
        request.generator,
        len(request.prompt),
    )
    generator = get_generator(request.generator)
    answer = generator.generate(request.prompt)
    logger.info("llm/chat done answer_len=%d", len(answer))
    return ChatResponse(answer=answer)

