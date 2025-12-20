from fastapi import APIRouter

from api.schemas import ChatRequest, ChatResponse
from core.llm.registry import get_generator

router = APIRouter(tags=["LLM"])


@router.post("/llm/chat", response_model=ChatResponse)
def chat_llm(request: ChatRequest):
    generator = get_generator(request.generator)
    answer = generator.generate(request.prompt)
    return ChatResponse(answer=answer)

