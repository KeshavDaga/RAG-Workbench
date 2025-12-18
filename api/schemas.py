from pydantic import BaseModel


# Request schemas
class QueryRequest(BaseModel):
    """Request schema for RAG query endpoint."""

    question: str
    generator: str
    embedder: str


class ChatRequest(BaseModel):
    """Request schema for chat endpoint."""

    prompt: str
    generator: str


# Response schemas
class QueryResponse(BaseModel):
    """Response schema for RAG query endpoint."""

    answer: str


class ChatResponse(BaseModel):
    """Response schema for chat endpoint."""

    answer: str
