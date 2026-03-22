from pydantic import BaseModel


# Request schemas
class QueryRequest(BaseModel):
    """Request schema for RAG query endpoint."""

    question: str
    generator: str
    embedder: str


class SummaryRequest(BaseModel):
    """Request schema for hierarchical video summary (uses ingested chunks only)."""

    generator: str


class ChatRequest(BaseModel):
    """Request schema for chat endpoint."""

    prompt: str
    generator: str


class IngestRequest(BaseModel):
    """Request schema for YouTube ingestion endpoint."""

    video_id: str
    embedder: str
    max_chars: int = 2000
    overlap_ratio: float = 0.1


# Response schemas
class QueryResponse(BaseModel):
    """Response schema for RAG query endpoint."""

    answer: str


class SummaryResponse(BaseModel):
    """Response schema for hierarchical summary endpoint."""

    summary: str


class ChatResponse(BaseModel):
    """Response schema for chat endpoint."""

    answer: str


class IngestResponse(BaseModel):
    """Response schema for ingestion endpoint."""

    success: bool
    message: str
