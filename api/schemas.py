from pydantic import BaseModel


class QueryRequest(BaseModel):
    question: str
    generator: str
    embedder: str


class QueryResponse(BaseModel):
    answer: str
