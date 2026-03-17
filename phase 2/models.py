from pydantic import BaseModel, Field
from typing import Any


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    top_k: int = Field(default=5, ge=1, le=20)
    filters: dict[str, Any] = Field(
        default_factory=dict,
        description="Metadata filters applied before retrieval, e.g. {'source_type':'pdf'}",
    )
    stream: bool = False


class RetrievedChunk(BaseModel):
    chunk_id: str
    doc_id: str
    text: str
    score: float                     # final re-ranked score
    retrieval_score: float           # raw dense score before re-ranking
    chunk_index: int
    metadata: dict[str, Any] = Field(default_factory=dict)


class QueryResponse(BaseModel):
    answer: str
    chunks: list[RetrievedChunk]     # grounding sources
    query: str
    model: str
    usage: dict[str, int]            # prompt_tokens, completion_tokens, total_tokens
    latency_ms: float
    reranker_model: str
    retrieval_top_k: int
    rerank_top_k: int


class HealthResponse(BaseModel):
    status: str
    qdrant_ok: bool
    collection: str
    vectors_count: int | None
