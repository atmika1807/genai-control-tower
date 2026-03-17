from pydantic import BaseModel, Field
from typing import Any
from enum import Enum
import uuid


class DocStatus(str, Enum):
    PENDING   = "pending"
    CHUNKING  = "chunking"
    EMBEDDING = "embedding"
    INDEXING  = "indexing"
    DONE      = "done"
    FAILED    = "failed"


class SourceType(str, Enum):
    PDF    = "pdf"
    DOCX   = "docx"
    TXT    = "txt"
    URL    = "url"
    S3     = "s3"


# ── Inbound ──────────────────────────────────────────────────────────────────

class IngestRequest(BaseModel):
    """Submitted by callers to start an ingestion job."""
    source_type: SourceType
    source_uri: str = Field(
        ...,
        description="File path, S3 URI, or HTTP URL to ingest",
        examples=["s3://my-bucket/docs/annual-report.pdf"],
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Arbitrary key-value pairs stored alongside every chunk",
    )
    priority: int = Field(default=5, ge=1, le=10)


class BatchIngestRequest(BaseModel):
    documents: list[IngestRequest]
    webhook_url: str | None = None    # called when entire batch is done


# ── Internal ─────────────────────────────────────────────────────────────────

class Chunk(BaseModel):
    chunk_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    doc_id: str
    text: str
    token_count: int
    chunk_index: int
    metadata: dict[str, Any] = Field(default_factory=dict)


class EmbeddedChunk(Chunk):
    embedding: list[float]


# ── Outbound ─────────────────────────────────────────────────────────────────

class IngestResponse(BaseModel):
    job_id: str
    status: DocStatus
    message: str


class JobStatusResponse(BaseModel):
    job_id: str
    doc_id: str | None
    status: DocStatus
    chunks_total: int | None
    chunks_indexed: int | None
    error: str | None
    elapsed_ms: float | None
