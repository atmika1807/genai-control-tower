"""
Celery worker — ingestion tasks
Each document ingestion runs as a Celery task so the FastAPI endpoints
return immediately with a job_id. Progress is tracked in Redis.

MLflow is used to log:
  - doc_id, source_type, chunk count, total tokens
  - embedding cost estimate (tokens × price)
  - end-to-end wall-clock time
"""

import time
import uuid
import asyncio
import mlflow
import structlog
from celery import Celery

from ingestion.config import settings
from ingestion.loader import DocumentLoader
from ingestion.chunker import ChunkingEngine
from ingestion.embedder import EmbeddingService
from ingestion.vector_store import VectorStoreClient
from shared.models import DocStatus, SourceType

log = structlog.get_logger(__name__)

# ── Celery app ────────────────────────────────────────────────────────────────

celery_app = Celery(
    "ingestion",
    broker=settings.redis_url,
    backend=settings.redis_url,
)
celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    task_track_started=True,
    worker_concurrency=settings.celery_concurrency,
    task_soft_time_limit=600,    # 10 min per document
    task_time_limit=720,
)

# ── Job state helpers (stored in Redis) ───────────────────────────────────────

import redis as _redis

_redis_client = _redis.from_url(settings.redis_url, decode_responses=True)


def _set_job_state(job_id: str, **kwargs) -> None:
    _redis_client.hset(f"job:{job_id}", mapping={k: str(v) for k, v in kwargs.items()})
    _redis_client.expire(f"job:{job_id}", 86_400)   # 24 h TTL


def get_job_state(job_id: str) -> dict:
    return _redis_client.hgetall(f"job:{job_id}")


# ── Main ingestion task ───────────────────────────────────────────────────────

@celery_app.task(bind=True, name="ingest_document")
def ingest_document(
    self,
    job_id: str,
    source_type: str,
    source_uri: str,
    metadata: dict,
    chunk_size: int       = settings.chunk_size,
    chunk_overlap: int    = settings.chunk_overlap,
    chunk_strategy: str   = settings.chunking_strategy,
) -> dict:
    """
    Full pipeline: load → chunk → embed → index.
    Runs in a Celery worker process.
    """
    doc_id   = str(uuid.uuid4())
    t_start  = time.perf_counter()

    _set_job_state(
        job_id,
        doc_id=doc_id,
        status=DocStatus.PENDING,
        chunks_total=0,
        chunks_indexed=0,
    )

    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(settings.mlflow_experiment_name)

    with mlflow.start_run(run_name=f"ingest-{doc_id[:8]}"):
        mlflow.log_params({
            "doc_id":         doc_id,
            "source_type":    source_type,
            "source_uri":     source_uri,
            "chunk_size":     chunk_size,
            "chunk_overlap":  chunk_overlap,
            "chunk_strategy": chunk_strategy,
            "embedding_model": settings.embedding_model,
        })

        try:
            # ── 1. Load ───────────────────────────────────────────────────────
            _set_job_state(job_id, status=DocStatus.CHUNKING)
            log.info("ingestion_load", job_id=job_id, uri=source_uri)

            loader = DocumentLoader()
            text = asyncio.get_event_loop().run_until_complete(
                loader.load(SourceType(source_type), source_uri)
            )
            mlflow.log_metric("doc_char_count", len(text))

            # ── 2. Chunk ──────────────────────────────────────────────────────
            chunker = ChunkingEngine(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                strategy=chunk_strategy,
            )
            chunks = chunker.chunk(
                text,
                doc_id=doc_id,
                metadata={"source_uri": source_uri, "source_type": source_type, **metadata},
            )
            total_tokens = sum(c.token_count for c in chunks)
            mlflow.log_metrics({
                "chunks_count": len(chunks),
                "total_tokens": total_tokens,
            })
            _set_job_state(job_id, status=DocStatus.EMBEDDING, chunks_total=len(chunks))

            # ── 3. Embed ──────────────────────────────────────────────────────
            log.info("ingestion_embed", job_id=job_id, n_chunks=len(chunks))
            embedder = EmbeddingService()
            embedded = asyncio.get_event_loop().run_until_complete(
                embedder.embed_chunks(chunks)
            )

            # Rough cost estimate: text-embedding-3-large ≈ $0.00013 / 1K tokens
            cost_usd = (total_tokens / 1_000) * 0.00013
            mlflow.log_metrics({
                "embedded_count": len(embedded),
                "embedding_cost_usd": round(cost_usd, 6),
            })
            _set_job_state(job_id, status=DocStatus.INDEXING)

            # ── 4. Index ──────────────────────────────────────────────────────
            log.info("ingestion_index", job_id=job_id)
            vs = VectorStoreClient()
            asyncio.get_event_loop().run_until_complete(vs.ensure_collection())
            indexed = asyncio.get_event_loop().run_until_complete(vs.upsert(embedded))

            elapsed_ms = (time.perf_counter() - t_start) * 1_000
            mlflow.log_metrics({
                "indexed_count": indexed,
                "elapsed_ms": round(elapsed_ms, 1),
            })

            _set_job_state(
                job_id,
                status=DocStatus.DONE,
                chunks_indexed=indexed,
                elapsed_ms=round(elapsed_ms, 1),
            )
            log.info(
                "ingestion_done",
                job_id=job_id,
                doc_id=doc_id,
                indexed=indexed,
                elapsed_ms=round(elapsed_ms, 1),
            )
            return {"job_id": job_id, "doc_id": doc_id, "status": DocStatus.DONE}

        except Exception as exc:
            elapsed_ms = (time.perf_counter() - t_start) * 1_000
            mlflow.log_metric("elapsed_ms", round(elapsed_ms, 1))
            mlflow.set_tag("error", str(exc))
            _set_job_state(job_id, status=DocStatus.FAILED, error=str(exc))
            log.error("ingestion_failed", job_id=job_id, error=str(exc))
            self.update_state(state="FAILURE", meta={"error": str(exc)})
            raise
