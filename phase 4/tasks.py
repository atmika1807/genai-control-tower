"""
Celery tasks — production config
Priority queues · dead letter · exponential backoff · periodic maintenance
"""

import time, uuid, asyncio, mlflow, structlog
from celery import Celery
from celery.schedules import crontab
from kombu import Queue, Exchange

from ingestion.config import settings
from ingestion.loader import DocumentLoader
from ingestion.chunker import ChunkingEngine
from ingestion.embedder import EmbeddingService
from ingestion.vector_store import VectorStoreClient
from shared.models import DocStatus, SourceType

log = structlog.get_logger(__name__)

# ── Queue topology ────────────────────────────────────────────────────────────

default_exchange = Exchange("default",     type="direct")
high_exchange    = Exchange("high",        type="direct")
dl_exchange      = Exchange("dead_letter", type="direct")

celery_app = Celery("ingestion", broker=settings.redis_url, backend=settings.redis_url)

celery_app.conf.update(
    task_serializer="json", result_serializer="json", accept_content=["json"],
    task_queues=(
        Queue("high",        high_exchange,    routing_key="high",
              queue_arguments={"x-max-priority": 10}),
        Queue("celery",      default_exchange, routing_key="celery",
              queue_arguments={"x-max-priority": 7}),
        Queue("dead_letter", dl_exchange,      routing_key="dead_letter"),
    ),
    task_default_queue="celery",
    task_routes={"ingest_document": {"queue": "high", "routing_key": "high"}},
    # Reliability
    task_track_started=True,
    task_acks_late=True,
    task_reject_on_worker_lost=True,
    worker_prefetch_multiplier=1,
    # Timeouts
    task_soft_time_limit=600,
    task_time_limit=720,
    result_expires=86_400,
    # Periodic tasks
    beat_schedule={
        "refresh-bm25-index":  {"task": "refresh_bm25_index",  "schedule": crontab(minute=0)},
        "cleanup-stale-jobs":  {"task": "cleanup_stale_jobs",   "schedule": crontab(minute=0, hour="*/6")},
    },
)

# ── Redis state helpers ───────────────────────────────────────────────────────

import redis as _redis
_redis_client = _redis.from_url(settings.redis_url, decode_responses=True)

def _set_job_state(job_id: str, **kwargs) -> None:
    _redis_client.hset(f"job:{job_id}", mapping={k: str(v) for k, v in kwargs.items()})
    _redis_client.expire(f"job:{job_id}", 86_400)

def get_job_state(job_id: str) -> dict:
    return _redis_client.hgetall(f"job:{job_id}")

# ── Ingestion task ────────────────────────────────────────────────────────────

@celery_app.task(bind=True, name="ingest_document", max_retries=3, default_retry_delay=30)
def ingest_document(self, job_id, source_type, source_uri, metadata,
                    chunk_size=settings.chunk_size, chunk_overlap=settings.chunk_overlap,
                    chunk_strategy=settings.chunking_strategy):
    doc_id, t_start = str(uuid.uuid4()), time.perf_counter()
    _set_job_state(job_id, doc_id=doc_id, status=DocStatus.PENDING, chunks_total=0, chunks_indexed=0)

    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(settings.mlflow_experiment_name)

    with mlflow.start_run(run_name=f"ingest-{doc_id[:8]}"):
        mlflow.log_params({"doc_id": doc_id, "source_type": source_type,
                           "chunk_size": chunk_size, "attempt": self.request.retries + 1})
        try:
            _set_job_state(job_id, status=DocStatus.CHUNKING)
            loop = asyncio.get_event_loop()

            text    = loop.run_until_complete(DocumentLoader().load(SourceType(source_type), source_uri))
            chunks  = ChunkingEngine(chunk_size, chunk_overlap, chunk_strategy).chunk(
                        text, doc_id=doc_id, metadata={"source_uri": source_uri, **metadata})
            tokens  = sum(c.token_count for c in chunks)
            mlflow.log_metrics({"chunks_count": len(chunks), "total_tokens": tokens})
            _set_job_state(job_id, status=DocStatus.EMBEDDING, chunks_total=len(chunks))

            embedded = loop.run_until_complete(EmbeddingService().embed_chunks(chunks))
            cost_usd = (tokens / 1_000) * 0.00013
            mlflow.log_metrics({"embedded_count": len(embedded), "embedding_cost_usd": round(cost_usd, 6)})
            _set_job_state(job_id, status=DocStatus.INDEXING)

            vs = VectorStoreClient()
            loop.run_until_complete(vs.ensure_collection())
            indexed    = loop.run_until_complete(vs.upsert(embedded))
            elapsed_ms = (time.perf_counter() - t_start) * 1_000
            mlflow.log_metrics({"indexed_count": indexed, "elapsed_ms": round(elapsed_ms, 1)})

            _set_job_state(job_id, status=DocStatus.DONE, chunks_indexed=indexed, elapsed_ms=round(elapsed_ms, 1))
            log.info("ingestion_done", job_id=job_id, indexed=indexed)
            return {"job_id": job_id, "doc_id": doc_id, "status": DocStatus.DONE}

        except Exception as exc:
            mlflow.set_tag("error", str(exc))
            log.error("ingestion_failed", job_id=job_id, attempt=self.request.retries + 1, error=str(exc))

            if self.request.retries < self.max_retries:
                delay = 30 * (2 ** self.request.retries)   # 30s, 60s, 120s
                _set_job_state(job_id, status=DocStatus.PENDING, error=f"retry {self.request.retries+1}: {exc}")
                raise self.retry(exc=exc, countdown=delay)

            # All retries exhausted → dead letter
            _set_job_state(job_id, status=DocStatus.FAILED, error=str(exc))
            celery_app.send_task("handle_dead_letter", kwargs={"job_id": job_id, "error": str(exc)}, queue="dead_letter")
            raise

# ── Periodic tasks ────────────────────────────────────────────────────────────

@celery_app.task(name="refresh_bm25_index")
def refresh_bm25_index():
    import httpx
    try:
        httpx.post("http://query-api:8001/internal/refresh-bm25", timeout=10).raise_for_status()
        log.info("bm25_refresh_ok")
        return {"status": "ok"}
    except Exception as exc:
        log.warning("bm25_refresh_failed", error=str(exc))
        return {"status": "error", "error": str(exc)}

@celery_app.task(name="cleanup_stale_jobs")
def cleanup_stale_jobs():
    deleted, cursor = 0, 0
    while True:
        cursor, keys = _redis_client.scan(cursor, match="job:*", count=200)
        for key in keys:
            if _redis_client.hget(key, "status") == DocStatus.PENDING and _redis_client.ttl(key) < 0:
                _redis_client.delete(key)
                deleted += 1
        if cursor == 0:
            break
    log.info("cleanup_done", deleted=deleted)
    return {"deleted": deleted}

@celery_app.task(name="handle_dead_letter")
def handle_dead_letter(job_id: str, error: str):
    log.error("dead_letter", job_id=job_id, error=error)
    _redis_client.expire(f"job:{job_id}", 7 * 86_400)  # keep 7 days for inspection
