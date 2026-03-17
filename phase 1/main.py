"""
Ingestion Service — FastAPI app
Endpoints:
  POST /v1/ingest          — queue a single document
  POST /v1/ingest/batch    — queue up to 100 documents
  GET  /v1/jobs/{job_id}   — poll job status
  GET  /v1/index/stats     — vector store health + counts
  DELETE /v1/docs/{doc_id} — remove all chunks for a document
"""

import uuid
import structlog
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from prometheus_fastapi_instrumentator import Instrumentator

from ingestion.config import settings
from ingestion.tasks import celery_app, ingest_document, get_job_state
from ingestion.vector_store import VectorStoreClient
from shared.models import (
    IngestRequest,
    BatchIngestRequest,
    IngestResponse,
    JobStatusResponse,
    DocStatus,
)

log = structlog.get_logger(__name__)


# ── App lifecycle ─────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Ensure the Qdrant collection exists before accepting requests
    vs = VectorStoreClient()
    await vs.ensure_collection()
    log.info("ingestion_service_ready", collection=settings.collection_name)
    yield
    log.info("ingestion_service_shutdown")


app = FastAPI(
    title=settings.app_name,
    version=settings.api_version,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Prometheus metrics at /metrics
Instrumentator().instrument(app).expose(app)


# ── Routes ────────────────────────────────────────────────────────────────────

@app.post(
    "/v1/ingest",
    response_model=IngestResponse,
    status_code=202,
    summary="Queue a single document for ingestion",
)
async def ingest_single(req: IngestRequest) -> IngestResponse:
    job_id = str(uuid.uuid4())

    ingest_document.apply_async(
        kwargs={
            "job_id":      job_id,
            "source_type": req.source_type.value,
            "source_uri":  req.source_uri,
            "metadata":    req.metadata,
        },
        priority=req.priority,
        task_id=job_id,
    )

    log.info("job_queued", job_id=job_id, uri=req.source_uri)
    return IngestResponse(
        job_id=job_id,
        status=DocStatus.PENDING,
        message="Document queued for ingestion",
    )


@app.post(
    "/v1/ingest/batch",
    response_model=list[IngestResponse],
    status_code=202,
    summary="Queue up to 100 documents",
)
async def ingest_batch(req: BatchIngestRequest) -> list[IngestResponse]:
    if len(req.documents) > 100:
        raise HTTPException(status_code=422, detail="Max 100 documents per batch")

    responses = []
    for doc in req.documents:
        job_id = str(uuid.uuid4())
        ingest_document.apply_async(
            kwargs={
                "job_id":      job_id,
                "source_type": doc.source_type.value,
                "source_uri":  doc.source_uri,
                "metadata":    doc.metadata,
            },
            priority=doc.priority,
            task_id=job_id,
        )
        responses.append(
            IngestResponse(
                job_id=job_id,
                status=DocStatus.PENDING,
                message="Queued",
            )
        )

    log.info("batch_queued", count=len(responses))
    return responses


@app.post(
    "/v1/ingest/upload",
    response_model=IngestResponse,
    status_code=202,
    summary="Upload a file directly (PDF, DOCX, TXT)",
)
async def ingest_upload(
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = BackgroundTasks(),
) -> IngestResponse:
    import tempfile, os, shutil

    # Save upload to a temp file so Celery worker can read it
    suffix = os.path.splitext(file.filename or "upload")[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    # Infer source type from extension
    ext_map = {".pdf": "pdf", ".docx": "docx", ".txt": "txt"}
    source_type = ext_map.get(suffix.lower(), "txt")
    job_id = str(uuid.uuid4())

    ingest_document.apply_async(
        kwargs={
            "job_id":      job_id,
            "source_type": source_type,
            "source_uri":  tmp_path,
            "metadata":    {"original_filename": file.filename},
        },
        task_id=job_id,
    )

    return IngestResponse(
        job_id=job_id,
        status=DocStatus.PENDING,
        message=f"File '{file.filename}' queued for ingestion",
    )


@app.get(
    "/v1/jobs/{job_id}",
    response_model=JobStatusResponse,
    summary="Poll job status",
)
async def job_status(job_id: str) -> JobStatusResponse:
    state = get_job_state(job_id)
    if not state:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    return JobStatusResponse(
        job_id=job_id,
        doc_id=state.get("doc_id"),
        status=DocStatus(state.get("status", DocStatus.PENDING)),
        chunks_total=int(state["chunks_total"]) if state.get("chunks_total") else None,
        chunks_indexed=int(state["chunks_indexed"]) if state.get("chunks_indexed") else None,
        error=state.get("error"),
        elapsed_ms=float(state["elapsed_ms"]) if state.get("elapsed_ms") else None,
    )


@app.get(
    "/v1/index/stats",
    summary="Vector store health and document counts",
)
async def index_stats() -> dict:
    vs = VectorStoreClient()
    stats = await vs.collection_stats()
    return {"collection": settings.collection_name, **stats}


@app.delete(
    "/v1/docs/{doc_id}",
    status_code=204,
    summary="Delete all chunks for a document",
)
async def delete_document(doc_id: str) -> None:
    vs = VectorStoreClient()
    await vs.delete_by_doc_id(doc_id)


@app.get("/health")
async def health() -> dict:
    return {"status": "ok", "service": "ingestion"}
