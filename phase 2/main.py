"""
Query Service — FastAPI app
Endpoints:
  POST /v1/query         — single query, full JSON response
  POST /v1/query/stream  — SSE token stream for real-time UI
  GET  /v1/health        — Qdrant + collection health check
  GET  /metrics          — Prometheus metrics

Full pipeline per request:
  embed query → hybrid retrieve (dense + BM25 + RRF) →
  cross-encoder re-rank → LLM synthesis → MLflow trace
"""

import time
import json
import asyncio
import structlog
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from prometheus_fastapi_instrumentator import Instrumentator
from qdrant_client import AsyncQdrantClient

from query.config import settings
from query.retriever import HybridRetriever
from query.reranker import CrossEncoderReranker
from query.synthesizer import LLMSynthesizer
from query.observer import QueryObserver
from query.models import (
    QueryRequest,
    QueryResponse,
    RetrievedChunk,
    HealthResponse,
)

log = structlog.get_logger(__name__)

# ── Singletons (created once at startup) ─────────────────────────────────────
# CrossEncoder model load is slow (~2s); do it at startup, not per-request.

_retriever:   HybridRetriever      | None = None
_reranker:    CrossEncoderReranker | None = None
_synthesizer: LLMSynthesizer       | None = None
_observer:    QueryObserver        | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _retriever, _reranker, _synthesizer, _observer

    log.info("query_service_starting")
    _retriever   = HybridRetriever()
    _reranker    = CrossEncoderReranker()    # loads cross-encoder model
    _synthesizer = LLMSynthesizer()
    _observer    = QueryObserver()
    log.info("query_service_ready")
    yield
    log.info("query_service_shutdown")


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

Instrumentator().instrument(app).expose(app)


# ── Latency header middleware ─────────────────────────────────────────────────

@app.middleware("http")
async def add_latency_header(request: Request, call_next):
    t0 = time.perf_counter()
    response = await call_next(request)
    elapsed_ms = (time.perf_counter() - t0) * 1_000
    response.headers["X-Latency-Ms"] = f"{elapsed_ms:.1f}"
    if elapsed_ms > settings.latency_slo_ms:
        response.headers["X-SLO-Breach"] = "true"
    return response


# ── Core pipeline ─────────────────────────────────────────────────────────────

async def _run_pipeline(
    req: QueryRequest,
    trace,
) -> tuple[list[dict], str, dict]:
    """
    Shared pipeline used by both /query and /query/stream.
    Returns (reranked_chunks, answer, usage).
    """
    # 1. Hybrid retrieval
    t0 = time.perf_counter()
    candidates = await _retriever.retrieve(
        query=req.query,
        top_k=req.top_k * 4,       # retrieve 4× top_k, re-rank down to top_k
        filters=req.filters or None,
    )
    trace.t_retrieve_ms = (time.perf_counter() - t0) * 1_000
    trace.n_candidates  = len(candidates)

    if not candidates:
        return [], "I don't have enough information in the provided documents to answer this question.", {}

    # 2. Cross-encoder re-rank
    t1 = time.perf_counter()
    reranked = await asyncio.get_event_loop().run_in_executor(
        None,                           # default thread pool
        _reranker.rerank,
        req.query,
        candidates,
        req.top_k,
    )
    trace.t_rerank_ms       = (time.perf_counter() - t1) * 1_000
    trace.n_chunks_returned = len(reranked)
    trace.top_rerank_score  = reranked[0]["score"] if reranked else 0.0

    return reranked, None, None     # answer filled by caller


# ── Routes ────────────────────────────────────────────────────────────────────

@app.post(
    "/v1/query",
    response_model=QueryResponse,
    summary="Single RAG query — returns full JSON response",
)
async def query(req: QueryRequest) -> QueryResponse:
    async with _observer.trace(req.query) as trace:

        reranked, _, _ = await _run_pipeline(req, trace)

        # 3. LLM synthesis
        t2 = time.perf_counter()
        answer, usage = await _synthesizer.synthesize(req.query, reranked)
        trace.t_llm_ms         = (time.perf_counter() - t2) * 1_000
        trace.prompt_tokens    = usage.get("prompt_tokens", 0)
        trace.completion_tokens = usage.get("completion_tokens", 0)
        trace.total_tokens     = usage.get("total_tokens", 0)

    chunks_out = [
        RetrievedChunk(
            chunk_id=c["chunk_id"],
            doc_id=c["doc_id"],
            text=c["text"],
            score=c["score"],
            retrieval_score=c["retrieval_score"],
            chunk_index=c["chunk_index"],
            metadata=c.get("metadata", {}),
        )
        for c in reranked
    ]

    return QueryResponse(
        answer=answer,
        chunks=chunks_out,
        query=req.query,
        model=settings.llm_model,
        usage=usage,
        latency_ms=round(trace.t_total_ms, 1),
        reranker_model=settings.reranker_model,
        retrieval_top_k=settings.retrieval_top_k,
        rerank_top_k=req.top_k,
    )


@app.post(
    "/v1/query/stream",
    summary="Streaming RAG query — SSE token stream",
    response_class=StreamingResponse,
)
async def query_stream(req: QueryRequest) -> StreamingResponse:
    """
    Server-Sent Events stream.
    First event: {"type":"chunks","data":[...]}  — grounding sources
    Subsequent:  {"type":"token","data":"..."}    — LLM tokens
    Final:       {"type":"done","data":""}
    """

    async def event_stream() -> AsyncIterator[str]:
        async with _observer.trace(req.query) as trace:
            reranked, early_answer, _ = await _run_pipeline(req, trace)

            # Send grounding sources first
            chunks_payload = [
                {
                    "chunk_id":       c["chunk_id"],
                    "doc_id":         c["doc_id"],
                    "score":          round(c["score"], 4),
                    "retrieval_score": round(c["retrieval_score"], 4),
                    "metadata":       c.get("metadata", {}),
                }
                for c in reranked
            ]
            yield f"data: {json.dumps({'type':'chunks','data':chunks_payload})}\n\n"

            if early_answer:
                yield f"data: {json.dumps({'type':'token','data':early_answer})}\n\n"
            else:
                t2 = time.perf_counter()
                async for token in _synthesizer.synthesize_stream(req.query, reranked):
                    yield f"data: {json.dumps({'type':'token','data':token})}\n\n"
                trace.t_llm_ms = (time.perf_counter() - t2) * 1_000

            yield f"data: {json.dumps({'type':'done','data':''})}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@app.get(
    "/v1/health",
    response_model=HealthResponse,
    summary="Qdrant + collection health",
)
async def health() -> HealthResponse:
    try:
        client = AsyncQdrantClient(
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key,
            timeout=5,
        )
        info = await client.get_collection(settings.collection_name)
        return HealthResponse(
            status="ok",
            qdrant_ok=True,
            collection=settings.collection_name,
            vectors_count=info.vectors_count,
        )
    except Exception as exc:
        log.error("health_check_failed", error=str(exc))
        return HealthResponse(
            status="degraded",
            qdrant_ok=False,
            collection=settings.collection_name,
            vectors_count=None,
        )
