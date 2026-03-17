"""
Query observer — MLflow cost + latency tracking
Logs every query as an MLflow run with:
  - query text (hashed for privacy)
  - retrieval latency, rerank latency, llm latency, total latency
  - token counts and estimated USD cost
  - whether the SLO was met (latency < 200ms)
  - top re-ranked score (proxy for retrieval quality)
"""

import time
import hashlib
import mlflow
import structlog
from dataclasses import dataclass, field
from contextlib import asynccontextmanager

from query.config import settings

log = structlog.get_logger(__name__)

# GPT-4o pricing (as of mid-2024, update as needed)
_COST_PER_1K_INPUT  = 0.005
_COST_PER_1K_OUTPUT = 0.015


@dataclass
class QueryTrace:
    query_hash: str         = ""
    t_embed_ms: float       = 0.0
    t_retrieve_ms: float    = 0.0
    t_rerank_ms: float      = 0.0
    t_llm_ms: float         = 0.0
    t_total_ms: float       = 0.0
    n_candidates: int       = 0
    n_chunks_returned: int  = 0
    top_rerank_score: float = 0.0
    prompt_tokens: int      = 0
    completion_tokens: int  = 0
    total_tokens: int       = 0
    cost_usd: float         = 0.0
    slo_met: bool           = False


class QueryObserver:

    def __init__(self) -> None:
        mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
        mlflow.set_experiment(settings.mlflow_experiment_name)

    @asynccontextmanager
    async def trace(self, query: str):
        """
        Async context manager that yields a QueryTrace object.
        Caller fills in fields during execution; we log on exit.

        Usage:
            async with observer.trace(query) as t:
                t.n_candidates = len(candidates)
                ...
        """
        trace = QueryTrace(
            query_hash=hashlib.md5(query.encode()).hexdigest()[:8]
        )
        t_start = time.perf_counter()

        try:
            yield trace
        finally:
            trace.t_total_ms = (time.perf_counter() - t_start) * 1_000
            trace.slo_met    = trace.t_total_ms <= settings.latency_slo_ms
            trace.cost_usd   = (
                (trace.prompt_tokens     / 1_000) * _COST_PER_1K_INPUT +
                (trace.completion_tokens / 1_000) * _COST_PER_1K_OUTPUT
            )

            self._log(trace)

            if not trace.slo_met:
                log.warning(
                    "slo_breach",
                    total_ms=round(trace.t_total_ms, 1),
                    slo_ms=settings.latency_slo_ms,
                )

    def _log(self, t: QueryTrace) -> None:
        try:
            with mlflow.start_run(run_name=f"query-{t.query_hash}"):
                mlflow.log_metrics({
                    "t_embed_ms":        round(t.t_embed_ms,    1),
                    "t_retrieve_ms":     round(t.t_retrieve_ms, 1),
                    "t_rerank_ms":       round(t.t_rerank_ms,   1),
                    "t_llm_ms":          round(t.t_llm_ms,      1),
                    "t_total_ms":        round(t.t_total_ms,    1),
                    "n_candidates":      t.n_candidates,
                    "n_chunks_returned": t.n_chunks_returned,
                    "top_rerank_score":  round(t.top_rerank_score, 4),
                    "prompt_tokens":     t.prompt_tokens,
                    "completion_tokens": t.completion_tokens,
                    "total_tokens":      t.total_tokens,
                    "cost_usd":          round(t.cost_usd, 6),
                    "slo_met":           int(t.slo_met),
                })
                mlflow.log_params({
                    "llm_model":      settings.llm_model,
                    "reranker_model": settings.reranker_model,
                    "retrieval_top_k": settings.retrieval_top_k,
                    "rerank_top_k":   settings.rerank_top_k,
                })
        except Exception as exc:
            log.warning("mlflow_log_failed", error=str(exc))
