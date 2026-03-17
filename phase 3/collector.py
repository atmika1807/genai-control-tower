"""
MLflow metrics collector
Queries MLflow tracking server and aggregates metrics for the dashboard.

Collects from two experiments:
  - ingestion-pipeline : chunk counts, token totals, embedding costs, latency
  - query-service      : query latency, LLM cost, SLO compliance, rerank scores
"""

from __future__ import annotations

import mlflow
from mlflow.tracking import MlflowClient
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import Any
import structlog

log = structlog.get_logger(__name__)


@dataclass
class IngestionStats:
    total_docs: int             = 0
    total_chunks: int           = 0
    total_tokens: int           = 0
    total_cost_usd: float       = 0.0
    avg_latency_ms: float       = 0.0
    docs_last_24h: int          = 0
    daily_doc_counts: list[dict] = field(default_factory=list)   # [{date, count}]
    cost_by_day: list[dict]      = field(default_factory=list)   # [{date, cost_usd}]


@dataclass
class QueryStats:
    total_queries: int          = 0
    avg_latency_ms: float       = 0.0
    p95_latency_ms: float       = 0.0
    p99_latency_ms: float       = 0.0
    slo_compliance_pct: float   = 0.0
    avg_cost_usd: float         = 0.0
    total_cost_usd: float       = 0.0
    avg_rerank_score: float     = 0.0
    avg_tokens_per_query: float = 0.0
    queries_last_24h: int       = 0
    latency_trend: list[dict]   = field(default_factory=list)    # [{ts, latency_ms}]
    cost_trend: list[dict]      = field(default_factory=list)    # [{date, cost_usd}]
    slo_trend: list[dict]       = field(default_factory=list)    # [{date, pct}]


class MLflowCollector:

    def __init__(self, tracking_uri: str, window_days: int = 30):
        mlflow.set_tracking_uri(tracking_uri)
        self._client     = MlflowClient()
        self._window_days = window_days
        self._cutoff_ms  = int(
            (datetime.utcnow() - timedelta(days=window_days)).timestamp() * 1000
        )

    # ── Public ────────────────────────────────────────────────────────────────

    def get_ingestion_stats(self) -> IngestionStats:
        runs = self._get_runs("ingestion-pipeline")
        if not runs:
            return IngestionStats()

        stats          = IngestionStats()
        stats.total_docs    = len(runs)
        stats.total_chunks  = sum(self._metric(r, "chunks_count")  for r in runs)
        stats.total_tokens  = sum(self._metric(r, "total_tokens")  for r in runs)
        stats.total_cost_usd = round(sum(self._metric(r, "embedding_cost_usd") for r in runs), 4)

        latencies = [self._metric(r, "elapsed_ms") for r in runs if self._metric(r, "elapsed_ms") > 0]
        stats.avg_latency_ms = round(sum(latencies) / len(latencies), 1) if latencies else 0.0

        cutoff_24h = datetime.utcnow() - timedelta(hours=24)
        stats.docs_last_24h = sum(
            1 for r in runs
            if datetime.utcfromtimestamp(r.info.start_time / 1000) >= cutoff_24h
        )

        stats.daily_doc_counts = self._bucket_by_day(runs, value_fn=lambda _: 1,   label="count")
        stats.cost_by_day      = self._bucket_by_day(runs, value_fn=lambda r: self._metric(r, "embedding_cost_usd"), label="cost_usd")
        return stats

    def get_query_stats(self) -> QueryStats:
        runs = self._get_runs("query-service")
        if not runs:
            return QueryStats()

        stats = QueryStats()
        stats.total_queries = len(runs)

        latencies = sorted([self._metric(r, "t_total_ms") for r in runs if self._metric(r, "t_total_ms") > 0])
        if latencies:
            stats.avg_latency_ms = round(sum(latencies) / len(latencies), 1)
            stats.p95_latency_ms = round(latencies[int(len(latencies) * 0.95)], 1)
            stats.p99_latency_ms = round(latencies[int(len(latencies) * 0.99)], 1)

        slo_flags = [self._metric(r, "slo_met") for r in runs]
        if slo_flags:
            stats.slo_compliance_pct = round(sum(slo_flags) / len(slo_flags) * 100, 1)

        costs = [self._metric(r, "cost_usd") for r in runs if self._metric(r, "cost_usd") > 0]
        if costs:
            stats.avg_cost_usd   = round(sum(costs) / len(costs), 6)
            stats.total_cost_usd = round(sum(costs), 4)

        rerank_scores = [self._metric(r, "top_rerank_score") for r in runs if self._metric(r, "top_rerank_score") > 0]
        if rerank_scores:
            stats.avg_rerank_score = round(sum(rerank_scores) / len(rerank_scores), 4)

        tokens = [self._metric(r, "total_tokens") for r in runs if self._metric(r, "total_tokens") > 0]
        if tokens:
            stats.avg_tokens_per_query = round(sum(tokens) / len(tokens), 1)

        cutoff_24h = datetime.utcnow() - timedelta(hours=24)
        stats.queries_last_24h = sum(
            1 for r in runs
            if datetime.utcfromtimestamp(r.info.start_time / 1000) >= cutoff_24h
        )

        stats.latency_trend = self._time_series(runs, "t_total_ms")
        stats.cost_trend    = self._bucket_by_day(runs, lambda r: self._metric(r, "cost_usd"), "cost_usd")
        stats.slo_trend     = self._slo_by_day(runs)
        return stats

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _get_runs(self, experiment_name: str) -> list:
        try:
            exp = self._client.get_experiment_by_name(experiment_name)
            if exp is None:
                return []
            return self._client.search_runs(
                experiment_ids=[exp.experiment_id],
                filter_string=f"attributes.start_time >= {self._cutoff_ms}",
                max_results=5000,
                order_by=["attributes.start_time DESC"],
            )
        except Exception as exc:
            log.warning("mlflow_fetch_failed", experiment=experiment_name, error=str(exc))
            return []

    @staticmethod
    def _metric(run, key: str, default: float = 0.0) -> float:
        return run.data.metrics.get(key, default)

    def _bucket_by_day(self, runs, value_fn, label: str) -> list[dict]:
        from collections import defaultdict
        buckets: dict[str, float] = defaultdict(float)
        for r in runs:
            day = datetime.utcfromtimestamp(r.info.start_time / 1000).strftime("%Y-%m-%d")
            buckets[day] += value_fn(r)
        return [{"date": k, label: round(v, 4)} for k, v in sorted(buckets.items())]

    def _time_series(self, runs, metric_key: str) -> list[dict]:
        points = []
        for r in runs:
            v = self._metric(r, metric_key)
            if v > 0:
                points.append({
                    "ts": r.info.start_time,
                    metric_key: round(v, 1),
                })
        return sorted(points, key=lambda x: x["ts"])

    def _slo_by_day(self, runs) -> list[dict]:
        from collections import defaultdict
        met: dict[str, int]   = defaultdict(int)
        total: dict[str, int] = defaultdict(int)
        for r in runs:
            day = datetime.utcfromtimestamp(r.info.start_time / 1000).strftime("%Y-%m-%d")
            total[day] += 1
            if self._metric(r, "slo_met") == 1.0:
                met[day] += 1
        return [
            {"date": d, "pct": round(met[d] / total[d] * 100, 1)}
            for d in sorted(total)
        ]
