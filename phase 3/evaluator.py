"""
Retrieval accuracy eval suite
Measures retrieval quality against a labeled evaluation set.

Metrics:
  Hit@K     — fraction of queries where the gold chunk appears in top-K
  MRR       — Mean Reciprocal Rank (1/rank of first relevant result)
  nDCG@K    — Normalized Discounted Cumulative Gain at K

Usage:
  evaluator = RetrievalEvaluator(retriever, reranker)
  results   = await evaluator.run(eval_dataset)
  # results logged to MLflow automatically
"""

from __future__ import annotations

import asyncio
import math
import time
import mlflow
import structlog
from dataclasses import dataclass

log = structlog.get_logger(__name__)


@dataclass
class EvalSample:
    query: str
    relevant_chunk_ids: list[str]   # ground-truth chunk IDs
    metadata: dict = None


@dataclass
class EvalResults:
    hit_at_1: float  = 0.0
    hit_at_3: float  = 0.0
    hit_at_5: float  = 0.0
    mrr: float       = 0.0
    ndcg_at_5: float = 0.0
    n_samples: int   = 0
    avg_latency_ms: float = 0.0


class RetrievalEvaluator:

    def __init__(self, retriever, reranker, tracking_uri: str, experiment: str = "retrieval-evals"):
        self._retriever = retriever
        self._reranker  = reranker
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment)

    async def run(
        self,
        samples: list[EvalSample],
        top_k: int = 5,
        run_name: str = "eval",
    ) -> EvalResults:
        log.info("eval_start", n_samples=len(samples), top_k=top_k)

        hits_1, hits_3, hits_5, rr_scores, ndcg_scores, latencies = [], [], [], [], [], []

        for sample in samples:
            t0         = time.perf_counter()
            candidates = await self._retriever.retrieve(sample.query, top_k=top_k * 4)
            reranked   = await asyncio.get_event_loop().run_in_executor(
                None, self._reranker.rerank, sample.query, candidates, top_k
            )
            latencies.append((time.perf_counter() - t0) * 1_000)

            retrieved_ids = [c["chunk_id"] for c in reranked]
            relevant      = set(sample.relevant_chunk_ids)

            hits_1.append(1 if any(i in relevant for i in retrieved_ids[:1]) else 0)
            hits_3.append(1 if any(i in relevant for i in retrieved_ids[:3]) else 0)
            hits_5.append(1 if any(i in relevant for i in retrieved_ids[:5]) else 0)

            # MRR
            rr = 0.0
            for rank, cid in enumerate(retrieved_ids, start=1):
                if cid in relevant:
                    rr = 1.0 / rank
                    break
            rr_scores.append(rr)

            # nDCG@K
            dcg, idcg = 0.0, 0.0
            for rank, cid in enumerate(retrieved_ids[:top_k], start=1):
                rel = 1.0 if cid in relevant else 0.0
                dcg += rel / math.log2(rank + 1)
            for rank in range(1, min(len(relevant), top_k) + 1):
                idcg += 1.0 / math.log2(rank + 1)
            ndcg_scores.append(dcg / idcg if idcg > 0 else 0.0)

        n = len(samples)
        results = EvalResults(
            hit_at_1=round(sum(hits_1) / n, 4),
            hit_at_3=round(sum(hits_3) / n, 4),
            hit_at_5=round(sum(hits_5) / n, 4),
            mrr=round(sum(rr_scores) / n, 4),
            ndcg_at_5=round(sum(ndcg_scores) / n, 4),
            n_samples=n,
            avg_latency_ms=round(sum(latencies) / n, 1),
        )

        self._log_to_mlflow(results, run_name)
        log.info("eval_done", **results.__dict__)
        return results

    def _log_to_mlflow(self, r: EvalResults, run_name: str) -> None:
        with mlflow.start_run(run_name=run_name):
            mlflow.log_metrics({
                "hit_at_1":        r.hit_at_1,
                "hit_at_3":        r.hit_at_3,
                "hit_at_5":        r.hit_at_5,
                "mrr":             r.mrr,
                "ndcg_at_5":       r.ndcg_at_5,
                "n_samples":       r.n_samples,
                "avg_latency_ms":  r.avg_latency_ms,
            })
