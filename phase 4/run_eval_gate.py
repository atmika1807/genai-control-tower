"""
Eval gate runner
Called by CI to run the retrieval eval suite and write results to JSON.
Exits with code 1 if any threshold is not met.

Usage:
  python -m tests.integration.run_eval_gate \
    --min-hit-at-5 0.90 \
    --min-mrr 0.85 \
    --min-ndcg 0.83 \
    --output eval_results.json
"""

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

# Allow running as a module from the project root
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from observability.evaluator import RetrievalEvaluator, EvalSample
from tests.integration.seed_eval_corpus import EVAL_DATASET, build_retriever_and_reranker


async def run(args) -> dict:
    retriever, reranker = await build_retriever_and_reranker()

    evaluator = RetrievalEvaluator(
        retriever=retriever,
        reranker=reranker,
        tracking_uri=os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow_eval.db"),
        experiment="ci-eval-gate",
    )

    samples = [EvalSample(**s) for s in EVAL_DATASET]
    results = await evaluator.run(samples, top_k=5, run_name=f"ci-gate-{os.getenv('GITHUB_SHA','local')[:7]}")

    passed = (
        results.hit_at_5      >= args.min_hit_at_5 and
        results.mrr           >= args.min_mrr       and
        results.ndcg_at_5     >= args.min_ndcg      and
        results.avg_latency_ms < 200.0
    )

    output = {
        "passed":          passed,
        "hit_at_1":        results.hit_at_1,
        "hit_at_3":        results.hit_at_3,
        "hit_at_5":        results.hit_at_5,
        "mrr":             results.mrr,
        "ndcg_at_5":       results.ndcg_at_5,
        "avg_latency_ms":  results.avg_latency_ms,
        "n_samples":       results.n_samples,
        "thresholds": {
            "min_hit_at_5": args.min_hit_at_5,
            "min_mrr":      args.min_mrr,
            "min_ndcg":     args.min_ndcg,
            "latency_slo":  200.0,
        },
    }

    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)

    print(json.dumps(output, indent=2))
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-hit-at-5", type=float, default=0.90)
    parser.add_argument("--min-mrr",      type=float, default=0.85)
    parser.add_argument("--min-ndcg",     type=float, default=0.83)
    parser.add_argument("--output",       default="eval_results.json")
    args = parser.parse_args()

    result = asyncio.run(run(args))
    sys.exit(0 if result["passed"] else 1)
