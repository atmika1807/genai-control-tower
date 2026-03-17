"""
Cross-encoder re-ranker
Uses sentence-transformers ms-marco-MiniLM-L-6-v2 to score
(query, chunk) pairs and return the top-k most relevant chunks.

Why cross-encoder over bi-encoder for re-ranking?
  Bi-encoders (used in retrieval) encode query and document independently.
  Cross-encoders see the query AND document together — full attention across
  both — giving much more accurate relevance scores. The trade-off is speed:
  cross-encoders are ~10-50× slower, so we only run them on the top-20
  candidates from the retriever, not the whole corpus.

Latency budget:
  MiniLM-L-6 on CPU: ~15-30ms for 20 pairs
  This keeps total query latency under 200ms alongside embedding + LLM.
"""

import structlog
from functools import lru_cache
from sentence_transformers import CrossEncoder

from query.config import settings

log = structlog.get_logger(__name__)


@lru_cache(maxsize=1)
def _load_model() -> CrossEncoder:
    """Load once, reuse across requests. Thread-safe after first call."""
    log.info("loading_reranker", model=settings.reranker_model)
    model = CrossEncoder(
        settings.reranker_model,
        device=settings.reranker_device,
        max_length=512,
    )
    log.info("reranker_loaded")
    return model


class CrossEncoderReranker:

    def __init__(self) -> None:
        self._model = _load_model()

    def rerank(
        self,
        query: str,
        candidates: list[dict],
        top_k: int = settings.rerank_top_k,
    ) -> list[dict]:
        """
        Score each (query, chunk_text) pair and return the top_k
        candidates sorted by cross-encoder score (desc).

        Input:  list of dicts with at least {"chunk_id", "text", ...}
        Output: same dicts with "score" (cross-encoder) and
                "retrieval_score" (original RRF score) preserved.
        """
        if not candidates:
            return []

        pairs = [(query, c["text"]) for c in candidates]
        scores = self._model.predict(pairs, show_progress_bar=False)

        ranked = sorted(
            zip(candidates, scores),
            key=lambda x: x[1],
            reverse=True,
        )[:top_k]

        result = []
        for chunk, score in ranked:
            result.append({
                **chunk,
                "score": float(score),
                # preserve original retrieval score for debugging
                "retrieval_score": chunk.get("retrieval_score", 0.0),
            })

        log.info(
            "rerank_done",
            input_candidates=len(candidates),
            output_top_k=len(result),
            top_score=round(float(ranked[0][1]), 4) if ranked else None,
        )
        return result
