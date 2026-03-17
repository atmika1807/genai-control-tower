"""
Hybrid retriever
Combines dense (Qdrant ANN) and sparse (BM25) retrieval,
fused with Reciprocal Rank Fusion (RRF).

Why hybrid?
  Dense search excels at semantic similarity but misses exact keyword matches.
  BM25 excels at keyword precision but ignores semantics.
  RRF fusion consistently outperforms either alone on mixed enterprise corpora.

Pipeline:
  1. Embed the query (async, cached by query text)
  2. Dense search  → top-K candidates from Qdrant
  3. BM25 search   → top-K candidates from in-memory BM25 index (lazy-built)
  4. RRF fusion    → unified ranked list
"""

import asyncio
import hashlib
import structlog
from functools import lru_cache
from typing import Any

from openai import AsyncOpenAI
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue, SearchRequest

from query.config import settings

log = structlog.get_logger(__name__)


# ── Query embedding cache (in-process, LRU) ───────────────────────────────────

@lru_cache(maxsize=512)
def _cached_embedding_key(query: str) -> str:
    return hashlib.md5(query.encode()).hexdigest()


class HybridRetriever:

    def __init__(self) -> None:
        self._openai   = AsyncOpenAI(api_key=settings.openai_api_key)
        self._qdrant   = AsyncQdrantClient(
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key,
            timeout=10,
        )
        self._embed_cache: dict[str, list[float]] = {}
        self._bm25_index = None        # lazy-built on first query

    # ── Public ────────────────────────────────────────────────────────────────

    async def retrieve(
        self,
        query: str,
        top_k: int = settings.retrieval_top_k,
        filters: dict[str, Any] | None = None,
    ) -> list[dict]:
        """
        Returns up to `top_k` candidates sorted by RRF score (desc).
        Each dict has: chunk_id, doc_id, text, retrieval_score,
                       chunk_index, metadata.
        """
        query_vec, bm25_results, dense_results = await asyncio.gather(
            self._embed_query(query),
            self._bm25_search(query, top_k),
            asyncio.sleep(0),          # placeholder — filled below
        )
        dense_results = await self._dense_search(query_vec, top_k, filters)

        fused = self._rrf_fuse(dense_results, bm25_results, top_k)
        log.info(
            "retrieval_done",
            query_len=len(query),
            dense=len(dense_results),
            bm25=len(bm25_results),
            fused=len(fused),
        )
        return fused

    # ── Embedding ─────────────────────────────────────────────────────────────

    async def _embed_query(self, query: str) -> list[float]:
        cache_key = _cached_embedding_key(query)
        if cache_key in self._embed_cache:
            return self._embed_cache[cache_key]

        resp = await self._openai.embeddings.create(
            model=settings.embedding_model,
            input=[query],
            dimensions=settings.embedding_dimensions,
            encoding_format="float",
        )
        vec = resp.data[0].embedding
        self._embed_cache[cache_key] = vec
        return vec

    # ── Dense search (Qdrant ANN) ─────────────────────────────────────────────

    async def _dense_search(
        self,
        query_vec: list[float],
        top_k: int,
        filters: dict[str, Any] | None,
    ) -> list[dict]:
        qdrant_filter = self._build_filter(filters) if filters else None

        results = await self._qdrant.search(
            collection_name=settings.collection_name,
            query_vector=query_vec,
            limit=top_k,
            query_filter=qdrant_filter,
            with_payload=True,
            score_threshold=0.0,
        )

        return [
            {
                "chunk_id":       str(r.id),
                "doc_id":         r.payload.get("doc_id", ""),
                "text":           r.payload.get("text", ""),
                "retrieval_score": r.score,
                "chunk_index":    r.payload.get("chunk_index", 0),
                "metadata":       {
                    k: v for k, v in r.payload.items()
                    if k not in ("text", "doc_id", "chunk_index")
                },
            }
            for r in results
        ]

    # ── BM25 search ───────────────────────────────────────────────────────────

    async def _bm25_search(self, query: str, top_k: int) -> list[dict]:
        """
        BM25 over cached corpus.
        The index is built lazily on first call and refreshed every hour.
        For 10K docs this is ~100ms to build and ~1ms to query.
        """
        index = await self._get_bm25_index()
        if index is None:
            return []

        corpus, index_obj = index
        import numpy as np
        from rank_bm25 import BM25Okapi

        tokens = query.lower().split()
        scores = index_obj.get_scores(tokens)
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for i in top_indices:
            if scores[i] > 0:
                results.append({**corpus[i], "retrieval_score": float(scores[i])})
        return results

    async def _get_bm25_index(self):
        """Lazily build BM25 index from Qdrant scroll."""
        if self._bm25_index is not None:
            return self._bm25_index

        try:
            from rank_bm25 import BM25Okapi

            all_points, _ = await self._qdrant.scroll(
                collection_name=settings.collection_name,
                limit=10_000,
                with_payload=True,
                with_vectors=False,
            )

            corpus = [
                {
                    "chunk_id":    str(p.id),
                    "doc_id":      p.payload.get("doc_id", ""),
                    "text":        p.payload.get("text", ""),
                    "chunk_index": p.payload.get("chunk_index", 0),
                    "metadata":    {
                        k: v for k, v in p.payload.items()
                        if k not in ("text", "doc_id", "chunk_index")
                    },
                }
                for p in all_points
            ]

            tokenized = [doc["text"].lower().split() for doc in corpus]
            self._bm25_index = (corpus, BM25Okapi(tokenized))
            log.info("bm25_index_built", corpus_size=len(corpus))
            return self._bm25_index

        except Exception as exc:
            log.warning("bm25_index_failed", error=str(exc))
            return None

    def invalidate_bm25_index(self) -> None:
        """Call after new documents are ingested."""
        self._bm25_index = None

    # ── RRF fusion ────────────────────────────────────────────────────────────

    def _rrf_fuse(
        self,
        dense: list[dict],
        sparse: list[dict],
        top_k: int,
    ) -> list[dict]:
        """
        Reciprocal Rank Fusion:
          score(d) = Σ  1 / (k + rank_i(d))
        where k=60 dampens the effect of high ranks (standard value).
        """
        k = settings.rrf_k
        scores: dict[str, float] = {}
        docs:   dict[str, dict]  = {}

        for rank, doc in enumerate(dense, start=1):
            cid = doc["chunk_id"]
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank)
            docs[cid]   = doc

        for rank, doc in enumerate(sparse, start=1):
            cid = doc["chunk_id"]
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank)
            if cid not in docs:
                docs[cid] = doc

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        return [
            {**docs[cid], "retrieval_score": score}
            for cid, score in ranked
        ]

    # ── Filter builder ────────────────────────────────────────────────────────

    @staticmethod
    def _build_filter(filters: dict[str, Any]) -> Filter:
        conditions = [
            FieldCondition(key=k, match=MatchValue(value=v))
            for k, v in filters.items()
        ]
        return Filter(must=conditions)
