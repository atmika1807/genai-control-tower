"""
Tests for the query service.
Run with:  pytest tests/test_query.py -v

Most tests mock the external services (OpenAI, Qdrant) so they run
without any infrastructure dependencies.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch


# ── RRF fusion ────────────────────────────────────────────────────────────────

class TestRRFFusion:

    def _make_retriever(self):
        from query.retriever import HybridRetriever
        with patch("query.retriever.AsyncQdrantClient"), \
             patch("query.retriever.AsyncOpenAI"):
            return HybridRetriever()

    def test_rrf_scores_overlap_higher(self):
        r = self._make_retriever()
        dense  = [{"chunk_id": "a", "text": "x", "retrieval_score": 0.9, "doc_id": "d1", "chunk_index": 0, "metadata": {}}]
        sparse = [{"chunk_id": "a", "text": "x", "retrieval_score": 1.5, "doc_id": "d1", "chunk_index": 0, "metadata": {}}]
        fused = r._rrf_fuse(dense, sparse, top_k=5)
        # chunk "a" appears in both lists — should score higher than a single-list doc
        assert fused[0]["chunk_id"] == "a"
        # score should be 2 * 1/(60+1) ≈ 0.0328
        assert fused[0]["retrieval_score"] > 1 / (60 + 1)

    def test_rrf_no_duplicates(self):
        r = self._make_retriever()
        dense  = [
            {"chunk_id": "a", "text": "x", "retrieval_score": 0.9, "doc_id": "d1", "chunk_index": 0, "metadata": {}},
            {"chunk_id": "b", "text": "y", "retrieval_score": 0.8, "doc_id": "d2", "chunk_index": 0, "metadata": {}},
        ]
        sparse = [
            {"chunk_id": "b", "text": "y", "retrieval_score": 1.0, "doc_id": "d2", "chunk_index": 0, "metadata": {}},
            {"chunk_id": "c", "text": "z", "retrieval_score": 0.7, "doc_id": "d3", "chunk_index": 0, "metadata": {}},
        ]
        fused = r._rrf_fuse(dense, sparse, top_k=10)
        ids = [f["chunk_id"] for f in fused]
        assert len(ids) == len(set(ids)), "No duplicates in fused output"

    def test_rrf_respects_top_k(self):
        r = self._make_retriever()
        dense  = [{"chunk_id": str(i), "text": "x", "retrieval_score": 1.0, "doc_id": "d", "chunk_index": i, "metadata": {}} for i in range(20)]
        sparse = []
        fused = r._rrf_fuse(dense, sparse, top_k=5)
        assert len(fused) <= 5

    def test_rrf_empty_inputs(self):
        r = self._make_retriever()
        assert r._rrf_fuse([], [], top_k=5) == []


# ── Re-ranker ────────────────────────────────────────────────────────────────

class TestReranker:

    def test_rerank_returns_top_k(self):
        from query.reranker import CrossEncoderReranker
        mock_model = MagicMock()
        mock_model.predict.return_value = [0.9, 0.3, 0.7, 0.1, 0.5]

        reranker = CrossEncoderReranker.__new__(CrossEncoderReranker)
        reranker._model = mock_model

        candidates = [
            {"chunk_id": str(i), "text": f"chunk {i}", "retrieval_score": 0.5, "doc_id": "d", "chunk_index": i, "metadata": {}}
            for i in range(5)
        ]
        result = reranker.rerank("test query", candidates, top_k=3)
        assert len(result) == 3
        # Should be sorted by score descending
        assert result[0]["score"] >= result[1]["score"] >= result[2]["score"]

    def test_rerank_empty_returns_empty(self):
        from query.reranker import CrossEncoderReranker
        reranker = CrossEncoderReranker.__new__(CrossEncoderReranker)
        reranker._model = MagicMock()
        assert reranker.rerank("query", [], top_k=5) == []

    def test_rerank_preserves_retrieval_score(self):
        from query.reranker import CrossEncoderReranker
        mock_model = MagicMock()
        mock_model.predict.return_value = [0.8]

        reranker = CrossEncoderReranker.__new__(CrossEncoderReranker)
        reranker._model = mock_model

        candidates = [{"chunk_id": "x", "text": "hello", "retrieval_score": 0.42, "doc_id": "d", "chunk_index": 0, "metadata": {}}]
        result = reranker.rerank("query", candidates, top_k=1)
        assert result[0]["retrieval_score"] == 0.42


# ── Grounding prompt ──────────────────────────────────────────────────────────

class TestSynthesizerPrompt:

    def test_context_numbered(self):
        from query.synthesizer import _build_context
        chunks = [
            {"text": "Transformers use attention.", "metadata": {"source_uri": "doc1.pdf"}},
            {"text": "BERT is bidirectional.",      "metadata": {"source_uri": "doc2.pdf"}},
        ]
        ctx = _build_context(chunks)
        assert "[1]" in ctx
        assert "[2]" in ctx
        assert "Transformers use attention." in ctx

    def test_empty_context(self):
        from query.synthesizer import _build_context
        assert _build_context([]) == ""


# ── Observer ──────────────────────────────────────────────────────────────────

class TestQueryObserver:

    @pytest.mark.asyncio
    async def test_trace_computes_total_latency(self):
        import asyncio
        from query.observer import QueryObserver

        obs = QueryObserver.__new__(QueryObserver)

        with patch("query.observer.mlflow"):
            async with obs.trace("test query") as trace:
                trace.prompt_tokens     = 100
                trace.completion_tokens = 50
                await asyncio.sleep(0.01)  # simulate 10ms work

        assert trace.t_total_ms >= 10
        assert trace.total_tokens == 0   # not set → default
        assert trace.cost_usd >= 0

    @pytest.mark.asyncio
    async def test_slo_met_flag(self):
        from query.observer import QueryObserver
        obs = QueryObserver.__new__(QueryObserver)

        with patch("query.observer.mlflow"):
            async with obs.trace("q") as trace:
                pass  # near-instant

        assert trace.slo_met is True
