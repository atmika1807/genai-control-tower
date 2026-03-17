"""
Tests for the ingestion pipeline.
Run with:  pytest tests/ -v
"""

import pytest
from ingestion.chunker import ChunkingEngine, _token_count


# ── Chunker tests ─────────────────────────────────────────────────────────────

SAMPLE_TEXT = """
Large language models (LLMs) are neural networks trained on massive text corpora.
They learn statistical patterns that allow them to generate coherent text.

The transformer architecture, introduced in "Attention Is All You Need" (2017),
underpins virtually all modern LLMs. It uses self-attention to relate tokens across
the full input sequence — enabling the model to capture long-range dependencies.

Retrieval-Augmented Generation (RAG) grounds LLM responses in external knowledge.
A retriever fetches relevant document chunks; the LLM synthesises an answer
using only those chunks. This reduces hallucinations and keeps answers current.
""".strip()


class TestChunkingEngine:

    def test_recursive_produces_chunks(self):
        engine = ChunkingEngine(chunk_size=128, chunk_overlap=16, strategy="recursive")
        chunks = engine.chunk(SAMPLE_TEXT, doc_id="test-doc-1")
        assert len(chunks) >= 1
        for c in chunks:
            assert c.text.strip()
            assert c.token_count > 0
            assert c.doc_id == "test-doc-1"

    def test_fixed_respects_chunk_size(self):
        engine = ChunkingEngine(chunk_size=64, chunk_overlap=8, strategy="fixed")
        chunks = engine.chunk(SAMPLE_TEXT, doc_id="test-doc-2")
        for c in chunks:
            # Allow slight overrun from tokeniser rounding
            assert c.token_count <= 80

    def test_semantic_produces_chunks(self):
        engine = ChunkingEngine(chunk_size=128, chunk_overlap=16, strategy="semantic")
        chunks = engine.chunk(SAMPLE_TEXT, doc_id="test-doc-3")
        assert len(chunks) >= 1

    def test_chunk_indices_are_sequential(self):
        engine = ChunkingEngine(chunk_size=128, chunk_overlap=16)
        chunks = engine.chunk(SAMPLE_TEXT, doc_id="test-doc-4")
        indices = [c.chunk_index for c in chunks]
        assert indices == list(range(len(chunks)))

    def test_metadata_propagated(self):
        engine = ChunkingEngine()
        meta = {"source": "unit-test", "author": "pytest"}
        chunks = engine.chunk(SAMPLE_TEXT, doc_id="test-doc-5", metadata=meta)
        for c in chunks:
            assert c.metadata["source"] == "unit-test"
            assert c.metadata["author"] == "pytest"

    def test_empty_text_returns_no_chunks(self):
        engine = ChunkingEngine()
        chunks = engine.chunk("   \n\n   ", doc_id="test-doc-empty")
        assert chunks == []

    def test_token_count_helper(self):
        count = _token_count("Hello, world!")
        assert isinstance(count, int)
        assert count > 0


# ── Model schema tests ────────────────────────────────────────────────────────

class TestModels:

    def test_chunk_id_auto_generated(self):
        from shared.models import Chunk
        c1 = Chunk(doc_id="d1", text="hello", token_count=1, chunk_index=0)
        c2 = Chunk(doc_id="d1", text="hello", token_count=1, chunk_index=1)
        assert c1.chunk_id != c2.chunk_id

    def test_ingest_request_validation(self):
        from shared.models import IngestRequest, SourceType
        req = IngestRequest(source_type=SourceType.PDF, source_uri="/tmp/test.pdf")
        assert req.priority == 5

    def test_ingest_request_bad_priority(self):
        from pydantic import ValidationError
        from shared.models import IngestRequest, SourceType
        with pytest.raises(ValidationError):
            IngestRequest(source_type=SourceType.PDF, source_uri="/tmp/x.pdf", priority=11)
