"""
Chunking engine
Splits raw document text into overlapping, token-aware chunks.
Strategy is configurable per-request; defaults to 'recursive'.

recursive  — splits on paragraph → sentence → word boundaries
             (best for mixed documents; preserves semantic units)
semantic   — groups sentences by embedding similarity
             (slower, higher accuracy on long technical docs)
fixed      — fixed token windows with overlap
             (fastest, use for pre-structured data)
"""

import tiktoken
import structlog
from typing import Any
from shared.models import Chunk
from ingestion.config import settings

log = structlog.get_logger(__name__)

# Single tokenizer instance — thread-safe, reuse it
_TOKENIZER = tiktoken.get_encoding("cl100k_base")


def _token_count(text: str) -> int:
    return len(_TOKENIZER.encode(text, disallowed_special=()))


class ChunkingEngine:

    def __init__(
        self,
        chunk_size: int   = settings.chunk_size,
        chunk_overlap: int = settings.chunk_overlap,
        strategy: str      = settings.chunking_strategy,
    ):
        self.chunk_size    = chunk_size
        self.chunk_overlap = chunk_overlap
        self.strategy      = strategy

    def chunk(
        self,
        text: str,
        doc_id: str,
        metadata: dict[str, Any] | None = None,
    ) -> list[Chunk]:
        metadata = metadata or {}
        log.info(
            "chunking",
            doc_id=doc_id,
            strategy=self.strategy,
            input_tokens=_token_count(text),
        )

        raw_chunks = self._split(text)
        chunks = []
        for i, chunk_text in enumerate(raw_chunks):
            chunks.append(
                Chunk(
                    doc_id=doc_id,
                    text=chunk_text,
                    token_count=_token_count(chunk_text),
                    chunk_index=i,
                    metadata={
                        **metadata,
                        "chunk_strategy": self.strategy,
                        "chunk_size_cfg": self.chunk_size,
                    },
                )
            )

        log.info("chunking_done", doc_id=doc_id, n_chunks=len(chunks))
        return chunks

    # ── Strategies ────────────────────────────────────────────────────────────

    def _split(self, text: str) -> list[str]:
        if self.strategy == "recursive":
            return self._recursive_split(text)
        if self.strategy == "semantic":
            return self._semantic_split(text)
        return self._fixed_split(text)

    def _recursive_split(self, text: str) -> list[str]:
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        # We work in characters but gate on token count
        # ~4 chars/token is a safe approximation for English
        char_size    = self.chunk_size    * 4
        char_overlap = self.chunk_overlap * 4

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=char_size,
            chunk_overlap=char_overlap,
            length_function=_token_count,   # override with real tokenizer
            separators=["\n\n", "\n", ". ", " ", ""],
            keep_separator=True,
        )
        return [c for c in splitter.split_text(text) if c.strip()]

    def _fixed_split(self, text: str) -> list[str]:
        tokens = _TOKENIZER.encode(text, disallowed_special=())
        chunks = []
        start  = 0
        while start < len(tokens):
            end        = min(start + self.chunk_size, len(tokens))
            chunk_text = _TOKENIZER.decode(tokens[start:end])
            if chunk_text.strip():
                chunks.append(chunk_text)
            start += self.chunk_size - self.chunk_overlap
        return chunks

    def _semantic_split(self, text: str) -> list[str]:
        """
        Simple sentence-grouping semantic split.
        Groups consecutive sentences until token budget is exhausted,
        then starts a new chunk — with overlap carried from the tail of
        the previous group.
        """
        import re

        sentences = re.split(r"(?<=[.!?])\s+", text)
        chunks: list[str] = []
        current: list[str] = []
        current_tokens = 0

        for sent in sentences:
            sent_tokens = _token_count(sent)
            if current_tokens + sent_tokens > self.chunk_size and current:
                chunks.append(" ".join(current))
                # carry last N tokens worth of sentences as overlap
                overlap_sents: list[str] = []
                overlap_tokens = 0
                for s in reversed(current):
                    t = _token_count(s)
                    if overlap_tokens + t > self.chunk_overlap:
                        break
                    overlap_sents.insert(0, s)
                    overlap_tokens += t
                current        = overlap_sents
                current_tokens = overlap_tokens

            current.append(sent)
            current_tokens += sent_tokens

        if current:
            chunks.append(" ".join(current))

        return [c for c in chunks if c.strip()]
