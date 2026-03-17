"""
Embedding service
Converts chunks → float vectors using OpenAI text-embedding-3-large.

Key design decisions:
  - Batches up to `embedding_batch_size` chunks per API call
  - Exponential backoff via tenacity on rate-limit errors
  - Token budget guard: skips chunks that exceed the model's 8192-token limit
  - Returns EmbeddedChunk list in the same order as input
"""

import asyncio
import structlog
from openai import AsyncOpenAI
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)
from openai import RateLimitError, APIStatusError

from shared.models import Chunk, EmbeddedChunk
from ingestion.config import settings

log = structlog.get_logger(__name__)

MAX_TOKENS_PER_CHUNK = 8192   # OpenAI embedding model hard limit


class EmbeddingService:

    def __init__(self) -> None:
        self._client = AsyncOpenAI(api_key=settings.openai_api_key)
        self._semaphore = asyncio.Semaphore(
            # crude RPM guard: allow N concurrent batch calls
            max(1, settings.embedding_requests_per_minute // 60)
        )

    async def embed_chunks(self, chunks: list[Chunk]) -> list[EmbeddedChunk]:
        """Embed all chunks, returning EmbeddedChunk in input order."""
        # Filter out anything that would exceed the model's token limit
        safe, skipped = [], []
        for c in chunks:
            if c.token_count <= MAX_TOKENS_PER_CHUNK:
                safe.append(c)
            else:
                skipped.append(c.chunk_id)

        if skipped:
            log.warning("chunks_skipped_token_limit", chunk_ids=skipped)

        # Split into batches
        batches = [
            safe[i : i + settings.embedding_batch_size]
            for i in range(0, len(safe), settings.embedding_batch_size)
        ]

        log.info(
            "embedding_start",
            total_chunks=len(safe),
            n_batches=len(batches),
        )

        results: list[EmbeddedChunk] = []
        for batch in batches:
            embedded = await self._embed_batch(batch)
            results.extend(embedded)

        return results

    @retry(
        retry=retry_if_exception_type((RateLimitError, APIStatusError)),
        wait=wait_exponential(multiplier=1, min=2, max=60),
        stop=stop_after_attempt(5),
    )
    async def _embed_batch(self, chunks: list[Chunk]) -> list[EmbeddedChunk]:
        texts = [c.text for c in chunks]

        async with self._semaphore:
            response = await self._client.embeddings.create(
                model=settings.embedding_model,
                input=texts,
                dimensions=settings.embedding_dimensions,
                encoding_format="float",
            )

        log.info(
            "embedding_batch_done",
            n=len(chunks),
            total_tokens=response.usage.total_tokens,
        )

        embedded = []
        for chunk, emb_obj in zip(chunks, response.data):
            embedded.append(
                EmbeddedChunk(
                    **chunk.model_dump(),
                    embedding=emb_obj.embedding,
                )
            )
        return embedded
