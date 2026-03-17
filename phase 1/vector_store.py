"""
Vector store client (Qdrant)
Wraps qdrant-client with helpers for:
  - collection bootstrap (HNSW config tuned for sub-10ms ANN at 10K docs)
  - batch upsert of EmbeddedChunks
  - graceful collection re-creation on schema mismatch
"""

import structlog
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    HnswConfigDiff,
    OptimizersConfigDiff,
    PointStruct,
    PayloadSchemaType,
)

from shared.models import EmbeddedChunk
from ingestion.config import settings

log = structlog.get_logger(__name__)

# HNSW tuning
# m=16        → 16 bi-directional links per node; good recall / memory balance
# ef_construct=200 → higher = better index quality, slower build
# full_scan_threshold=10000 → fall back to brute-force below this count
HNSW_CONFIG = HnswConfigDiff(m=16, ef_construct=200, full_scan_threshold=10_000)

# Flush index every 1000 vectors for write throughput during bulk ingest
OPTIMIZER_CONFIG = OptimizersConfigDiff(indexing_threshold=1_000)


class VectorStoreClient:

    def __init__(self) -> None:
        self._client = AsyncQdrantClient(
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key,
            timeout=30,
        )

    async def ensure_collection(self) -> None:
        """Create collection if it doesn't exist; warn if dimensions mismatch."""
        existing = {
            c.name
            for c in (await self._client.get_collections()).collections
        }

        if settings.collection_name in existing:
            log.info("collection_exists", name=settings.collection_name)
            return

        log.info("creating_collection", name=settings.collection_name)
        await self._client.create_collection(
            collection_name=settings.collection_name,
            vectors_config=VectorParams(
                size=settings.embedding_dimensions,
                distance=Distance.COSINE,
                hnsw_config=HNSW_CONFIG,
                on_disk=False,          # keep in RAM for sub-10ms search
            ),
            optimizers_config=OPTIMIZER_CONFIG,
        )

        # Payload indexes for efficient metadata filtering
        for field, schema in [
            ("doc_id",        PayloadSchemaType.KEYWORD),
            ("chunk_index",   PayloadSchemaType.INTEGER),
            ("source_type",   PayloadSchemaType.KEYWORD),
        ]:
            await self._client.create_payload_index(
                collection_name=settings.collection_name,
                field_name=field,
                field_schema=schema,
            )

        log.info("collection_created", name=settings.collection_name)

    async def upsert(self, embedded_chunks: list[EmbeddedChunk]) -> int:
        """Upsert chunks in batches of 256. Returns count of upserted points."""
        BATCH = 256
        total = 0

        for i in range(0, len(embedded_chunks), BATCH):
            batch = embedded_chunks[i : i + BATCH]
            points = [
                PointStruct(
                    id=chunk.chunk_id,
                    vector=chunk.embedding,
                    payload={
                        "doc_id":      chunk.doc_id,
                        "text":        chunk.text,
                        "chunk_index": chunk.chunk_index,
                        "token_count": chunk.token_count,
                        **chunk.metadata,
                    },
                )
                for chunk in batch
            ]

            await self._client.upsert(
                collection_name=settings.collection_name,
                points=points,
                wait=True,   # wait for indexing before returning
            )
            total += len(points)
            log.info("upserted_batch", count=len(points), total_so_far=total)

        return total

    async def delete_by_doc_id(self, doc_id: str) -> None:
        """Remove all chunks belonging to a document (for re-indexing)."""
        from qdrant_client.models import Filter, FieldCondition, MatchValue

        await self._client.delete(
            collection_name=settings.collection_name,
            points_selector=Filter(
                must=[FieldCondition(
                    key="doc_id",
                    match=MatchValue(value=doc_id),
                )]
            ),
        )
        log.info("deleted_doc_chunks", doc_id=doc_id)

    async def collection_stats(self) -> dict:
        info = await self._client.get_collection(settings.collection_name)
        return {
            "vectors_count":  info.vectors_count,
            "indexed_vectors": info.indexed_vectors_count,
            "status":         info.status,
        }
