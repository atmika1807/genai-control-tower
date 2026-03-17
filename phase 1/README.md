# Enterprise GenAI Control Tower — Ingestion Pipeline

Async document ingestion service: load → chunk → embed → vector index.

## Architecture

```
POST /v1/ingest
      │
      ▼
  FastAPI (ingestion/main.py)
      │  queues job
      ▼
  Celery worker (ingestion/tasks.py)
      │
      ├─► DocumentLoader  (PDF / DOCX / TXT / S3 / URL)
      ├─► ChunkingEngine  (recursive / semantic / fixed)
      ├─► EmbeddingService (OpenAI text-embedding-3-large, batched)
      ├─► VectorStoreClient (Qdrant, HNSW)
      └─► MLflow (cost, latency, chunk metrics)
```

## Quickstart

### 1. Environment

```bash
cp .env.example .env
# Set OPENAI_API_KEY in .env
```

### 2. Start all services

```bash
docker compose up -d
```

| Service         | URL                        |
|-----------------|----------------------------|
| Ingestion API   | http://localhost:8000/docs |
| Celery Flower   | http://localhost:5555      |
| Qdrant UI       | http://localhost:6333/dashboard |
| MLflow          | http://localhost:5000      |

### 3. Ingest a document

```bash
# Single document
curl -X POST http://localhost:8000/v1/ingest \
  -H "Content-Type: application/json" \
  -d '{
    "source_type": "pdf",
    "source_uri": "/data/annual-report.pdf",
    "metadata": {"department": "finance", "year": 2024}
  }'

# Response → { "job_id": "abc-123", "status": "pending" }

# Poll status
curl http://localhost:8000/v1/jobs/abc-123
```

### 4. Upload a file directly

```bash
curl -X POST http://localhost:8000/v1/ingest/upload \
  -F "file=@/path/to/document.pdf"
```

### 5. Batch ingest

```bash
curl -X POST http://localhost:8000/v1/ingest/batch \
  -H "Content-Type: application/json" \
  -d '{
    "documents": [
      {"source_type": "pdf", "source_uri": "/data/doc1.pdf"},
      {"source_type": "url", "source_uri": "https://example.com/doc.pdf"},
      {"source_type": "s3",  "source_uri": "s3://my-bucket/reports/q3.pdf"}
    ]
  }'
```

## Configuration

Key settings in `ingestion/config.py` (all overridable via env vars):

| Variable              | Default                    | Notes                     |
|-----------------------|----------------------------|---------------------------|
| `CHUNK_SIZE`          | 512 tokens                 | Tune for your docs        |
| `CHUNK_OVERLAP`       | 64 tokens                  | ~12% overlap              |
| `CHUNKING_STRATEGY`   | recursive                  | recursive / semantic / fixed |
| `EMBEDDING_MODEL`     | text-embedding-3-large     |                           |
| `EMBEDDING_BATCH_SIZE`| 64                         | Chunks per OpenAI call    |
| `CELERY_CONCURRENCY`  | 4                          | Workers per replica       |

## Development

```bash
# Install deps
pip install -r requirements.txt

# Run tests
pytest tests/ -v

# Run API locally (no Docker)
uvicorn ingestion.main:app --reload --port 8000

# Run worker locally
celery -A ingestion.tasks.celery_app worker --loglevel=info
```

## Next: Query Service

The query service reads from the same Qdrant collection.
See `query/` for the FastAPI query service with hybrid retrieval and
cross-encoder re-ranking.
