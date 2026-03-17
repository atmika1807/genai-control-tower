from pydantic_settings import BaseSettings
from pydantic import Field
from typing import Literal


class Settings(BaseSettings):
    # API
    app_name: str = "GenAI Control Tower — Ingestion Service"
    api_version: str = "v1"
    debug: bool = False

    # OpenAI / Embeddings
    openai_api_key: str = Field(..., env="OPENAI_API_KEY")
    embedding_model: str = "text-embedding-3-large"
    embedding_dimensions: int = 3072
    embedding_batch_size: int = 64          # docs per OpenAI batch call

    # Chunking
    chunk_size: int = 512                   # tokens per chunk
    chunk_overlap: int = 64                 # ~12% overlap
    chunking_strategy: Literal[
        "recursive", "semantic", "fixed"
    ] = "recursive"

    # Vector store
    vector_store: Literal["qdrant", "weaviate"] = "qdrant"
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str | None = None
    collection_name: str = "enterprise_docs"

    # Redis / Celery
    redis_url: str = "redis://localhost:6379/0"
    celery_concurrency: int = 4

    # MLflow
    mlflow_tracking_uri: str = "http://localhost:5000"
    mlflow_experiment_name: str = "ingestion-pipeline"

    # Rate limiting (OpenAI RPM guard)
    embedding_requests_per_minute: int = 500

    class Config:
        env_file = ".env"
        case_sensitive = False


settings = Settings()
