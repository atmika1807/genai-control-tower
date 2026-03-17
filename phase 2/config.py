from pydantic_settings import BaseSettings
from pydantic import Field


class QuerySettings(BaseSettings):
    app_name: str = "GenAI Control Tower — Query Service"
    api_version: str = "v1"
    debug: bool = False

    # OpenAI
    openai_api_key: str = Field(..., env="OPENAI_API_KEY")
    embedding_model: str = "text-embedding-3-large"
    embedding_dimensions: int = 3072
    llm_model: str = "gpt-4o"
    llm_max_tokens: int = 1024
    llm_temperature: float = 0.0          # deterministic for grounded RAG

    # Retrieval
    retrieval_top_k: int = 20             # candidates from vector + BM25
    rerank_top_k: int = 5                 # kept after cross-encoder
    rrf_k: int = 60                       # RRF constant (standard = 60)

    # Cross-encoder re-ranker
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    reranker_device: str = "cpu"          # "cuda" if GPU available

    # Vector store
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str | None = None
    collection_name: str = "enterprise_docs"

    # MLflow
    mlflow_tracking_uri: str = "http://localhost:5000"
    mlflow_experiment_name: str = "query-service"

    # Latency SLO
    latency_slo_ms: float = 200.0

    class Config:
        env_file = ".env"
        case_sensitive = False


settings = QuerySettings()
