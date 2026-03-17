"""
Observability dashboard API
Serves aggregated metrics from MLflow to the Control Tower UI.

Endpoints:
  GET /metrics/ingestion   — ingestion pipeline stats
  GET /metrics/query       — query service stats
  GET /metrics/evals       — latest retrieval eval results
  GET /metrics/summary     — combined one-page summary
"""

from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from observability.collector import MLflowCollector
import os

MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")

app = FastAPI(title="GenAI Control Tower — Observability API", version="v1")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


def _collector(window_days: int = 30) -> MLflowCollector:
    return MLflowCollector(MLFLOW_URI, window_days=window_days)


@app.get("/metrics/ingestion")
async def ingestion_metrics(window_days: int = Query(default=30, ge=1, le=90)):
    return _collector(window_days).get_ingestion_stats().__dict__


@app.get("/metrics/query")
async def query_metrics(window_days: int = Query(default=30, ge=1, le=90)):
    return _collector(window_days).get_query_stats().__dict__


@app.get("/metrics/summary")
async def summary(window_days: int = Query(default=30, ge=1, le=90)):
    c = _collector(window_days)
    return {
        "ingestion": c.get_ingestion_stats().__dict__,
        "query":     c.get_query_stats().__dict__,
        "window_days": window_days,
    }


@app.get("/health")
async def health():
    return {"status": "ok"}
