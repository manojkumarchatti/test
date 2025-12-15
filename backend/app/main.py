import logging
import time
from typing import List

from fastapi import Depends, FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware

from app.core.config import get_settings
from app.models.schemas import AnswerResponse, IngestResponse, QuestionRequest
from app.services.ingestion import IngestionService
from app.services.retrieval import retrieval_service

logger = logging.getLogger(__name__)
settings = get_settings()


def rate_limiter():
    # simple in-memory rate limiter keyed by minute bucket
    bucket = int(time.time() // 60)
    if not hasattr(rate_limiter, "counts"):
        rate_limiter.counts = {}
    counts = rate_limiter.counts
    counts.setdefault(bucket, 0)
    counts[bucket] += 1
    if counts[bucket] > settings.max_requests_per_minute:
        raise HTTPException(status_code=429, detail="Rate limit exceeded")


class LoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        start = time.time()
        response = await call_next(request)
        duration = (time.time() - start) * 1000
        logger.info("%s %s completed in %.2fms", request.method, request.url.path, duration)
        return response


def create_app() -> FastAPI:
    app = FastAPI(title=settings.app_name, version="0.1.0")
    app.add_middleware(LoggingMiddleware)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    ingestion_service = IngestionService()

    @app.get("/health")
    async def health() -> dict:
        return {"status": "ok", "environment": settings.environment}

    @app.post(f"{settings.api_prefix}/ingest", response_model=IngestResponse)
    async def ingest(file: UploadFile = File(...), _: None = Depends(rate_limiter)):
        content = await file.read()
        summary = ingestion_service.load_document(file.filename, content)
        count = ingestion_service.persist_embeddings(summary.chunks)
        return IngestResponse(document_id=summary.document_id, chunks=count)

    @app.post(f"{settings.api_prefix}/query", response_model=AnswerResponse)
    async def query(request: QuestionRequest, _: None = Depends(rate_limiter)):
        return retrieval_service.answer(request)

    return app


app = create_app()
