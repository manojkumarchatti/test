import logging
import os
from functools import lru_cache
from pathlib import Path
from typing import Optional

from pydantic import BaseSettings, Field


class Settings(BaseSettings):
    app_name: str = "company-knowledge-rag"
    environment: str = Field("local", env="APP_ENV")
    api_prefix: str = "/api"

    storage_path: Path = Field(Path(os.getenv("STORAGE_PATH", "backend/data")))
    vector_store_path: Path = Field(Path(os.getenv("VECTOR_STORE_PATH", "backend/data/vector_store")))
    uploads_path: Path = Field(Path(os.getenv("UPLOADS_PATH", "backend/data/uploads")))

    database_url: str = Field(os.getenv("DATABASE_URL", "sqlite:///backend/data/metadata.db"))

    openai_api_key: Optional[str] = Field(default=None, env="OPENAI_API_KEY")
    reranker_model: str = Field("cross-encoder/ms-marco-MiniLM-L-6-v2", env="RERANKER_MODEL")

    max_requests_per_minute: int = Field(60, env="RATE_LIMIT_RPM")
    cache_ttl_seconds: int = Field(300, env="CACHE_TTL_SECONDS")
    enable_langsmith: bool = Field(False, env="LANGSMITH_ENABLED")

    class Config:
        case_sensitive = False
        env_file = ".env"


@lru_cache()
def get_settings() -> Settings:
    settings = Settings()
    settings.storage_path.mkdir(parents=True, exist_ok=True)
    settings.vector_store_path.mkdir(parents=True, exist_ok=True)
    settings.uploads_path.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    return settings
