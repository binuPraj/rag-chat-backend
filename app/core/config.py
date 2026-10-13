from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    gemini_api_key: str = ""

    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_collection: str = "documents"

    redis_url: str = "redis://localhost:6379/0"
    chat_ttl_seconds: int = 3600

    database_url: str = "sqlite:///./app.db"

    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dim: int = 384


@lru_cache
def get_settings() -> Settings:
    """Return a cached Settings instance (one read of the environment)."""
    return Settings()