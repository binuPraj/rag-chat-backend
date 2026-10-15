"""Central application settings, loaded from environment variables / .env."""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    gemini_api_key: str = ""
    llm_model: str = "gemini-3.5-flash-lite"

    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_collection: str = "documents"

    redis_url: str = "redis://localhost:6379/0"
    chat_ttl_seconds: int = 3600
    max_history_messages: int = 20  
    prompt_history_messages: int = 6  
    retrieval_top_k: int = 4

    database_url: str = "sqlite:///./app.db"

    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dim: int = 384


@lru_cache
def get_settings() -> Settings:
    """Return a cached Settings instance (one read of the environment)."""
    return Settings()
