from functools import lru_cache
from typing import Protocol

from app.core.config import get_settings


class Embedder(Protocol):
    """Turns text into fixed-length vectors."""

    @property
    def dimension(self) -> int:
        """Length of every vector this embedder returns."""
        ...

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed many chunks at once (used during ingestion)."""
        ...

    def embed_query(self, text: str) -> list[float]:
        """Embed one user question (used during retrieval)."""
        ...


class SentenceTransformerEmbedder:
    """Local embeddings via sentence-transformers (no API key, no cost)."""

    def __init__(self, model_name: str) -> None:
        # Imported here so merely importing this module (e.g. in tests or at
        # app startup) does not pay the cost of loading PyTorch.
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(model_name)
        dim = self._model.get_sentence_embedding_dimension()
        if dim is None:
            raise RuntimeError(f"Could not determine embedding size for {model_name}")
        self._dimension: int = dim

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        vectors: list[list[float]] = self._model.encode(
            texts,
            batch_size=32,
            normalize_embeddings=True,
            show_progress_bar=False,
        ).tolist()
        return vectors

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]


@lru_cache
def get_embedder() -> Embedder:
    return SentenceTransformerEmbedder(get_settings().embedding_model)
