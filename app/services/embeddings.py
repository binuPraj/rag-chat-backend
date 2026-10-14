from functools import lru_cache
from typing import Protocol

from app.core.config import get_settings


class Embedder(Protocol):

    @property
    def dimension(self) -> int: ...

    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    def embed_query(self, text:str) -> list[float]: ...

class SentenceTrandsformerEmbedder:

    def __init__(self, model_name:str) -> None:

        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(model_name)
        dim=self.model.get_sentence_embedding_dimension()

        if dim is None:
            raise RuntimeError(f"Failed to get embedding dimension for model {model_name}")
        self._dimension: int=dim

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        vectors: list[list[float]] =self._model.encode(
            texts,
            batch_size=32,
            normalize_embeddings=True,
            show_progress_bar=False,
        ).tolist()
        return vectors

    def embed_query(self, text:str) -> list[float]:
        return self.embed_documents([text])[0]

@lru_cache
def get_embedder() -> Embedder:
    return SentenceTrandsformerEmbedder(get_settings().embedding_model)



