import uuid
from dataclasses import dataclass
from functools import lru_cache

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    VectorParams,
)

from app.core.config import get_settings


@dataclass(frozen=True)
class SearchHit:
    text: str
    score: float
    document_id: str
    filename: str
    chunk_index: int


class VectorStore:
    def __init__(self, client: QdrantClient, collection: str, dimension: int) -> None:
        self._client = client
        self._collection = collection
        self._dimension = dimension

    def ensure_collection(self) -> None:
        """Create the collection if it does not exist yet (safe to call often)."""
        if not self._client.collection_exists(self._collection):
            self._client.create_collection(
                collection_name=self._collection,
                vectors_config=VectorParams(size=self._dimension, distance=Distance.COSINE),
            )

    def upsert_chunks(
        self,
        *,
        document_id: str,
        filename: str,
        strategy: str,
        chunks: list[str],
        vectors: list[list[float]],
    ) -> int:
        """Store chunks with their vectors. Returns how many were stored."""
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors must have the same length")
        if not chunks:
            return 0

        points = [
            PointStruct(
                # Deterministic ID: re-ingesting the same document overwrites
                # its chunks instead of creating duplicates.
                id=str(uuid.uuid5(uuid.NAMESPACE_URL, f"{document_id}:{index}")),
                vector=vector,
                payload={
                    "document_id": document_id,
                    "filename": filename,
                    "strategy": strategy,
                    "chunk_index": index,
                    "text": text,
                },
            )
            for index, (text, vector) in enumerate(zip(chunks, vectors, strict=True))
        ]
        self._client.upsert(collection_name=self._collection, points=points)
        return len(points)

    def search(
        self,
        vector: list[float],
        top_k: int = 4,
        document_id: str | None = None,
    ) -> list[SearchHit]:
        query_filter = (
            Filter(must=[FieldCondition(key="document_id", match=MatchValue(value=document_id))])
            if document_id is not None
            else None
        )
        response = self._client.query_points(
            collection_name=self._collection,
            query=vector,
            limit=top_k,
            query_filter=query_filter,
            with_payload=True,
        )
        hits: list[SearchHit] = []
        for point in response.points:
            payload = point.payload or {}
            hits.append(
                SearchHit(
                    text=str(payload.get("text", "")),
                    score=point.score,
                    document_id=str(payload.get("document_id", "")),
                    filename=str(payload.get("filename", "")),
                    chunk_index=int(payload.get("chunk_index", 0)),
                )
            )
        return hits

    def delete_document(self, document_id: str) -> None:
        self._client.delete(
            collection_name=self._collection,
            points_selector=Filter(
                must=[FieldCondition(key="document_id", match=MatchValue(value=document_id))]
            ),
        )


@lru_cache
def get_vector_store() -> VectorStore:
    settings = get_settings()
    client = QdrantClient(host=settings.qdrant_host, port=settings.qdrant_port)
    store = VectorStore(client, settings.qdrant_collection, settings.embedding_dim)
    store.ensure_collection()
    return store
