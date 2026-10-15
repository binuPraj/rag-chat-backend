import pytest
from qdrant_client import QdrantClient

from app.services.embeddings import Embedder
from app.services.vector_store import VectorStore
from tests.fakes import FakeEmbedder

CHUNKS = [
    "Python is a programming language used for data science.",
    "Qdrant is a vector database for similarity search.",
    "Bananas are yellow fruit rich in potassium.",
]


@pytest.fixture
def embedder() -> Embedder:
    return FakeEmbedder()


@pytest.fixture
def store(embedder: Embedder) -> VectorStore:
    vs = VectorStore(QdrantClient(":memory:"), "test_docs", embedder.dimension)
    vs.ensure_collection()
    return vs


def _ingest(store: VectorStore, embedder: Embedder, document_id: str = "doc-1") -> None:
    store.upsert_chunks(
        document_id=document_id,
        filename=f"{document_id}.txt",
        strategy="fixed",
        chunks=CHUNKS,
        vectors=embedder.embed_documents(CHUNKS),
    )


def test_ensure_collection_is_idempotent(store: VectorStore) -> None:
    store.ensure_collection()
    store.ensure_collection()


def test_search_returns_most_similar_chunk_first(store: VectorStore, embedder: Embedder) -> None:
    _ingest(store, embedder)
    hits = store.search(embedder.embed_query("what is a vector database?"), top_k=2)
    assert len(hits) == 2
    assert "Qdrant" in hits[0].text
    assert hits[0].score >= hits[1].score
    assert hits[0].filename == "doc-1.txt"


def test_search_can_be_limited_to_one_document(store: VectorStore, embedder: Embedder) -> None:
    _ingest(store, embedder, "doc-1")
    _ingest(store, embedder, "doc-2")
    hits = store.search(embedder.embed_query("vector database"), top_k=10, document_id="doc-2")
    assert hits
    assert {h.document_id for h in hits} == {"doc-2"}


def test_reingesting_same_document_does_not_duplicate(
    store: VectorStore, embedder: Embedder
) -> None:
    _ingest(store, embedder)
    _ingest(store, embedder)
    hits = store.search(embedder.embed_query("anything"), top_k=10)
    assert len(hits) == len(CHUNKS)


def test_delete_document_removes_its_chunks(store: VectorStore, embedder: Embedder) -> None:
    _ingest(store, embedder, "doc-1")
    store.delete_document("doc-1")
    assert store.search(embedder.embed_query("vector database")) == []


def test_mismatched_chunks_and_vectors_raise(store: VectorStore) -> None:
    with pytest.raises(ValueError):
        store.upsert_chunks(
            document_id="d", filename="d.txt", strategy="fixed", chunks=["a", "b"], vectors=[[0.0]]
        )
