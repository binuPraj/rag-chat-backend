from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient
from qdrant_client import QdrantClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from app.api.ingest import get_ingestion_service
from app.core.database import Base, get_db
from app.main import app
from app.models import document  # noqa: F401  (registers the table)
from app.services.embeddings import Embedder
from app.services.ingestion import IngestionService
from app.services.vector_store import VectorStore
from tests.fakes import FakeEmbedder, make_pdf

LONG_TEXT = " ".join(f"Sentence number {i} talks about vector databases." for i in range(60))


@pytest.fixture
def embedder() -> Embedder:
    return FakeEmbedder()


@pytest.fixture
def store(embedder: Embedder) -> VectorStore:
    vs = VectorStore(QdrantClient(":memory:"), "test_docs", embedder.dimension)
    vs.ensure_collection()
    return vs


@pytest.fixture
def client(embedder: Embedder, store: VectorStore) -> Iterator[TestClient]:
    # In-memory SQLite shared across threads, so tests never touch app.db.
    engine = create_engine(
        "sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool
    )
    Base.metadata.create_all(engine)
    session_factory = sessionmaker(bind=engine, expire_on_commit=False)

    def override_db() -> Iterator[Session]:
        db = session_factory()
        try:
            yield db
        finally:
            db.close()

    service = IngestionService(embedder, store)
    app.dependency_overrides[get_db] = override_db
    app.dependency_overrides[get_ingestion_service] = lambda: service
    # No `with` block: that would run the real startup and create app.db.
    yield TestClient(app)
    app.dependency_overrides.clear()


def test_ingest_txt_stores_chunks_and_metadata(
    client: TestClient, store: VectorStore, embedder: Embedder
) -> None:
    response = client.post(
        "/ingest",
        files={"file": ("facts.txt", LONG_TEXT.encode(), "text/plain")},
        data={"strategy": "sentence"},
    )
    assert response.status_code == 201
    body = response.json()
    assert body["filename"] == "facts.txt"
    assert body["strategy"] == "sentence"
    assert body["chunk_count"] > 1

    hits = store.search(embedder.embed_query("vector databases"), top_k=3, document_id=body["id"])
    assert hits
    assert hits[0].filename == "facts.txt"


def test_ingest_pdf(client: TestClient) -> None:
    response = client.post(
        "/ingest",
        files={"file": ("doc.pdf", make_pdf("Interview scheduling policy"), "application/pdf")},
    )
    assert response.status_code == 201
    assert response.json()["strategy"] == "fixed"  # the default


def test_strategies_produce_different_chunking(client: TestClient) -> None:
    def ingest(strategy: str) -> int:
        r = client.post(
            "/ingest",
            files={"file": ("t.txt", LONG_TEXT.encode(), "text/plain")},
            data={"strategy": strategy},
        )
        return int(r.json()["chunk_count"])

    assert ingest("fixed") != ingest("sentence")


def test_unsupported_file_type_returns_415(client: TestClient) -> None:
    response = client.post("/ingest", files={"file": ("pic.png", b"data", "image/png")})
    assert response.status_code == 415


def test_empty_file_returns_422(client: TestClient) -> None:
    response = client.post("/ingest", files={"file": ("empty.txt", b"  ", "text/plain")})
    assert response.status_code == 422


def test_invalid_strategy_returns_422(client: TestClient) -> None:
    response = client.post(
        "/ingest",
        files={"file": ("a.txt", b"hello there.", "text/plain")},
        data={"strategy": "nonsense"},
    )
    assert response.status_code == 422


def test_list_documents_returns_newest_first(client: TestClient) -> None:
    for name in ("first.txt", "second.txt"):
        client.post("/ingest", files={"file": (name, b"Some text here.", "text/plain")})
    response = client.get("/documents")
    assert response.status_code == 200
    assert [d["filename"] for d in response.json()] == ["second.txt", "first.txt"]
