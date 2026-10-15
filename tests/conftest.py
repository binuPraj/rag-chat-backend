"""Shared fixtures: in-memory Qdrant, SQLite and Redis so tests need no servers."""

from collections.abc import Iterator
from datetime import date

import fakeredis
import pytest
from qdrant_client import QdrantClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from app.core.database import Base
from app.models import booking, document  # noqa: F401  (registers the tables)
from app.services.booking import BookingService
from app.services.chat import ChatService
from app.services.embeddings import Embedder
from app.services.intent import IntentAnalyzer
from app.services.memory import RedisChatMemory
from app.services.rag import RagService
from app.services.vector_store import VectorStore
from tests.fakes import FakeEmbedder, FakeLLM

TODAY = date(2026, 10, 14)


@pytest.fixture
def embedder() -> Embedder:
    return FakeEmbedder()


@pytest.fixture
def store(embedder: Embedder) -> VectorStore:
    vs = VectorStore(QdrantClient(":memory:"), "test_docs", embedder.dimension)
    vs.ensure_collection()
    return vs


@pytest.fixture
def session_factory() -> sessionmaker[Session]:
    engine = create_engine(
        "sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool
    )
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine, expire_on_commit=False)


@pytest.fixture
def db(session_factory: sessionmaker[Session]) -> Iterator[Session]:
    session = session_factory()
    yield session
    session.close()


@pytest.fixture
def memory() -> RedisChatMemory:
    return RedisChatMemory(fakeredis.FakeRedis(decode_responses=True), ttl_seconds=60, max_messages=20)


def make_chat_service(llm: FakeLLM, memory: RedisChatMemory, store: VectorStore, embedder: Embedder) -> ChatService:
    return ChatService(
        memory=memory,
        analyzer=IntentAnalyzer(llm),
        rag=RagService(embedder, store, llm, top_k=3),
        bookings=BookingService(),
        prompt_history=6,
    )
