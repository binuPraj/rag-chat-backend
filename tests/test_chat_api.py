from collections.abc import Callable, Iterator

import pytest
from fastapi.testclient import TestClient
from redis.exceptions import ConnectionError as RedisConnectionError
from sqlalchemy.orm import Session, sessionmaker

from app.api.chat import get_chat_service
from app.core.database import get_db
from app.main import app
from app.services.embeddings import Embedder
from app.services.llm import LLMError
from app.services.memory import ChatMessage, RedisChatMemory, get_chat_memory
from app.services.vector_store import VectorStore
from tests.conftest import make_chat_service
from tests.fakes import FakeLLM, analysis_json

MakeClient = Callable[[FakeLLM], TestClient]


@pytest.fixture
def make_client(
    session_factory: sessionmaker[Session],
    memory: RedisChatMemory,
    store: VectorStore,
    embedder: Embedder,
) -> Iterator[MakeClient]:
    def _make(llm: FakeLLM) -> TestClient:
        def override_db() -> Iterator[Session]:
            db = session_factory()
            try:
                yield db
            finally:
                db.close()

        service = make_chat_service(llm, memory, store, embedder)
        app.dependency_overrides[get_db] = override_db
        app.dependency_overrides[get_chat_service] = lambda: service
        app.dependency_overrides[get_chat_memory] = lambda: memory
        return TestClient(app)

    yield _make
    app.dependency_overrides.clear()


def test_chat_endpoint_books_an_interview_and_lists_it(make_client: MakeClient) -> None:
    client = make_client(
        FakeLLM(
            [
                analysis_json(
                    "booking", name="Binu Rai", email="binu@example.com", date="2099-01-05", time="10:00"
                )
            ]
        )
    )
    response = client.post("/chat", json={"session_id": "demo", "message": "Book me in please"})

    assert response.status_code == 200
    body = response.json()
    assert body["intent"] == "booking"
    assert body["booking"]["email"] == "binu@example.com"

    listed = client.get("/bookings").json()
    assert [b["name"] for b in listed] == ["Binu Rai"]


def test_history_and_clear_endpoints(make_client: MakeClient) -> None:
    client = make_client(FakeLLM([analysis_json("question", standalone_question="Hi?")]))
    client.post("/chat", json={"session_id": "demo", "message": "Hi"})

    history = client.get("/chat/demo/history").json()
    assert [m["role"] for m in history] == ["user", "assistant"]

    assert client.delete("/chat/demo").status_code == 204
    assert client.get("/chat/demo/history").json() == []


@pytest.mark.parametrize(
    "payload",
    [
        {"session_id": "bad id!", "message": "hi"},
        {"session_id": "ok", "message": "   "},
        {"message": "hi"},
    ],
)
def test_request_validation(make_client: MakeClient, payload: dict[str, str]) -> None:
    client = make_client(FakeLLM([]))
    assert client.post("/chat", json=payload).status_code == 422


def test_llm_failure_becomes_502(make_client: MakeClient) -> None:
    class BrokenLLM(FakeLLM):
        def generate(
            self,
            prompt: str,
            *,
            system: str = "",
            json_mode: bool = False,
            temperature: float = 0.2,
        ) -> str:
            raise LLMError("Gemini request failed: quota exceeded")

    client = make_client(BrokenLLM([]))
    response = client.post("/chat", json={"session_id": "demo", "message": "hello"})
    assert response.status_code == 502
    assert "quota" in response.json()["detail"]


def test_redis_failure_becomes_503(make_client: MakeClient) -> None:
    class BrokenMemory(RedisChatMemory):
        def get_messages(self, session_id: str, limit: int | None = None) -> list[ChatMessage]:
            raise RedisConnectionError("down")

    import fakeredis

    client = make_client(FakeLLM([]))
    broken = BrokenMemory(fakeredis.FakeRedis(decode_responses=True), 60, 20)
    app.dependency_overrides[get_chat_memory] = lambda: broken
    assert client.get("/chat/demo/history").status_code == 503
