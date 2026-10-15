import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models.booking import Booking
from app.services.embeddings import Embedder
from app.services.memory import ChatMessage, RedisChatMemory
from app.services.vector_store import VectorStore
from tests.conftest import TODAY, make_chat_service
from tests.fakes import FakeLLM, analysis_json

DOC_CHUNKS = [
    "Qdrant is a vector database used for similarity search.",
    "The company offers remote interviews every weekday.",
]


@pytest.fixture
def loaded_store(store: VectorStore, embedder: Embedder) -> VectorStore:
    store.upsert_chunks(
        document_id="doc-1",
        filename="handbook.txt",
        strategy="fixed",
        chunks=DOC_CHUNKS,
        vectors=embedder.embed_documents(DOC_CHUNKS),
    )
    return store


def test_question_prompt_contains_retrieved_context(
    db: Session, memory: RedisChatMemory, loaded_store: VectorStore, embedder: Embedder
) -> None:
    llm = FakeLLM(
        [
            analysis_json("question", standalone_question="What is Qdrant?"),
            "Qdrant is a vector database [handbook.txt].",
        ]
    )
    service = make_chat_service(llm, memory, loaded_store, embedder)

    result = service.handle(db, session_id="s1", message="What is Qdrant?", today=TODAY)

    assert result.intent == "question"
    assert "vector database" in result.answer
    assert result.sources[0].filename == "handbook.txt"
    _, rag_prompt = llm.calls[1]
    assert "Qdrant is a vector database used for similarity search." in rag_prompt
    assert "Question: What is Qdrant?" in rag_prompt


def test_follow_up_uses_history_and_rewritten_question(
    db: Session, memory: RedisChatMemory, loaded_store: VectorStore, embedder: Embedder
) -> None:
    llm = FakeLLM(
        [
            analysis_json("question", standalone_question="What is Qdrant?"),
            "It is a vector database.",
            analysis_json("question", standalone_question="What is Qdrant used for?"),
            "It is used for similarity search.",
        ]
    )
    service = make_chat_service(llm, memory, loaded_store, embedder)
    service.handle(db, session_id="s1", message="What is Qdrant?", today=TODAY)
    service.handle(db, session_id="s1", message="What is it used for?", today=TODAY)

    analyzer_prompt = llm.calls[2][1]
    rag_prompt = llm.calls[3][1]
    assert "user: What is Qdrant?" in analyzer_prompt  # history reached the analyzer
    assert "assistant: It is a vector database." in rag_prompt  # and the answer prompt
    assert "Question: What is Qdrant used for?" in rag_prompt  # rewritten, standalone
    assert len(memory.get_messages("s1")) == 4


def test_no_documents_skips_the_answer_llm_call(
    db: Session, memory: RedisChatMemory, store: VectorStore, embedder: Embedder
) -> None:
    llm = FakeLLM([analysis_json("question", standalone_question="Anything?")])
    service = make_chat_service(llm, memory, store, embedder)

    result = service.handle(db, session_id="s1", message="Anything?", today=TODAY)

    assert "couldn't find" in result.answer
    assert len(llm.calls) == 1


def test_malformed_analysis_falls_back_to_question(
    db: Session, memory: RedisChatMemory, loaded_store: VectorStore, embedder: Embedder
) -> None:
    llm = FakeLLM(["this is not json", "Remote interviews run every weekday."])
    service = make_chat_service(llm, memory, loaded_store, embedder)

    result = service.handle(db, session_id="s1", message="remote interviews?", today=TODAY)

    assert result.intent == "question"
    assert "Question: remote interviews?" in llm.calls[1][1]


def test_booking_collected_over_several_turns_is_saved(
    db: Session, memory: RedisChatMemory, store: VectorStore, embedder: Embedder
) -> None:
    llm = FakeLLM(
        [
            analysis_json("booking"),
            analysis_json("booking", name="Binu Rai", email="binu@example.com"),
            analysis_json("booking", date="2026-10-20", time="14:30"),
        ]
    )
    service = make_chat_service(llm, memory, store, embedder)

    first = service.handle(db, session_id="s1", message="I'd like to book an interview", today=TODAY)
    assert "your full name" in first.answer
    assert first.booking is None

    second = service.handle(db, session_id="s1", message="Binu Rai, binu@example.com", today=TODAY)
    assert "your preferred date and your preferred time" in second.answer
    assert memory.get_draft("s1").name == "Binu Rai"

    third = service.handle(db, session_id="s1", message="Oct 20 at 2:30pm", today=TODAY)
    assert third.booking is not None
    assert "booked" in third.answer

    rows = db.scalars(select(Booking)).all()
    assert len(rows) == 1
    assert rows[0].email == "binu@example.com"
    assert rows[0].interview_date.isoformat() == "2026-10-20"
    assert rows[0].interview_time.strftime("%H:%M") == "14:30"
    assert memory.get_draft("s1").name is None  # draft cleared after saving


def test_invalid_email_is_rejected_and_nothing_saved(
    db: Session, memory: RedisChatMemory, store: VectorStore, embedder: Embedder
) -> None:
    llm = FakeLLM(
        [analysis_json("booking", name="Binu Rai", email="nope", date="2026-10-20", time="10:00")]
    )
    service = make_chat_service(llm, memory, store, embedder)

    result = service.handle(db, session_id="s1", message="book it", today=TODAY)

    assert result.booking is None
    assert "doesn't look valid" in result.answer
    assert db.scalars(select(Booking)).all() == []


def test_past_date_is_rejected(
    db: Session, memory: RedisChatMemory, store: VectorStore, embedder: Embedder
) -> None:
    llm = FakeLLM([analysis_json("booking", date="2026-01-01")])
    service = make_chat_service(llm, memory, store, embedder)

    result = service.handle(db, session_id="s1", message="book for 1 Jan", today=TODAY)

    assert "in the past" in result.answer
    assert memory.get_draft("s1").date is None


def test_question_during_booking_keeps_the_draft_and_reminds(
    db: Session, memory: RedisChatMemory, loaded_store: VectorStore, embedder: Embedder
) -> None:
    llm = FakeLLM(
        [
            analysis_json("booking", name="Binu Rai"),
            analysis_json("question", standalone_question="What is Qdrant?"),
            "Qdrant is a vector database.",
        ]
    )
    service = make_chat_service(llm, memory, loaded_store, embedder)
    service.handle(db, session_id="s1", message="Book an interview, I'm Binu Rai", today=TODAY)

    result = service.handle(db, session_id="s1", message="What is Qdrant?", today=TODAY)

    assert "vector database" in result.answer
    assert "still in progress" in result.answer
    assert memory.get_draft("s1").name == "Binu Rai"


def test_cancel_clears_the_draft(
    db: Session, memory: RedisChatMemory, store: VectorStore, embedder: Embedder
) -> None:
    llm = FakeLLM([analysis_json("booking", name="Binu Rai"), analysis_json("cancel_booking")])
    service = make_chat_service(llm, memory, store, embedder)
    service.handle(db, session_id="s1", message="book, I'm Binu Rai", today=TODAY)

    result = service.handle(db, session_id="s1", message="never mind", today=TODAY)

    assert "cancelled" in result.answer
    assert memory.get_draft("s1").name is None


def test_sessions_are_isolated(
    db: Session, memory: RedisChatMemory, store: VectorStore, embedder: Embedder
) -> None:
    llm = FakeLLM([analysis_json("booking", name="Binu Rai"), analysis_json("booking")])
    service = make_chat_service(llm, memory, store, embedder)
    service.handle(db, session_id="a", message="book, I'm Binu Rai", today=TODAY)
    service.handle(db, session_id="b", message="book an interview", today=TODAY)

    assert memory.get_draft("a").name == "Binu Rai"
    assert memory.get_draft("b").name is None


def test_redis_memory_trims_old_messages_and_sets_expiry() -> None:
    import fakeredis

    client = fakeredis.FakeRedis(decode_responses=True)
    memory = RedisChatMemory(client, ttl_seconds=120, max_messages=4)
    for i in range(6):
        memory.append("s1", ChatMessage(role="user", content=f"m{i}"))

    assert [m.content for m in memory.get_messages("s1")] == ["m2", "m3", "m4", "m5"]
    assert 0 < client.ttl("chat:s1:messages") <= 120
    memory.clear("s1")
    assert memory.get_messages("s1") == []
