
from dataclasses import dataclass
from datetime import date

from sqlalchemy.orm import Session

from app.models.booking import Booking
from app.services.booking import (
    BookingService,
    booking_prompt,
    confirmation_text,
    describe_missing,
    has_progress,
    merge_and_validate,
    missing_fields,
)
from app.services.intent import Analysis, IntentAnalyzer
from app.services.memory import ChatMemory, ChatMessage
from app.services.rag import RagService
from app.services.vector_store import SearchHit


@dataclass(frozen=True)
class ChatResult:
    answer: str
    intent: str
    sources: list[SearchHit]
    booking: Booking | None = None


class ChatService:
    def __init__(
        self,
        memory: ChatMemory,
        analyzer: IntentAnalyzer,
        rag: RagService,
        bookings: BookingService,
        prompt_history: int = 6,
    ) -> None:
        self._memory = memory
        self._analyzer = analyzer
        self._rag = rag
        self._bookings = bookings
        self._prompt_history = prompt_history

    def handle(
        self,
        db: Session,
        *,
        session_id: str,
        message: str,
        today: date | None = None,
    ) -> ChatResult:
        today = today or date.today()
        history = self._memory.get_messages(session_id, limit=self._prompt_history)
        draft = self._memory.get_draft(session_id)

        analysis = self._analyzer.analyze(message, history, draft, today)

        if analysis.intent == "cancel_booking":
            self._memory.clear_draft(session_id)
            result = ChatResult("No problem, I've cancelled the booking request.", analysis.intent, [])
        elif analysis.intent == "booking":
            result = self._handle_booking(db, session_id, analysis, today)
        else:
            result = self._handle_question(session_id, message, analysis, history)

        self._memory.append(
            session_id,
            ChatMessage(role="user", content=message),
            ChatMessage(role="assistant", content=result.answer),
        )
        return result

    def _handle_booking(
        self, db: Session, session_id: str, analysis: Analysis, today: date
    ) -> ChatResult:
        draft = self._memory.get_draft(session_id)
        merged = merge_and_validate(draft, analysis.to_draft(), today)
        missing = missing_fields(merged.draft)

        if not missing and not merged.errors:
            booking = self._bookings.save(db, session_id, merged.draft)
            self._memory.clear_draft(session_id)
            return ChatResult(confirmation_text(booking), "booking", [], booking)

        self._memory.save_draft(session_id, merged.draft)
        return ChatResult(booking_prompt(merged.errors, missing), "booking", [])

    def _handle_question(
        self, session_id: str, message: str, analysis: Analysis, history: list[ChatMessage]
    ) -> ChatResult:
        answer = self._rag.answer(analysis.standalone_question or message, history)
        text = answer.text

        # A question in the middle of a booking should not lose the booking.
        draft = self._memory.get_draft(session_id)
        missing = missing_fields(draft)
        if has_progress(draft) and missing:
            text += (
                "\n\n(Your interview booking is still in progress. "
                f"I still need {describe_missing(missing)}.)"
            )
        return ChatResult(text, "question", answer.sources)
