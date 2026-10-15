import json
from datetime import date
from typing import Literal

from pydantic import BaseModel, ConfigDict, ValidationError

from app.schemas.booking import BookingDraft
from app.services.llm import LLM
from app.services.memory import ChatMessage

ANALYZER_SYSTEM_PROMPT = """You analyze the latest message in a chat with an assistant that \
answers questions about uploaded documents and can also book interviews.

Reply with ONE JSON object and nothing else, using exactly these keys:
{
  "intent": "question" | "booking" | "cancel_booking",
  "standalone_question": string,
  "name": string | null,
  "email": string | null,
  "date": string | null,
  "time": string | null
}

Rules:
- intent "booking": the user wants to schedule or book an interview, OR is supplying details \
(name, email, date, time) for a booking already in progress.
- intent "cancel_booking": the user wants to stop or cancel the booking in progress.
- intent "question": anything else, including a question asked while a booking is in progress.
- standalone_question: for "question", rewrite the latest message as a complete question that \
makes sense without the conversation (resolve words like "it", "that", "they"). Otherwise "".
- name, email, date, time: only values the user stated in the LATEST message. Never guess or \
reuse values from earlier turns. Use null when not stated.
- date must be YYYY-MM-DD. Resolve relative dates ("tomorrow", "next Friday") using today's date.
- time must be 24-hour HH:MM (for example "2 pm" becomes "14:00").
"""


class Analysis(BaseModel):
    model_config = ConfigDict(extra="ignore")

    intent: Literal["question", "booking", "cancel_booking"] = "question"
    standalone_question: str = ""
    name: str | None = None
    email: str | None = None
    date: str | None = None
    time: str | None = None

    def to_draft(self) -> BookingDraft:
        """The booking details found in this message."""
        return BookingDraft(name=self.name, email=self.email, date=self.date, time=self.time)


def _parse(raw: str) -> Analysis | None:
    text = raw.strip()
    if text.startswith("```"):  # models sometimes wrap JSON in a code fence
        text = text.strip("`").strip()
        if text.lower().startswith("json"):
            text = text[4:]
    try:
        return Analysis.model_validate_json(text)
    except ValidationError:
        return None


class IntentAnalyzer:
    def __init__(self, llm: LLM) -> None:
        self._llm = llm

    def analyze(
        self,
        message: str,
        history: list[ChatMessage],
        draft: BookingDraft,
        today: date,
    ) -> Analysis:
        conversation = "\n".join(f"{m.role}: {m.content}" for m in history) or "(none)"
        pending = draft.model_dump_json() if any(draft.model_dump().values()) else "none"
        prompt = (
            f"Today's date: {today.isoformat()} ({today:%A})\n"
            f"Booking in progress: {pending}\n\n"
            f"Conversation so far:\n{conversation}\n\n"
            f"Latest message: {json.dumps(message)}"
        )
        raw = self._llm.generate(prompt, system=ANALYZER_SYSTEM_PROMPT, json_mode=True, temperature=0.0)
        analysis = _parse(raw)
        if analysis is None:
            # Unreadable output: fall back to treating it as a plain question.
            return Analysis(intent="question", standalone_question=message)
        if analysis.intent == "question" and not analysis.standalone_question.strip():
            analysis.standalone_question = message
        return analysis
