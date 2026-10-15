import re
from dataclasses import dataclass
from datetime import date, datetime
from typing import Final

from sqlalchemy.orm import Session

from app.models.booking import Booking
from app.schemas.booking import BookingDraft

FIELD_ORDER: Final = ("name", "email", "date", "time")
FIELD_LABELS: Final = {
    "name": "your full name",
    "email": "your email address",
    "date": "your preferred date",
    "time": "your preferred time",
}
_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


@dataclass(frozen=True)
class MergeResult:
    draft: BookingDraft
    errors: list[str]


def _normalize(field: str, value: str, today: date) -> str:
    if field == "name":
        if len(value) < 2 or len(value) > 100 or not any(c.isalpha() for c in value):
            raise ValueError("Please give me your full name.")
        return value
    if field == "email":
        if not _EMAIL_RE.match(value):
            raise ValueError("That email address doesn't look valid.")
        return value
    if field == "date":
        try:
            parsed = date.fromisoformat(value)
        except ValueError:
            raise ValueError("I couldn't understand that date. Try a format like 2026-10-20.") from None
        if parsed < today:
            raise ValueError("That date is in the past. Please choose a future date.")
        return parsed.isoformat()
    if field == "time":
        try:
            return datetime.strptime(value, "%H:%M").strftime("%H:%M")
        except ValueError:
            raise ValueError("I couldn't understand that time. Try a format like 14:30.") from None
    raise ValueError(f"Unknown field {field}")


def merge_and_validate(draft: BookingDraft, update: BookingDraft, today: date) -> MergeResult:
    values = draft.model_dump()
    errors: list[str] = []
    for field in FIELD_ORDER:
        raw = getattr(update, field)
        if raw is None or not raw.strip():
            continue
        try:
            values[field] = _normalize(field, raw.strip(), today)
        except ValueError as exc:
            errors.append(str(exc))
    return MergeResult(BookingDraft(**values), errors)


def missing_fields(draft: BookingDraft) -> list[str]:
    return [field for field in FIELD_ORDER if not getattr(draft, field)]


def has_progress(draft: BookingDraft) -> bool:
    return any(getattr(draft, field) for field in FIELD_ORDER)


def _join(labels: list[str]) -> str:
    if len(labels) <= 1:
        return "".join(labels)
    return ", ".join(labels[:-1]) + " and " + labels[-1]


def describe_missing(missing: list[str]) -> str:
    return _join([FIELD_LABELS[field] for field in missing])


def booking_prompt(errors: list[str], missing: list[str]) -> str:
    parts: list[str] = []
    if errors:
        parts.append(" ".join(errors))
    if missing:
        intro = "I can book that interview for you." if not errors else "No problem."
        parts.append(f"{intro} I still need {describe_missing(missing)}.")
    else:
        parts.append("Please send a corrected value.")
    return " ".join(parts)


class BookingService:
    def save(self, db: Session, session_id: str, draft: BookingDraft) -> Booking:
        if missing_fields(draft):
            raise ValueError("Booking is incomplete")
        assert draft.name and draft.email and draft.date and draft.time  # for the type checker
        booking = Booking(
            session_id=session_id,
            name=draft.name,
            email=draft.email,
            interview_date=date.fromisoformat(draft.date),
            interview_time=datetime.strptime(draft.time, "%H:%M").time(),
        )
        db.add(booking)
        db.commit()
        return booking


def confirmation_text(booking: Booking) -> str:
    when = datetime.combine(booking.interview_date, booking.interview_time)
    return (
        f"Your interview is booked, {booking.name}! "
        f"I've saved it for {when:%A, %d %B %Y} at {when:%H:%M} under {booking.email}."
    )
