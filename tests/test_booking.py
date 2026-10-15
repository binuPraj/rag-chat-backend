from datetime import date

from app.schemas.booking import BookingDraft
from app.services.booking import (
    booking_prompt,
    describe_missing,
    has_progress,
    merge_and_validate,
    missing_fields,
)

TODAY = date(2026, 10, 14)


def test_merge_accepts_valid_values_and_normalizes() -> None:
    result = merge_and_validate(
        BookingDraft(), BookingDraft(name=" Binu Rai ", date="2026-10-20", time="9:05"), TODAY
    )
    assert result.errors == []
    assert result.draft.name == "Binu Rai"
    assert result.draft.date == "2026-10-20"
    assert result.draft.time == "09:05"


def test_merge_rejects_bad_email_and_keeps_others() -> None:
    result = merge_and_validate(
        BookingDraft(), BookingDraft(name="Binu Rai", email="not-an-email"), TODAY
    )
    assert result.draft.name == "Binu Rai"
    assert result.draft.email is None
    assert any("email" in e for e in result.errors)


def test_merge_rejects_past_date_and_bad_time() -> None:
    result = merge_and_validate(BookingDraft(), BookingDraft(date="2026-10-01", time="25:99"), TODAY)
    assert result.draft.date is None
    assert result.draft.time is None
    assert len(result.errors) == 2


def test_today_is_allowed() -> None:
    assert merge_and_validate(BookingDraft(), BookingDraft(date="2026-10-14"), TODAY).errors == []


def test_merge_keeps_existing_values() -> None:
    existing = BookingDraft(name="Binu Rai")
    result = merge_and_validate(existing, BookingDraft(email="b@example.com"), TODAY)
    assert result.draft.name == "Binu Rai"
    assert result.draft.email == "b@example.com"


def test_missing_fields_and_wording() -> None:
    draft = BookingDraft(name="Binu Rai", date="2026-10-20")
    assert missing_fields(draft) == ["email", "time"]
    assert describe_missing(["email", "time"]) == "your email address and your preferred time"
    assert has_progress(draft)
    assert not has_progress(BookingDraft())


def test_booking_prompt_mentions_errors_and_missing() -> None:
    text = booking_prompt(["That email address doesn't look valid."], ["email"])
    assert "doesn't look valid" in text
    assert "your email address" in text
