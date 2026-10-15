from typing import Annotated

from pydantic import BaseModel, Field, StringConstraints

from app.schemas.booking import BookingOut

TrimmedText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=2000)]


class ChatRequest(BaseModel):
    session_id: str = Field(
        min_length=1,
        max_length=64,
        pattern=r"^[A-Za-z0-9_-]+$",
        description="Any id you choose, e.g. 'user-123'. Letters, digits, '-' and '_' only.",
    )
    message: TrimmedText


class SourceOut(BaseModel):

    filename: str
    chunk_index: int
    score: float
    snippet: str


class ChatResponse(BaseModel):

    session_id: str
    answer: str
    intent: str
    sources: list[SourceOut]
    booking: BookingOut | None = None
