import datetime as dt

from pydantic import BaseModel, ConfigDict


class BookingDraft(BaseModel):

    name: str | None = None
    email: str | None = None
    date: str | None = None
    time: str | None = None


class BookingOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    name: str
    email: str
    interview_date: dt.date
    interview_time: dt.time
    created_at: dt.datetime
