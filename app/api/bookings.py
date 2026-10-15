from collections.abc import Sequence
from typing import Annotated

from fastapi import APIRouter, Depends
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.models.booking import Booking
from app.schemas.booking import BookingOut

router = APIRouter(tags=["bookings"])


@router.get("/bookings", response_model=list[BookingOut], summary="List saved interview bookings")
def list_bookings(db: Annotated[Session, Depends(get_db)]) -> Sequence[Booking]:
    return db.scalars(select(Booking).order_by(Booking.created_at.desc())).all()
