from typing import Annotated

from fastapi import APIRouter, Depends, Response, status
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.core.database import get_db
from app.schemas.booking import BookingOut
from app.schemas.chat import ChatRequest, ChatResponse, SourceOut
from app.services.booking import BookingService
from app.services.chat import ChatService
from app.services.embeddings import get_embedder
from app.services.intent import IntentAnalyzer
from app.services.llm import get_llm
from app.services.memory import ChatMemory, ChatMessage, get_chat_memory
from app.services.rag import RagService
from app.services.vector_store import get_vector_store

router = APIRouter(tags=["chat"])

SNIPPET_CHARS = 200


def get_chat_service() -> ChatService:
    settings = get_settings()
    llm = get_llm()
    return ChatService(
        memory=get_chat_memory(),
        analyzer=IntentAnalyzer(llm),
        rag=RagService(get_embedder(), get_vector_store(), llm, settings.retrieval_top_k),
        bookings=BookingService(),
        prompt_history=settings.prompt_history_messages,
    )


@router.post(
    "/chat",
    response_model=ChatResponse,
    summary="Send a message: ask about your documents or book an interview",
)
def chat(
    request: ChatRequest,
    db: Annotated[Session, Depends(get_db)],
    service: Annotated[ChatService, Depends(get_chat_service)],
) -> ChatResponse:
    result = service.handle(db, session_id=request.session_id, message=request.message)
    return ChatResponse(
        session_id=request.session_id,
        answer=result.answer,
        intent=result.intent,
        sources=[
            SourceOut(
                filename=hit.filename,
                chunk_index=hit.chunk_index,
                score=hit.score,
                snippet=hit.text[:SNIPPET_CHARS],
            )
            for hit in result.sources
        ],
        booking=BookingOut.model_validate(result.booking) if result.booking else None,
    )


@router.get(
    "/chat/{session_id}/history",
    response_model=list[ChatMessage],
    summary="Show the remembered conversation for a session",
)
def get_history(
    session_id: str, memory: Annotated[ChatMemory, Depends(get_chat_memory)]
) -> list[ChatMessage]:
    return memory.get_messages(session_id)


@router.delete(
    "/chat/{session_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Forget a session (history and any half-finished booking)",
)
def clear_session(
    session_id: str, memory: Annotated[ChatMemory, Depends(get_chat_memory)]
) -> Response:
    memory.clear(session_id)
    return Response(status_code=status.HTTP_204_NO_CONTENT)
