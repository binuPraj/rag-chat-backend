from functools import lru_cache
from typing import Literal, Protocol

from pydantic import BaseModel
from redis import Redis

from app.core.config import get_settings
from app.schemas.booking import BookingDraft


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class ChatMemory(Protocol):
    def get_messages(self, session_id: str, limit: int | None = None) -> list[ChatMessage]: ...

    def append(self, session_id: str, *messages: ChatMessage) -> None: ...

    def get_draft(self, session_id: str) -> BookingDraft: ...

    def save_draft(self, session_id: str, draft: BookingDraft) -> None: ...

    def clear_draft(self, session_id: str) -> None: ...

    def clear(self, session_id: str) -> None: ...


class RedisChatMemory:
    def __init__(self, client: Redis, ttl_seconds: int, max_messages: int) -> None:
        self._client = client
        self._ttl = ttl_seconds
        self._max_messages = max_messages

    @staticmethod
    def _messages_key(session_id: str) -> str:
        return f"chat:{session_id}:messages"

    @staticmethod
    def _draft_key(session_id: str) -> str:
        return f"chat:{session_id}:booking"

    def get_messages(self, session_id: str, limit: int | None = None) -> list[ChatMessage]:
        count = limit or self._max_messages
        raw_items = self._client.lrange(self._messages_key(session_id), -count, -1)
        return [ChatMessage.model_validate_json(item) for item in raw_items]

    def append(self, session_id: str, *messages: ChatMessage) -> None:
        if not messages:
            return
        key = self._messages_key(session_id)
        pipe = self._client.pipeline()
        pipe.rpush(key, *[m.model_dump_json() for m in messages])
        pipe.ltrim(key, -self._max_messages, -1)  # keep only the newest N
        pipe.expire(key, self._ttl)  # every new message refreshes the timer
        pipe.execute()

    def get_draft(self, session_id: str) -> BookingDraft:
        raw = self._client.get(self._draft_key(session_id))
        return BookingDraft.model_validate_json(raw) if raw else BookingDraft()

    def save_draft(self, session_id: str, draft: BookingDraft) -> None:
        self._client.set(self._draft_key(session_id), draft.model_dump_json(), ex=self._ttl)

    def clear_draft(self, session_id: str) -> None:
        self._client.delete(self._draft_key(session_id))

    def clear(self, session_id: str) -> None:
        self._client.delete(self._messages_key(session_id), self._draft_key(session_id))


@lru_cache
def get_chat_memory() -> RedisChatMemory:
    settings = get_settings()
    client = Redis.from_url(settings.redis_url, decode_responses=True)
    return RedisChatMemory(client, settings.chat_ttl_seconds, settings.max_history_messages)
