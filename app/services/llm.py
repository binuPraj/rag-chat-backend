from functools import lru_cache
from typing import Protocol

from google import genai
from google.genai import types

from app.core.config import get_settings


class LLMError(RuntimeError):
    """The language model could not produce a usable response."""


class LLM(Protocol):
    def generate(
        self,
        prompt: str,
        *,
        system: str = "",
        json_mode: bool = False,
        temperature: float = 0.2,
    ) -> str: ...


class GeminiLLM:
    def __init__(self, api_key: str, model: str) -> None:
        if not api_key:
            raise LLMError("GEMINI_API_KEY is not set. Add it to your .env file.")
        self._client = genai.Client(api_key=api_key)
        self._model = model

    def generate(
        self,
        prompt: str,
        *,
        system: str = "",
        json_mode: bool = False,
        temperature: float = 0.2,
    ) -> str:
        config = types.GenerateContentConfig(
            system_instruction=system or None,
            temperature=temperature,
            response_mime_type="application/json" if json_mode else None,
        )
        try:
            response = self._client.models.generate_content(
                model=self._model, contents=prompt, config=config
            )
        except Exception as exc:  # SDK/network errors are many; expose one clean type
            raise LLMError(f"Gemini request failed: {exc}") from exc

        text = response.text
        if not text:
            raise LLMError("Gemini returned an empty response (it may have been blocked).")
        return text


@lru_cache
def get_llm() -> LLM:
    settings = get_settings()
    return GeminiLLM(settings.gemini_api_key, settings.llm_model)
