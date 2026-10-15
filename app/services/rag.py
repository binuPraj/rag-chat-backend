from dataclasses import dataclass

from app.services.embeddings import Embedder
from app.services.llm import LLM
from app.services.memory import ChatMessage
from app.services.vector_store import SearchHit, VectorStore

RAG_SYSTEM_PROMPT = """You are a helpful assistant that answers questions using ONLY the \
context excerpts taken from the user's uploaded documents.

- If the answer is not in the context, say you could not find it in the documents. Do not guess.
- Be concise and accurate.
- Mention the source file name in square brackets, for example [handbook.pdf].
- If the user just greets you or makes small talk, reply politely and briefly.
"""

NO_DOCUMENTS_MESSAGE = (
    "I couldn't find any relevant information in the uploaded documents. "
    "Try uploading a document first, or rephrase your question."
)


def build_prompt(question: str, hits: list[SearchHit], history: list[ChatMessage]) -> str:
    context = "\n\n".join(
        f"[{number}] (source: {hit.filename})\n{hit.text}" for number, hit in enumerate(hits, start=1)
    )
    conversation = "\n".join(f"{m.role}: {m.content}" for m in history) or "(no earlier messages)"
    return (
        f"Context excerpts:\n{context}\n\n"
        f"Conversation so far:\n{conversation}\n\n"
        f"Question: {question}\n\n"
        "Answer:"
    )


@dataclass(frozen=True)
class RagAnswer:
    text: str
    sources: list[SearchHit]


class RagService:
    def __init__(self, embedder: Embedder, store: VectorStore, llm: LLM, top_k: int) -> None:
        self._embedder = embedder
        self._store = store
        self._llm = llm
        self._top_k = top_k

    def answer(self, question: str, history: list[ChatMessage]) -> RagAnswer:
        hits = self._store.search(self._embedder.embed_query(question), top_k=self._top_k)
        if not hits:
            return RagAnswer(NO_DOCUMENTS_MESSAGE, [])
        text = self._llm.generate(build_prompt(question, hits, history), system=RAG_SYSTEM_PROMPT)
        return RagAnswer(text.strip(), hits)
