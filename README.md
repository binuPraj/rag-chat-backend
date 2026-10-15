# RAG Chat Backend

A FastAPI backend with two REST APIs:

1. **Document Ingestion API** – upload `.pdf` / `.txt` files, extract text, split it with one of two
   selectable chunking strategies, embed the chunks and store them in **Qdrant**. Document metadata is
   saved in **SQL** (SQLite by default).
2. **Conversational RAG API** – answers questions about the ingested documents with a **custom RAG
   pipeline** (no `RetrievalQAChain`), keeps **multi-turn chat memory in Redis**, and can **book
   interviews** (name, email, date, time) through the LLM. Bookings are stored in SQL.

No FAISS/Chroma, no UI. The code is modular and fully type-annotated (`mypy --strict` clean).

## Architecture
POST /ingest
  file ─► extract text ─► chunk (fixed | sentence) ─► embed ─► Qdrant (vectors + text)
                                          └──────────────────► SQL  (document metadata)

POST /chat
  message ─► Redis (history + half-filled booking)
          ─► analyzer (1 LLM call -> JSON: intent, standalone question, booking fields)
              ├─ question ─► embed ─► Qdrant top-k ─► hand-built prompt ─► LLM ─► answer
              ├─ booking  ─► validate in code ─► ask for missing fields ─► save to SQL
              └─ cancel   ─► clear the draft
```

## Tech stack

| Concern | Choice |
|---|---|
| API | FastAPI, Pydantic v2 |
| Vector DB | Qdrant (Docker) |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` (local, 384 dims, cosine) |
| LLM | Google Gemini via `google-genai` (model set with `LLM_MODEL`) |
| Chat memory | Redis (Docker) |
| Metadata + bookings | SQLAlchemy 2.0 + SQLite (swap `DATABASE_URL` for Postgres) |
| Tooling | pytest, mypy `--strict` |

## Setup

Requirements: Python 3.11+, Docker Desktop, a Gemini API key
(free from <https://aistudio.google.com>).

```bash
git clone <this-repo-url>
cd rag-chat-backend

python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env              # Windows: copy .env.example .env
# edit .env and set GEMINI_API_KEY (and LLM_MODEL if the default is not available to your key)

docker compose up -d              # starts Qdrant (6333) and Redis (6379)
uvicorn app.main:app --reload
```

Open the interactive docs at <http://localhost:8000/docs>.

The first upload is slower because the embedding model (~90 MB) is downloaded and loaded once.

```

### Configuration (`.env`)

| Variable | Default | Meaning |
|---|---|---|
| `GEMINI_API_KEY` | – | Required for `/chat` |
| `LLM_MODEL` | `gemini-3.5-flash-lite` | Gemini model name |
| `QDRANT_HOST` / `QDRANT_PORT` | `localhost` / `6333` | Qdrant location |
| `QDRANT_COLLECTION` | `documents` | Collection name |
| `REDIS_URL` | `redis://localhost:6379/0` | Redis location |
| `CHAT_TTL_SECONDS` | `3600` | A session is forgotten after this much inactivity |
| `DATABASE_URL` | `sqlite:///./app.db` | SQLAlchemy URL |
| `EMBEDDING_MODEL` / `EMBEDDING_DIM` | MiniLM / `384` | Must match each other |

## API

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/ingest` | Upload a `.pdf`/`.txt` file, choose `strategy` = `fixed` or `sentence` |
| `GET` | `/documents` | List ingested documents (metadata from SQL) |
| `POST` | `/chat` | Ask a question or take part in an interview booking |
| `GET` | `/chat/{session_id}/history` | Show the remembered conversation |
| `DELETE` | `/chat/{session_id}` | Forget a session |
| `GET` | `/bookings` | List saved interview bookings |
| `GET` | `/health` | Liveness check |

### Examples

A sample document is included at `samples/company_faq.txt`.
(On Windows PowerShell use `curl.exe`, or simply use the Swagger UI at `/docs`.)

```bash
# 1. Ingest with the sentence-aware strategy
curl -X POST http://localhost:8000/ingest \
  -F "file=@samples/company_faq.txt" -F "strategy=sentence"

# 2. Multi-turn questions (same session_id)
curl -X POST http://localhost:8000/chat -H "Content-Type: application/json" \
  -d '{"session_id": "demo", "message": "What plans does the company offer?"}'
curl -X POST http://localhost:8000/chat -H "Content-Type: application/json" \
  -d '{"session_id": "demo", "message": "What does the second one include?"}'

# 3. Book an interview over several messages
curl -X POST http://localhost:8000/chat -H "Content-Type: application/json" \
  -d '{"session_id": "book1", "message": "I would like to book an interview"}'
curl -X POST http://localhost:8000/chat -H "Content-Type: application/json" \
  -d '{"session_id": "book1", "message": "I am Binu Rai, binu@example.com"}'
curl -X POST http://localhost:8000/chat -H "Content-Type: application/json" \
  -d '{"session_id": "book1", "message": "Next Monday at 2pm"}'

curl http://localhost:8000/bookings
```

A `/chat` response contains the `answer`, the detected `intent` (`question`, `booking` or
`cancel_booking`), the `sources` (file, chunk, score, snippet) the answer was based on, and the
`booking` once one is saved.

## Design decisions

- **Two chunking strategies behind one protocol.** `fixed` splits into 800-character windows with a
  100-character overlap (predictable, never loses a boundary fact). `sentence` packs whole sentences up
  to 800 characters with a one-sentence overlap (reads naturally, embeds more cleanly). The API picks
  one with `get_chunker(strategy)` and never branches on type.
- **Custom RAG.** `services/rag.py` embeds the question, searches Qdrant, builds the prompt by hand
  from numbered context excerpts plus recent chat, and calls the LLM. The model is told to answer only
  from the context and to say so when the answer is missing. If nothing is retrieved, the LLM is not
  called.
- **Multi-turn queries.** The last 6 messages are sent with every turn. One analyzer call also rewrites
  a vague follow-up ("what does the second one include?") into a standalone question, so retrieval
  searches for the right thing.
- **Booking: the LLM extracts, code decides.** The analyzer returns name/email/date/time as JSON
  (relative dates are resolved against today's date given in the prompt). Python then validates the email,
  the date (not in the past) and the time (`HH:MM`), asks only for what is still missing, and saves the row
  when all four are valid. A half-filled booking lives in Redis, so it survives between messages; a
  question asked mid-booking is answered without losing it.
- **Redis memory.** `chat:<session>:messages` is a list trimmed to the newest 20 messages;
  `chat:<session>:booking` holds the draft. Both expire after `CHAT_TTL_SECONDS` of inactivity.
- **Consistency.** Chunk IDs are deterministic (`uuid5(document_id:index)`), so re-ingesting does not
  duplicate points. If saving metadata fails after the vectors were written, the vectors are removed.
- **Swappable parts.** `Embedder`, `LLM`, `ChatMemory` and `Chunker` are `Protocol`s, so providers can be
  replaced (or faked in tests) without touching the rest of the code.
- **Errors.** Unsupported file type → 415; empty/unreadable file → 422; oversized file (>10 MB) → 413;
  LLM failure → 502; Redis unavailable → 503.
- **Session ids** are restricted to letters, digits, `-` and `_`, so they cannot inject odd Redis keys.

## Project structure

```
app/
  api/        ingest.py, chat.py, bookings.py        # HTTP layer only
  core/       config.py, database.py                 # settings, SQLAlchemy engine/session
  models/     document.py, booking.py                # SQL tables
  schemas/    document.py, booking.py, chat.py       # request/response models
  services/   extraction.py, chunking.py, embeddings.py, vector_store.py, ingestion.py,
              llm.py, memory.py, intent.py, rag.py, booking.py, chat.py
  main.py
tests/        unit + API tests (no external servers needed)
samples/      company_faq.txt
docker-compose.yml
```

## Tests and type checking

```bash
pytest          # 51 tests; use in-memory Qdrant, SQLite and Redis (fakeredis) plus a fake LLM
mypy app tests  # strict mode
```

The tests need no Docker, no API key and no model download.

## Limitations and next steps

- Scanned PDFs (images without text) are rejected; OCR is not included.
- All documents share one Qdrant collection; there is no per-user separation or authentication.
- Relative dates such as "next Monday" rely on the LLM; the saved date is always shown back to the user.
- Booking confirmations are stored but no email is sent.
- Possible additions: streaming answers, re-ranking, per-document filtering from the chat API, a Dockerfile
  for the API itself.