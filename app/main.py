from fastapi import FastAPI

app=FastAPI(
    title="Palm Mind RAG Backend",
    description ="Document ingestion and conversational RAG with interview booking",
)

@app.get("/health", tags=["meta"])
def health() -> dict[str, str]:

    return {"status": "ok"}