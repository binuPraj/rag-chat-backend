import uuid

from sqlalchemy.orm import Session

from app.models.document import Document
from app.services.chunking import ChunkStrategy, get_chunker
from app.services.embeddings import Embedder
from app.services.extraction import EmptyDocumentError, extract_text
from app.services.vector_store import VectorStore


class IngestionService:
    def __init__(self, embedder: Embedder, store: VectorStore) -> None:
        self._embedder = embedder
        self._store = store

    def ingest(
        self,
        db: Session,
        *,
        filename: str,
        data: bytes,
        strategy: ChunkStrategy,
    ) -> Document:
        text = extract_text(filename, data)

        chunks = get_chunker(strategy).chunk(text)
        if not chunks:
            raise EmptyDocumentError("Document produced no chunks")

        vectors = self._embedder.embed_documents(chunks)

        document_id = str(uuid.uuid4())
        self._store.upsert_chunks(
            document_id=document_id,
            filename=filename,
            strategy=strategy.value,
            chunks=chunks,
            vectors=vectors,
        )

        document = Document(
            id=document_id,
            filename=filename,
            strategy=strategy.value,
            chunk_count=len(chunks),
            char_count=len(text),
        )
        try:
            db.add(document)
            db.commit()
        except Exception:
            # Keep the two stores consistent: no vectors without a metadata row.
            db.rollback()
            self._store.delete_document(document_id)
            raise
        return document
