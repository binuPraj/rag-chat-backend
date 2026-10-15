from collections.abc import Sequence
from typing import Annotated

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.models.document import Document
from app.schemas.document import DocumentOut
from app.services.chunking import ChunkStrategy
from app.services.embeddings import get_embedder
from app.services.extraction import DocumentError, UnsupportedFileTypeError
from app.services.ingestion import IngestionService
from app.services.vector_store import get_vector_store

router = APIRouter(tags=["ingestion"])

MAX_UPLOAD_BYTES = 10 * 1024 * 1024  # 10 MB


def get_ingestion_service() -> IngestionService:
    return IngestionService(get_embedder(), get_vector_store())


@router.post(
    "/ingest",
    response_model=DocumentOut,
    status_code=status.HTTP_201_CREATED,
    summary="Upload a .pdf or .txt file, chunk it, embed it and store it",
)
def ingest_document(
    file: Annotated[UploadFile, File(description="A .pdf or .txt file")],
    db: Annotated[Session, Depends(get_db)],
    service: Annotated[IngestionService, Depends(get_ingestion_service)],
    strategy: Annotated[
        ChunkStrategy,
        Form(description="'fixed' = fixed size with overlap, 'sentence' = sentence-aware"),
    ] = ChunkStrategy.FIXED,
) -> Document:
    # Read one byte past the limit so we can tell "too big" from "exactly at the limit".
    data = file.file.read(MAX_UPLOAD_BYTES + 1)
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File is larger than {MAX_UPLOAD_BYTES // (1024 * 1024)} MB",
        )

    try:
        return service.ingest(db, filename=file.filename or "", data=data, strategy=strategy)
    except UnsupportedFileTypeError as exc:
        raise HTTPException(status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, detail=str(exc)) from exc
    except DocumentError as exc:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc


@router.get(
    "/documents",
    response_model=list[DocumentOut],
    summary="List metadata of all ingested documents",
)
def list_documents(db: Annotated[Session, Depends(get_db)]) -> Sequence[Document]:
    return db.scalars(select(Document).order_by(Document.created_at.desc())).all()
