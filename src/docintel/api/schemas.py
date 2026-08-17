import uuid
from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel

from docintel.db.models import Document


class DocumentStatus(StrEnum):
    INGESTED = "INGESTED"
    INDEXED = "INDEXED"


class DocumentResponse(BaseModel):
    id: uuid.UUID
    filename: str
    media_type: str
    size_bytes: int
    content_hash: str
    page_count: int
    extracted_char_count: int
    status: DocumentStatus
    embedding_model: str | None
    embedding_dimensions: int | None
    created_at: datetime
    indexed_at: datetime | None

    @classmethod
    def from_model(cls, document: Document) -> DocumentResponse:
        status = (
            DocumentStatus.INDEXED if document.indexed_at is not None else DocumentStatus.INGESTED
        )
        return cls(
            id=document.id,
            filename=document.filename,
            media_type=document.media_type,
            size_bytes=document.size_bytes,
            content_hash=document.content_hash,
            page_count=document.page_count,
            extracted_char_count=document.extracted_char_count,
            status=status,
            embedding_model=document.embedding_model,
            embedding_dimensions=document.embedding_dimensions,
            created_at=document.created_at,
            indexed_at=document.indexed_at,
        )


class ErrorResponse(BaseModel):
    code: str
    message: str


class DuplicateDocumentResponse(ErrorResponse):
    document_id: uuid.UUID


class HealthResponse(BaseModel):
    status: str


class ReadinessResponse(BaseModel):
    status: str
