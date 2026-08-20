import uuid
from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, field_validator

from docintel.db.models import Document
from docintel.rag.service import RAGResult


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


class QueryRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question: str = Field(min_length=1, max_length=1000)
    document_ids: list[uuid.UUID] = Field(min_length=1, max_length=10)

    @field_validator("question", mode="before")
    @classmethod
    def strip_question(cls, value: object) -> object:
        return value.strip() if isinstance(value, str) else value

    @field_validator("document_ids")
    @classmethod
    def reject_duplicate_document_ids(cls, document_ids: list[uuid.UUID]) -> list[uuid.UUID]:
        if len(set(document_ids)) != len(document_ids):
            raise ValueError("document_ids must not contain duplicates")
        return document_ids


class QuerySourceResponse(BaseModel):
    source_id: str
    document_id: uuid.UUID
    filename: str
    page_number: int
    chunk_id: uuid.UUID
    chunk_index: int
    text: str
    similarity: float


class QueryRetrievalResponse(BaseModel):
    top_k: int
    chunks_returned: int


class QueryResponse(BaseModel):
    question: str
    answer: str
    abstained: bool
    citation_ids: list[str]
    sources: list[QuerySourceResponse]
    retrieval: QueryRetrievalResponse

    @classmethod
    def from_result(cls, result: RAGResult) -> QueryResponse:
        return cls(
            question=result.question,
            answer=result.answer,
            abstained=result.abstained,
            citation_ids=list(result.citation_ids),
            sources=[
                QuerySourceResponse(
                    source_id=source.source_id,
                    document_id=source.document_id,
                    filename=source.filename,
                    page_number=source.page_number,
                    chunk_id=source.chunk_id,
                    chunk_index=source.chunk_index,
                    text=source.text,
                    similarity=source.similarity,
                )
                for source in result.sources
            ],
            retrieval=QueryRetrievalResponse(
                top_k=result.retrieval.top_k,
                chunks_returned=result.retrieval.chunks_returned,
            ),
        )
