import hashlib
import uuid
from datetime import UTC, datetime

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from sqlalchemy.orm import Session

from docintel.db.models import Document, DocumentChunk, DocumentPage
from docintel.documents.extraction import DocumentIngestionError, extract_pdf
from docintel.indexing.service import prepare_index
from docintel.providers.embeddings import (
    EMBEDDING_DIMENSIONS,
    EMBEDDING_MODEL,
    EmbeddingProvider,
    embedding_to_pgvector,
)


class DocumentAlreadyExistsError(DocumentIngestionError):
    status_code = 409
    code = "document_already_exists"
    message = "A document with the same content already exists."

    def __init__(self, document_id: uuid.UUID) -> None:
        super().__init__()
        self.document_id = document_id

    def response_body(self) -> dict[str, str]:
        body = super().response_body()
        body["document_id"] = str(self.document_id)
        return body


class PersistenceError(DocumentIngestionError):
    status_code = 500
    code = "document_persistence_failed"
    message = "The document could not be persisted."


def sha256_digest(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _find_by_hash(session: Session, content_hash: str) -> Document | None:
    return session.scalar(select(Document).where(Document.content_hash == content_hash))


def ingest_document(
    session: Session,
    *,
    filename: str,
    media_type: str,
    file_bytes: bytes,
    embedding_provider: EmbeddingProvider,
) -> Document:
    # Hashing original bytes provides exact deduplication without retaining the binary PDF.
    content_hash = sha256_digest(file_bytes)
    try:
        existing = _find_by_hash(session, content_hash)
    except SQLAlchemyError as error:
        session.rollback()
        raise PersistenceError from error
    existing_id = existing.id if existing is not None else None
    # End the pre-check transaction before CPU work and the external provider call.
    session.rollback()
    if existing_id is not None:
        raise DocumentAlreadyExistsError(existing_id)

    extracted = extract_pdf(file_bytes)
    indexed_chunks = prepare_index(extracted.pages, embedding_provider)
    document = Document(
        id=uuid.uuid4(),
        filename=filename,
        media_type=media_type,
        size_bytes=len(file_bytes),
        content_hash=content_hash,
        page_count=extracted.page_count,
        extracted_char_count=extracted.extracted_char_count,
        embedding_model=EMBEDDING_MODEL,
        embedding_dimensions=EMBEDDING_DIMENSIONS,
        indexed_at=datetime.now(UTC),
    )
    document.pages = [
        DocumentPage(
            document_id=document.id,
            page_number=page.page_number,
            content=page.content,
            char_count=page.char_count,
        )
        for page in extracted.pages
    ]
    pages_by_number = {page.page_number: page for page in document.pages}
    for indexed_chunk in indexed_chunks:
        chunk = DocumentChunk(
            id=uuid.uuid4(),
            document_id=document.id,
            page_number=indexed_chunk.chunk.page_number,
            chunk_index=indexed_chunk.chunk.chunk_index,
            content=indexed_chunk.chunk.content,
            token_count=indexed_chunk.chunk.token_count,
            embedding=embedding_to_pgvector(indexed_chunk.embedding),
        )
        pages_by_number[indexed_chunk.chunk.page_number].chunks.append(chunk)

    try:
        # Pages and validated chunks commit atomically after every provider call has finished.
        session.add(document)
        session.commit()
    except IntegrityError as error:
        session.rollback()
        try:
            raced_document = _find_by_hash(session, content_hash)
        except SQLAlchemyError as lookup_error:
            session.rollback()
            raise PersistenceError from lookup_error
        if raced_document is not None:
            raise DocumentAlreadyExistsError(raced_document.id) from error
        raise PersistenceError from error
    except SQLAlchemyError as error:
        session.rollback()
        raise PersistenceError from error
    return document


def list_documents(session: Session) -> list[Document]:
    try:
        return list(
            session.scalars(
                select(Document).order_by(Document.created_at.desc(), Document.id.asc())
            )
        )
    except SQLAlchemyError as error:
        session.rollback()
        raise PersistenceError from error


def delete_document(session: Session, document_id: uuid.UUID) -> bool:
    try:
        document = session.get(Document, document_id)
        if document is None:
            return False
        session.delete(document)
        session.commit()
    except SQLAlchemyError as error:
        session.rollback()
        raise PersistenceError from error
    return True
