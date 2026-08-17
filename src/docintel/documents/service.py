import hashlib
import uuid

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from sqlalchemy.orm import Session

from docintel.db.models import Document, DocumentPage
from docintel.documents.extraction import DocumentIngestionError, extract_pdf


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
) -> Document:
    # Hashing original bytes provides exact deduplication without retaining the binary PDF.
    content_hash = sha256_digest(file_bytes)
    try:
        existing = _find_by_hash(session, content_hash)
    except SQLAlchemyError as error:
        session.rollback()
        raise PersistenceError from error
    if existing is not None:
        session.rollback()
        raise DocumentAlreadyExistsError(existing.id)

    extracted = extract_pdf(file_bytes)
    document = Document(
        id=uuid.uuid4(),
        filename=filename,
        media_type=media_type,
        size_bytes=len(file_bytes),
        content_hash=content_hash,
        page_count=extracted.page_count,
        extracted_char_count=extracted.extracted_char_count,
        embedding_model=None,
        embedding_dimensions=None,
        indexed_at=None,
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

    try:
        # Document and every page commit atomically; future indexing remains a separate phase.
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
