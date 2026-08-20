import uuid
from collections.abc import Sequence
from dataclasses import dataclass

from sqlalchemy import Float, String, bindparam, cast, select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from docintel.db.models import Document, DocumentChunk, Vector1536
from docintel.providers.embeddings import (
    EMBEDDING_DIMENSIONS,
    EMBEDDING_MODEL,
    EmbeddingProvider,
    EmbeddingProviderFailure,
    EmbeddingProviderUnconfigured,
    embedding_to_pgvector,
    validate_embedding_vectors,
)

MAX_QUESTION_CHARACTERS = 1000
MAX_RETRIEVAL_DOCUMENTS = 10
RETRIEVAL_TOP_K = 4


class RetrievalError(Exception):
    """Base error for the internal retrieval contract."""


class InvalidRetrievalRequestError(RetrievalError):
    """Question or document selection violates the M2 contract."""


class DocumentsNotSearchableError(RetrievalError):
    def __init__(
        self,
        *,
        missing: Sequence[uuid.UUID] = (),
        ingested: Sequence[uuid.UUID] = (),
        incompatible: Sequence[uuid.UUID] = (),
    ) -> None:
        super().__init__("Every requested document must exist and be indexed with the v1 model.")
        self.missing = tuple(missing)
        self.ingested = tuple(ingested)
        self.incompatible = tuple(incompatible)


class RetrievalProviderError(RetrievalError):
    """A sanitized question-embedding failure."""


class RetrievalProviderUnconfigured(RetrievalProviderError):
    """The question-embedding provider has no credentials."""


@dataclass(frozen=True, slots=True)
class RetrievalResult:
    rank: int
    document_id: uuid.UUID
    filename: str
    page_number: int
    chunk_id: uuid.UUID
    chunk_index: int
    content: str
    similarity: float


def _validate_request(
    question: str, document_ids: Sequence[uuid.UUID]
) -> tuple[str, list[uuid.UUID]]:
    normalized_question = question.strip()
    if not normalized_question or len(question) > MAX_QUESTION_CHARACTERS:
        raise InvalidRetrievalRequestError("Question length must be between 1 and 1000 characters.")

    requested_ids = list(document_ids)
    if not 1 <= len(requested_ids) <= MAX_RETRIEVAL_DOCUMENTS:
        raise InvalidRetrievalRequestError("Between 1 and 10 document IDs are required.")
    if len(set(requested_ids)) != len(requested_ids):
        raise InvalidRetrievalRequestError("Document IDs must not contain duplicates.")
    return normalized_question, requested_ids


def _validate_documents(session: Session, document_ids: list[uuid.UUID]) -> None:
    try:
        selected = session.execute(
            select(
                Document.id,
                Document.indexed_at,
                Document.embedding_model,
                Document.embedding_dimensions,
            ).where(Document.id.in_(document_ids))
        ).all()
    except SQLAlchemyError as error:
        session.rollback()
        raise RetrievalError("Document eligibility could not be checked.") from error

    states = {row.id: row for row in selected}
    missing = [document_id for document_id in document_ids if document_id not in states]
    ingested = [
        document_id
        for document_id in document_ids
        if document_id in states and states[document_id].indexed_at is None
    ]
    incompatible = [
        document_id
        for document_id in document_ids
        if document_id in states
        and states[document_id].indexed_at is not None
        and (
            states[document_id].embedding_model != EMBEDDING_MODEL
            or states[document_id].embedding_dimensions != EMBEDDING_DIMENSIONS
        )
    ]
    session.rollback()
    if missing or ingested or incompatible:
        raise DocumentsNotSearchableError(
            missing=missing,
            ingested=ingested,
            incompatible=incompatible,
        )


def validate_retrieval_request(
    session: Session,
    *,
    question: str,
    document_ids: Sequence[uuid.UUID],
) -> tuple[str, list[uuid.UUID]]:
    normalized_question, requested_ids = _validate_request(question, document_ids)
    _validate_documents(session, requested_ids)
    return normalized_question, requested_ids


def retrieve_chunks(
    session: Session,
    *,
    question: str,
    document_ids: Sequence[uuid.UUID],
    embedding_provider: EmbeddingProvider,
) -> list[RetrievalResult]:
    normalized_question, requested_ids = validate_retrieval_request(
        session,
        question=question,
        document_ids=document_ids,
    )
    if (
        embedding_provider.model != EMBEDDING_MODEL
        or embedding_provider.dimensions != EMBEDDING_DIMENSIONS
    ):
        raise RetrievalProviderError("The embedding provider is incompatible with indexed data.")
    try:
        query_vectors = embedding_provider.embed([normalized_question])
        query_vector = validate_embedding_vectors(query_vectors, expected_count=1)[0]
    except EmbeddingProviderUnconfigured as error:
        raise RetrievalProviderUnconfigured(
            "The question embedding provider is not configured."
        ) from error
    except EmbeddingProviderFailure as error:
        raise RetrievalProviderError("The question embedding could not be generated.") from error

    query_parameter = bindparam(
        "query_embedding",
        value=embedding_to_pgvector(query_vector),
        type_=String(),
    )
    query_embedding = cast(query_parameter, Vector1536())
    cosine_distance = cast(DocumentChunk.embedding.op("<=>")(query_embedding), Float).label(
        "cosine_distance"
    )
    statement = (
        select(
            DocumentChunk.document_id,
            Document.filename,
            DocumentChunk.page_number,
            DocumentChunk.id.label("chunk_id"),
            DocumentChunk.chunk_index,
            DocumentChunk.content,
            cosine_distance,
        )
        .join(Document, Document.id == DocumentChunk.document_id)
        .where(
            DocumentChunk.document_id.in_(requested_ids),
            Document.indexed_at.is_not(None),
        )
        .order_by(
            cosine_distance.asc(),
            DocumentChunk.document_id.asc(),
            DocumentChunk.chunk_index.asc(),
        )
        .limit(RETRIEVAL_TOP_K)
    )

    try:
        rows = session.execute(statement).all()
    except SQLAlchemyError as error:
        session.rollback()
        raise RetrievalError("Vector retrieval failed.") from error

    return [
        RetrievalResult(
            rank=rank,
            document_id=row.document_id,
            filename=row.filename,
            page_number=row.page_number,
            chunk_id=row.chunk_id,
            chunk_index=row.chunk_index,
            content=row.content,
            similarity=1.0 - float(row.cosine_distance),
        )
        for rank, row in enumerate(rows, start=1)
    ]
