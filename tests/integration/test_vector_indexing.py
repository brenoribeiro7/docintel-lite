import uuid
from collections.abc import Sequence
from datetime import UTC, datetime

import pytest
from sqlalchemy import delete, func, inspect, select, text
from sqlalchemy.exc import DataError, IntegrityError
from sqlalchemy.orm import Session

from docintel.db.models import Document, DocumentChunk, DocumentPage
from docintel.documents.service import ingest_document
from docintel.providers.embeddings import (
    EMBEDDING_DIMENSIONS,
    EMBEDDING_MODEL,
    embedding_to_pgvector,
)
from docintel.retrieval.service import (
    DocumentsNotSearchableError,
    InvalidRetrievalRequestError,
    retrieve_chunks,
)
from tests.fakes import FakeEmbeddingProvider, controlled_vector
from tests.fixtures.pdfs import make_text_pdf

pytestmark = pytest.mark.database


def indexed_document(
    *,
    document_id: uuid.UUID | None = None,
    filename: str = "indexed.pdf",
    chunks: Sequence[tuple[str, Sequence[float]]] = (),
    indexed: bool = True,
) -> Document:
    identifier = document_id or uuid.uuid4()
    page_content = "\n".join(content for content, _vector in chunks) or "page"
    page = DocumentPage(
        document_id=identifier,
        page_number=1,
        content=page_content,
        char_count=len(page_content),
    )
    page.chunks = [
        DocumentChunk(
            id=uuid.uuid4(),
            document_id=identifier,
            page_number=1,
            chunk_index=index,
            content=content,
            token_count=1,
            embedding=embedding_to_pgvector(vector),
        )
        for index, (content, vector) in enumerate(chunks)
    ]
    return Document(
        id=identifier,
        filename=filename,
        media_type="application/pdf",
        size_bytes=100,
        content_hash=identifier.hex.ljust(64, "0"),
        page_count=1,
        extracted_char_count=len(page_content),
        embedding_model=EMBEDDING_MODEL if indexed else None,
        embedding_dimensions=EMBEDDING_DIMENSIONS if indexed else None,
        indexed_at=datetime.now(UTC) if indexed else None,
        pages=[page],
    )


def test_vector_schema_has_fixed_dimension_and_no_ann_index(database_session: Session) -> None:
    vector_type = database_session.scalar(
        text(
            """
            SELECT format_type(attribute.atttypid, attribute.atttypmod)
            FROM pg_attribute AS attribute
            JOIN pg_class AS relation ON relation.oid = attribute.attrelid
            WHERE relation.relname = 'document_chunks'
              AND attribute.attname = 'embedding'
            """
        )
    )
    assert vector_type == "vector(1536)"
    indexes = inspect(database_session.get_bind()).get_indexes("document_chunks")
    assert {index["name"] for index in indexes} == {"uq_document_chunks_document_index"}


def test_valid_vector_and_index_metadata_are_persisted(database_session: Session) -> None:
    document = indexed_document(chunks=[("chunk", controlled_vector(1.0))])
    database_session.add(document)
    database_session.commit()
    database_session.refresh(document)
    assert document.indexed_at is not None
    assert document.embedding_model == EMBEDDING_MODEL
    assert document.embedding_dimensions == EMBEDDING_DIMENSIONS
    assert database_session.scalar(select(func.count(DocumentChunk.id))) == 1


def test_wrong_vector_dimension_is_rejected(database_session: Session) -> None:
    document = indexed_document(chunks=[])
    database_session.add(document)
    database_session.commit()
    database_session.add(
        DocumentChunk(
            id=uuid.uuid4(),
            document_id=document.id,
            page_number=1,
            chunk_index=0,
            content="chunk",
            token_count=1,
            embedding="[1,2]",
        )
    )
    with pytest.raises(DataError):
        database_session.commit()
    database_session.rollback()


def test_document_chunk_index_is_unique(database_session: Session) -> None:
    document = indexed_document(chunks=[("first", controlled_vector(1.0))])
    duplicate = DocumentChunk(
        id=uuid.uuid4(),
        document_id=document.id,
        page_number=1,
        chunk_index=0,
        content="duplicate",
        token_count=1,
        embedding=embedding_to_pgvector(controlled_vector(0.0, 1.0)),
    )
    document.pages[0].chunks.append(duplicate)
    database_session.add(document)
    with pytest.raises(IntegrityError):
        database_session.commit()
    database_session.rollback()


def test_chunk_page_must_exist_for_same_document(database_session: Session) -> None:
    document = indexed_document(chunks=[])
    database_session.add(document)
    database_session.commit()
    database_session.add(
        DocumentChunk(
            id=uuid.uuid4(),
            document_id=document.id,
            page_number=2,
            chunk_index=0,
            content="orphan provenance",
            token_count=2,
            embedding=embedding_to_pgvector(controlled_vector(1.0)),
        )
    )
    with pytest.raises(IntegrityError):
        database_session.commit()
    database_session.rollback()


def test_delete_cascades_to_chunks(database_session: Session) -> None:
    document = indexed_document(chunks=[("chunk", controlled_vector(1.0))])
    database_session.add(document)
    database_session.commit()
    database_session.execute(delete(Document).where(Document.id == document.id))
    database_session.commit()
    assert database_session.scalar(select(func.count(DocumentChunk.id))) == 0


def test_exact_cosine_top_four_and_deterministic_tie(database_session: Session) -> None:
    first_id = uuid.UUID("00000000-0000-0000-0000-000000000001")
    second_id = uuid.UUID("00000000-0000-0000-0000-000000000002")
    tied_chunks = [(f"first-{index}", controlled_vector(1.0)) for index in range(3)]
    other_chunks = [(f"second-{index}", controlled_vector(1.0)) for index in range(3)]
    database_session.add_all(
        [
            indexed_document(document_id=first_id, chunks=tied_chunks),
            indexed_document(document_id=second_id, chunks=other_chunks),
        ]
    )
    database_session.commit()
    provider = FakeEmbeddingProvider(vectors={"question": controlled_vector(1.0)})

    results = retrieve_chunks(
        database_session,
        question="question",
        document_ids=[second_id, first_id],
        embedding_provider=provider,
    )

    assert [(result.document_id, result.chunk_index) for result in results] == [
        (first_id, 0),
        (first_id, 1),
        (first_id, 2),
        (second_id, 0),
    ]
    assert [result.rank for result in results] == [1, 2, 3, 4]
    assert all(result.similarity == pytest.approx(1.0) for result in results)
    assert provider.calls == [["question"]]


def test_retrieval_filters_exactly_requested_document_ids(database_session: Session) -> None:
    selected = indexed_document(chunks=[("selected", controlled_vector(0.0, 1.0))])
    excluded = indexed_document(chunks=[("excluded", controlled_vector(1.0))])
    database_session.add_all([selected, excluded])
    database_session.commit()
    provider = FakeEmbeddingProvider(vectors={"question": controlled_vector(1.0)})

    results = retrieve_chunks(
        database_session,
        question="question",
        document_ids=[selected.id],
        embedding_provider=provider,
    )
    assert [result.content for result in results] == ["selected"]


def test_partial_invalid_document_selection_fails_explicitly(database_session: Session) -> None:
    indexed = indexed_document(chunks=[("chunk", controlled_vector(1.0))])
    database_session.add(indexed)
    database_session.commit()
    missing_id = uuid.uuid4()
    provider = FakeEmbeddingProvider()

    with pytest.raises(DocumentsNotSearchableError) as captured:
        retrieve_chunks(
            database_session,
            question="question",
            document_ids=[indexed.id, missing_id],
            embedding_provider=provider,
        )
    assert captured.value.missing == (missing_id,)
    assert provider.calls == []


def test_ingested_document_is_not_searchable(database_session: Session) -> None:
    ingested = indexed_document(indexed=False)
    database_session.add(ingested)
    database_session.commit()
    with pytest.raises(DocumentsNotSearchableError) as captured:
        retrieve_chunks(
            database_session,
            question="question",
            document_ids=[ingested.id],
            embedding_provider=FakeEmbeddingProvider(),
        )
    assert captured.value.ingested == (ingested.id,)


def test_indexed_document_without_chunks_returns_empty_list(database_session: Session) -> None:
    document = indexed_document(chunks=[])
    database_session.add(document)
    database_session.commit()
    provider = FakeEmbeddingProvider(vectors={"question": controlled_vector(1.0)})
    assert (
        retrieve_chunks(
            database_session,
            question="question",
            document_ids=[document.id],
            embedding_provider=provider,
        )
        == []
    )
    assert provider.calls == [["question"]]


@pytest.mark.parametrize(
    ("question", "document_ids"),
    [
        ("", [uuid.uuid4()]),
        ("x" * 1_001, [uuid.uuid4()]),
        ("question", []),
        ("question", [uuid.uuid4()] * 2),
        ("question", [uuid.uuid4() for _index in range(11)]),
    ],
)
def test_retrieval_request_validation(question: str, document_ids: list[uuid.UUID]) -> None:
    with pytest.raises(InvalidRetrievalRequestError):
        retrieve_chunks(
            Session(),
            question=question,
            document_ids=document_ids,
            embedding_provider=FakeEmbeddingProvider(),
        )


def test_full_m2_pipeline_with_fake_embeddings_and_postgres(database_session: Session) -> None:
    vectors = {
        "Alpha": controlled_vector(0.0, 1.0),
        "Beta": controlled_vector(1.0, 0.0),
        "Gamma": controlled_vector(0.8, 0.6),
        "Which chunk?": controlled_vector(1.0, 0.0),
    }
    provider = FakeEmbeddingProvider(vectors=vectors)
    document = ingest_document(
        database_session,
        filename="pipeline.pdf",
        media_type="application/pdf",
        file_bytes=make_text_pdf(["Alpha", "Beta", "Gamma"]),
        embedding_provider=provider,
    )
    assert document.indexed_at is not None
    assert len(document.pages) == 3
    assert len(document.chunks) == 3

    results = retrieve_chunks(
        database_session,
        question="Which chunk?",
        document_ids=[document.id],
        embedding_provider=provider,
    )
    assert [result.content for result in results] == ["Beta", "Gamma", "Alpha"]
    assert [result.page_number for result in results] == [2, 3, 1]
    assert [result.rank for result in results] == [1, 2, 3]
    assert provider.calls == [["Alpha", "Beta", "Gamma"], ["Which chunk?"]]
