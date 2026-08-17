import uuid
from datetime import datetime

import pytest
from sqlalchemy import delete, func, inspect, select, text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from docintel.db.models import Document, DocumentPage

pytestmark = pytest.mark.database


def make_document(*, content_hash: str | None = None) -> Document:
    identifier = uuid.uuid4()
    return Document(
        id=identifier,
        filename="fixture.pdf",
        media_type="application/pdf",
        size_bytes=100,
        content_hash=content_hash or identifier.hex.ljust(64, "0"),
        page_count=2,
        extracted_char_count=4,
        embedding_model=None,
        embedding_dimensions=None,
        indexed_at=None,
        pages=[
            DocumentPage(
                document_id=identifier,
                page_number=1,
                content="text",
                char_count=4,
            ),
            DocumentPage(
                document_id=identifier,
                page_number=2,
                content="",
                char_count=0,
            ),
        ],
    )


def test_initial_migration_schema_and_vector_extension(database_session: Session) -> None:
    bind = database_session.get_bind()
    inspector = inspect(bind)
    assert {"alembic_version", "documents", "document_pages"} <= set(inspector.get_table_names())
    assert database_session.scalar(text("SELECT version()")) is not None
    assert database_session.scalar(text("SHOW server_version_num")) == "180006"
    assert (
        database_session.scalar(
            text("SELECT extversion FROM pg_extension WHERE extname = 'vector'")
        )
        == "0.8.6"
    )
    assert database_session.scalar(text("SELECT version_num FROM alembic_version")) == (
        "0001_document_ingestion"
    )

    columns = {column["name"] for column in inspector.get_columns("documents")}
    assert "binary_content" not in columns
    assert "document_chunks" not in inspector.get_table_names()
    primary_key = inspector.get_pk_constraint("document_pages")
    assert primary_key["constrained_columns"] == ["document_id", "page_number"]
    foreign_key = inspector.get_foreign_keys("document_pages")[0]
    assert foreign_key["options"]["ondelete"] == "CASCADE"


def test_document_and_pages_are_inserted(database_session: Session) -> None:
    document = make_document()
    database_session.add(document)
    database_session.commit()

    stored = database_session.get(Document, document.id)
    assert stored is not None
    assert [(page.page_number, page.content) for page in stored.pages] == [
        (1, "text"),
        (2, ""),
    ]


def test_content_hash_is_unique(database_session: Session) -> None:
    shared_hash = "a" * 64
    database_session.add(make_document(content_hash=shared_hash))
    database_session.commit()
    database_session.add(make_document(content_hash=shared_hash))
    with pytest.raises(IntegrityError):
        database_session.commit()
    database_session.rollback()
    assert database_session.scalar(select(func.count(Document.id))) == 1


def test_database_cascade_deletes_pages(database_session: Session) -> None:
    document = make_document()
    database_session.add(document)
    database_session.commit()
    database_session.execute(delete(Document).where(Document.id == document.id))
    database_session.commit()
    assert database_session.scalar(select(func.count(DocumentPage.document_id))) == 0


def test_failed_transaction_rolls_back_document_and_pages(database_session: Session) -> None:
    document = make_document()
    document.pages[1].char_count = -1
    database_session.add(document)
    with pytest.raises(IntegrityError):
        database_session.commit()
    database_session.rollback()
    assert database_session.scalar(select(func.count(Document.id))) == 0
    assert database_session.scalar(select(func.count(DocumentPage.document_id))) == 0


def test_created_at_is_timezone_aware(database_session: Session) -> None:
    document = make_document()
    database_session.add(document)
    database_session.commit()
    database_session.refresh(document)
    assert isinstance(document.created_at, datetime)
    assert document.created_at.tzinfo is not None
