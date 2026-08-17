"""Create the document ingestion schema.

Revision ID: 0001_document_ingestion
Revises:
Create Date: 2026-08-17
"""

from collections.abc import Sequence

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

revision: str = "0001_document_ingestion"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # M1 validates pgvector availability but intentionally creates no vector columns.
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")
    op.create_table(
        "documents",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("filename", sa.String(length=255), nullable=False),
        sa.Column("media_type", sa.String(length=64), nullable=False),
        sa.Column("size_bytes", sa.BigInteger(), nullable=False),
        sa.Column("content_hash", sa.String(length=64), nullable=False),
        sa.Column("page_count", sa.Integer(), nullable=False),
        sa.Column("extracted_char_count", sa.Integer(), nullable=False),
        sa.Column("embedding_model", sa.String(), nullable=True),
        sa.Column("embedding_dimensions", sa.Integer(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column("indexed_at", sa.DateTime(timezone=True), nullable=True),
        sa.CheckConstraint(
            "extracted_char_count > 0",
            name="extracted_char_count_positive",
        ),
        sa.CheckConstraint("page_count > 0", name="page_count_positive"),
        sa.CheckConstraint("size_bytes > 0", name="size_bytes_positive"),
        sa.PrimaryKeyConstraint("id", name="pk_documents"),
        sa.UniqueConstraint("content_hash", name="uq_documents_content_hash"),
    )
    op.create_table(
        "document_pages",
        sa.Column("document_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("page_number", sa.Integer(), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("char_count", sa.Integer(), nullable=False),
        sa.CheckConstraint(
            "char_count >= 0",
            name="char_count_nonnegative",
        ),
        sa.CheckConstraint(
            "page_number >= 1",
            name="page_number_positive",
        ),
        sa.ForeignKeyConstraint(
            ["document_id"],
            ["documents.id"],
            name="fk_document_pages_document_id_documents",
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint(
            "document_id",
            "page_number",
            name="pk_document_pages",
        ),
    )


def downgrade() -> None:
    op.drop_table("document_pages")
    op.drop_table("documents")
    op.execute("DROP EXTENSION IF EXISTS vector")
