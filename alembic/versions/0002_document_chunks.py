"""Create vector-backed document chunks.

Revision ID: 0002_document_chunks
Revises: 0001_document_ingestion
Create Date: 2026-08-17
"""

from collections.abc import Sequence

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

revision: str = "0002_document_chunks"
down_revision: str | None = "0001_document_ingestion"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


class Vector1536(sa.types.UserDefinedType[str]):
    cache_ok = True

    def get_col_spec(self, **_kwargs: object) -> str:
        return "vector(1536)"


def upgrade() -> None:
    op.create_table(
        "document_chunks",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("document_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("page_number", sa.Integer(), nullable=False),
        sa.Column("chunk_index", sa.Integer(), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("token_count", sa.Integer(), nullable=False),
        sa.Column("embedding", Vector1536(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint("page_number >= 1", name="page_number_positive"),
        sa.CheckConstraint("chunk_index >= 0", name="chunk_index_nonnegative"),
        sa.CheckConstraint("token_count > 0", name="token_count_positive"),
        sa.CheckConstraint("token_count <= 600", name="token_count_maximum"),
        sa.CheckConstraint("length(content) > 0", name="content_nonempty"),
        sa.ForeignKeyConstraint(
            ["document_id"],
            ["documents.id"],
            name="fk_document_chunks_document_id_documents",
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["document_id", "page_number"],
            ["document_pages.document_id", "document_pages.page_number"],
            name="fk_document_chunks_document_page",
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name="pk_document_chunks"),
        sa.UniqueConstraint(
            "document_id",
            "chunk_index",
            name="uq_document_chunks_document_index",
        ),
    )


def downgrade() -> None:
    op.drop_table("document_chunks")
