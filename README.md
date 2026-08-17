# DocIntel Lite

DocIntel Lite is a small document-intelligence backend. Its current milestone provides a
reproducible ingestion foundation for textual PDFs: validate an upload, extract and normalize
text page by page, and persist document metadata and pages atomically in PostgreSQL.

## Current status

M1 implements document ingestion only. RAG, indexing, chunking, embeddings, retrieval, grounded
answer generation, and evaluation are not implemented. The installed pgvector extension establishes
the database prerequisite for later indexing work; M1 creates no vector column or vector index.

The application is a synchronous modular monolith:

```text
multipart PDF
  -> bounded in-memory read and validation
  -> pypdf page extraction
  -> deterministic text normalization
  -> transactional Document + DocumentPage persistence
```

The original PDF binary is never stored. Each page is retained as a separate row, including empty
pages, so stored page numbers continue to match the source document.

## Stack

- Python 3.14.7 and uv
- FastAPI 0.140.6, Pydantic 2.13.4, and Pydantic Settings
- SQLAlchemy 2.0.52 with synchronous Psycopg 3 sessions
- Alembic 1.19.1
- PostgreSQL 18.6 and pgvector 0.8.6
- pypdf 6.16.1
- pytest 9.1.1, HTTPX 0.28.1, Ruff 0.16.3, and mypy 2.3.1

## Setup

Prerequisites are Docker with Compose v2 and `uv`. From the repository root:

```bash
cp .env.example .env
docker compose up -d
uv sync --frozen
uv run alembic upgrade head
```

The Compose file runs only `pgvector/pgvector:0.8.6-pg18-trixie`. Its development credentials are
local placeholders mirrored in `.env.example`; replace them outside local development and never
commit real credentials. If host port 5432 is occupied, set `DOCINTEL_DB_PORT` for Compose and use
the same port in `DATABASE_URL` (for example, `DOCINTEL_DB_PORT=55433 docker compose up -d`).
The image resolved during M1 validation to
`pgvector/pgvector@sha256:78bf48b801e792f99e3ac62b5036fd3876e9be48afda16c1e331af1c75ceb2ff`.

Verify the running database and extension:

```bash
docker compose exec db psql -U docintel -d docintel -c 'SELECT version();'
docker compose exec db psql -U docintel -d docintel \
  -c "SELECT extversion FROM pg_extension WHERE extname = 'vector';"
```

Apply migrations to any new database before starting the API:

```bash
uv run alembic upgrade head
uv run uvicorn docintel.app:app --host 127.0.0.1 --port 8000
```

Configuration is limited to:

- `DATABASE_URL`: SQLAlchemy URL using the `postgresql+psycopg` driver.
- `APP_ENV`: environment label; defaults to `development`.

No external API key is used by M1.

## M1 API

| Method | Path | Behavior |
| --- | --- | --- |
| `GET` | `/healthz` | Process liveness only |
| `GET` | `/readyz` | PostgreSQL connectivity and `vector` extension readiness |
| `POST` | `/api/v1/documents` | Ingest one textual PDF from multipart field `file` |
| `GET` | `/api/v1/documents` | List document metadata in deterministic order |
| `DELETE` | `/api/v1/documents/{document_id}` | Delete a document and cascade to its pages |

The list response never contains page text. A byte-identical upload is rejected with `409 Conflict`
and the existing document ID. Successfully ingested M1 documents report `INGESTED`; the `INDEXED`
status is reserved for later work and is derived from `indexed_at`.

## Ingestion limits and security boundaries

M1 accepts textual PDFs only, with all of these limits:

- media type `application/pdf` and a `.pdf` display filename;
- at most 10 MiB;
- at most 100 pages;
- at most 500,000 normalized characters;
- non-encrypted, structurally parseable content with at least one extracted character.

Filenames are untrusted display metadata: directory components and control characters are removed,
the result is Unicode-normalized and capped at 255 characters, and it is never used as a filesystem
path. The parser extracts text only; the application does not open URLs, follow links, execute
scripts, inspect attachments, run macros, or perform OCR. Uploads are processed through bounded
runtime-controlled storage and the original bytes are discarded after processing.

Exact SHA-256 over the received bytes backs database-enforced deduplication. Document metadata and
all pages commit in one transaction, so a failed ingestion leaves no orphan page.

## Tests and quality gates

The integration and API suites require the real PostgreSQL/pgvector service and an upgraded schema:

```bash
uv sync --frozen
uv run alembic upgrade head
uv run ruff check .
uv run ruff format --check .
uv run mypy src tests
uv run pytest
uv run alembic check
```

GitHub Actions runs these gates with the same PostgreSQL/pgvector image and requires no external
secrets. SQLite is not used as a database substitute.

See [ARCHITECTURE.md](ARCHITECTURE.md) for implemented and planned architectural decisions.
