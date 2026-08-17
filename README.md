# DocIntel Lite

DocIntel Lite is a small document-intelligence backend. Its current milestone validates textual
PDFs, extracts and normalizes page text, builds deterministic chunks, generates embeddings, and
persists an exact-search index in PostgreSQL/pgvector.

## Current status

M1 document ingestion and M2 vector indexing/retrieval are implemented. Retrieval is an internal
service with no public query endpoint. Generation, RAG, grounding, source mapping, abstention, and
evaluation are not implemented.

The application is a synchronous modular monolith:

```text
multipart PDF
  -> bounded in-memory read and validation
  -> pypdf page extraction
  -> deterministic text normalization
  -> page-bounded token chunking
  -> OpenAI embedding provider boundary
  -> atomic Document + DocumentPage + DocumentChunk persistence

question + selected document IDs
  -> one query embedding
  -> exact PostgreSQL cosine search
  -> top four typed chunk results with page provenance
```

The original PDF binary is never stored. Each page is retained as a separate row, including empty
pages, so stored page numbers continue to match the source document.

## Stack

- Python 3.14.7 and uv
- FastAPI 0.140.6, Pydantic 2.13.4, and Pydantic Settings
- SQLAlchemy 2.0.52 with synchronous Psycopg 3 sessions
- Alembic 1.19.1
- PostgreSQL 18.6 and pgvector 0.8.6
- pypdf 6.16.1, OpenAI SDK 3.1.0, and tiktoken 0.13.0
- pytest 9.1.1, HTTPX 0.28.1, Ruff 0.16.3, and mypy 2.3.1

## Setup

Prerequisites are Docker with Compose v2 and `uv`. From the repository root:

```bash
cp .env.example .env
docker compose up -d
uv sync --frozen
uv run alembic upgrade head
```

Set `OPENAI_API_KEY` in `.env` before performing a real upload. Health, readiness, metadata listing,
migrations, and the deterministic test suite do not require a key. `OPENAI_EMBEDDING_MODEL` accepts
only `text-embedding-3-small` in v1.

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
- `OPENAI_API_KEY`: required only when a real upload needs embeddings.
- `OPENAI_EMBEDDING_MODEL`: fixed to `text-embedding-3-small` for v1 compatibility.

Embedding dimensions are a schema invariant fixed in code at 1536, not environment configuration.

## API

| Method | Path | Behavior |
| --- | --- | --- |
| `GET` | `/healthz` | Process liveness only |
| `GET` | `/readyz` | PostgreSQL connectivity and `vector` extension readiness |
| `POST` | `/api/v1/documents` | Ingest and synchronously index one textual PDF from multipart field `file` |
| `GET` | `/api/v1/documents` | List document metadata in deterministic order |
| `DELETE` | `/api/v1/documents/{document_id}` | Delete a document and cascade to pages and chunks |

The list response never contains page or chunk text. A byte-identical upload is rejected with
`409 Conflict` before embedding work when detected by the pre-check. A successful new upload reports
`INDEXED`, `text-embedding-3-small`, 1536 dimensions, and a non-null `indexed_at`.

Documents created under M1 remain `INGESTED` and are excluded from retrieval; M2 performs no
automatic backfill. In development, delete and upload again to index one of those documents.

## Chunking and embeddings

Chunking uses `cl100k_base` and is deterministic, page-bounded, paragraph-aware, and token-aware.
Each chunk contains at most 600 tokens. Consecutive chunks on the same page repeat at most the prior
100 tokens as context, never cross a page, and receive a stable document-global zero-based index.
Empty pages produce no chunks. A document may produce at most 250 chunks.

The synchronous OpenAI adapter uses `text-embedding-3-small` at its fixed 1536 dimensions. It sends
at most 32 chunk texts per request, so 250 chunks require at most eight embedding calls. Automatic
SDK retries are disabled and each request has a 30-second timeout. All responses are checked for
count, index, order, dimension, and finite numeric values before persistence.

When the real provider is used, the text of each chunk is sent to OpenAI and those calls can incur
cost. The original PDF binary is not sent or retained. Provider calls finish before the final
database transaction begins; a failed batch persists no document, page, or chunk.

## Retrieval

The internal retrieval service accepts one 1–1000 character question and 1–10 unique document IDs.
It embeds the question once with the same model and dimension, then uses pgvector's exact cosine
distance operator. Results are ordered by distance, document ID, and chunk index and limited to the
top four. Similarity is `1 - cosine_distance`.

There is no ANN, HNSW, IVFFlat, hybrid search, reranking, fallback, or public retrieval endpoint.
Every requested document must exist and already be indexed; partially invalid selections fail
instead of silently searching a subset.

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
all pages and chunks commit in one transaction, so a failed indexing operation leaves no partial
document. Chunk text remains untrusted data: it cannot alter provider configuration and is never
executed, interpreted as instructions, or used to fetch a URL.

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
secrets. API and retrieval tests inject deterministic 1536-dimensional fake embeddings, so CI never
calls OpenAI. SQLite is not used as a database substitute.

See [ARCHITECTURE.md](ARCHITECTURE.md) for implemented and planned architectural decisions.
