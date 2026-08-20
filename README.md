# DocIntel Lite

## Problem

Document answers are useful only when their evidence can be inspected. A fluent answer without
traceable source text can hide retrieval mistakes, unsupported claims, or insufficient context.
DocIntel Lite keeps page provenance through ingestion and returns the exact retrieved chunks used
for generation so callers can verify the evidence behind an answer.

## What the project does

DocIntel Lite v1.0 is a synchronous API for evidence-grounded question answering over selected
textual PDF documents:

```text
PDF
  -> extraction
  -> page-aware chunking
  -> embeddings
  -> pgvector retrieval
  -> grounded generation
  -> validated sources
```

The original PDF binary is never stored. Each page is retained as a separate row, including empty
pages, so stored page numbers continue to match the source document.

## Architecture

The application is a modular FastAPI monolith with synchronous provider and database boundaries.
PostgreSQL is the system of record; pgvector performs exact cosine retrieval. Ingestion finishes
all external embedding work before atomically persisting the document, its pages, and its chunks.
Query generation is stateless, receives no tools, and is validated against application-owned
source IDs. See [ARCHITECTURE.md](ARCHITECTURE.md) for components, invariants, and trade-offs.

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

Set `OPENAI_API_KEY` in `.env` before performing a real upload or query. Health, readiness, metadata
listing, migrations, and deterministic tests/evaluation do not require a key. The v1 model settings
accept only `text-embedding-3-small` and `gpt-5.6-terra`.

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
- `OPENAI_API_KEY`: required when real upload/query operations call OpenAI.
- `OPENAI_EMBEDDING_MODEL`: fixed to `text-embedding-3-small` for v1 compatibility.
- `OPENAI_GENERATION_MODEL`: fixed to `gpt-5.6-terra` for v1 compatibility.

Embedding dimensions are a schema invariant fixed in code at 1536, not environment configuration.

## v1.0 limits

- Textual PDF only; no OCR.
- Maximum upload size: 10 MiB.
- Maximum pages: 100.
- Maximum normalized text: 500,000 characters.
- Maximum chunks per document: 250.
- Chunk size/overlap: 600/100 `cl100k_base` tokens, bounded to a page.
- Retrieval: exact cosine search with `top_k=4`, without a similarity threshold or ANN index.

## API

| Method | Path | Behavior |
| --- | --- | --- |
| `GET` | `/healthz` | Process liveness only |
| `GET` | `/readyz` | PostgreSQL connectivity and `vector` extension readiness |
| `POST` | `/api/v1/documents` | Ingest and synchronously index one textual PDF from multipart field `file` |
| `GET` | `/api/v1/documents` | List document metadata in deterministic order |
| `DELETE` | `/api/v1/documents/{document_id}` | Delete a document and cascade to pages and chunks |
| `POST` | `/api/v1/query` | Retrieve and generate one grounded, stateless answer |

The list response never contains page or chunk text. A byte-identical upload is rejected with
`409 Conflict` before embedding work when detected by the pre-check. A successful new upload reports
`INDEXED`, `text-embedding-3-small`, 1536 dimensions, and a non-null `indexed_at`.

Documents created under M1 remain `INGESTED` and are excluded from retrieval; M2 performs no
automatic backfill. In development, delete and upload again to index one of those documents.

Upload and synchronously index one PDF:

```bash
curl --fail-with-body \
  --form 'file=@./document.pdf;type=application/pdf' \
  http://127.0.0.1:8000/api/v1/documents
```

A query accepts only a 1–1000 character question and 1–10 unique indexed document UUIDs:

```bash
curl --fail-with-body \
  --header 'Content-Type: application/json' \
  --data '{
    "question": "What is the production request timeout?",
    "document_ids": ["07a4273c-b3e3-48ca-87cc-b38cb8f25939"]
  }' \
  http://127.0.0.1:8000/api/v1/query
```

A grounded response returns all and only the chunks sent to generation, in retrieval order. The
model-declared `citation_ids` is the subset referenced in the answer:

```json
{
  "question": "What is the production request timeout?",
  "answer": "The production request timeout is 30 seconds [S1].",
  "abstained": false,
  "citation_ids": ["S1"],
  "sources": [
    {
      "source_id": "S1",
      "document_id": "07a4273c-b3e3-48ca-87cc-b38cb8f25939",
      "filename": "operations.pdf",
      "page_number": 6,
      "chunk_id": "f25c391c-17a7-4747-b02e-773461293578",
      "chunk_index": 5,
      "text": "The production request timeout is 30 seconds.",
      "similarity": 0.86
    }
  ],
  "retrieval": {"top_k": 4, "chunks_returned": 1}
}
```

Insufficient evidence returns HTTP 200 with `abstained=true`, no citations, and the stable answer.
Retrieved chunks remain visible for inspection even though none is claimed as supporting an answer:

```json
{
  "question": "What is the annual support budget?",
  "answer": "Insufficient evidence in the selected documents to answer the question.",
  "abstained": true,
  "citation_ids": [],
  "sources": [
    {
      "source_id": "S1",
      "document_id": "07a4273c-b3e3-48ca-87cc-b38cb8f25939",
      "filename": "operations.pdf",
      "page_number": 2,
      "chunk_id": "03a956a2-80a8-4bc4-92a7-4d4f256f3cc0",
      "chunk_index": 1,
      "text": "Support requests are routed through the operations queue.",
      "similarity": 0.31
    }
  ],
  "retrieval": {"top_k": 4, "chunks_returned": 1}
}
```

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

There is no ANN, HNSW, IVFFlat, hybrid search, reranking, threshold, or fallback. Every requested
document must exist and already be indexed; partially invalid selections fail instead of silently
searching a subset.

## Generation, grounding, and sources

The generation boundary is synchronous and uses the OpenAI Responses API with Structured Outputs.
It is fixed to `gpt-5.6-terra`, `reasoning.effort=none`, `max_output_tokens=700`, `store=false`, a
30-second timeout, and zero automatic retries. It sets no temperature and supplies no tools,
conversation, previous response, background processing, streaming, or external search.

Prompt `v1` keeps static application instructions separate from a JSON user-data input containing
only the question and retrieved source IDs/text. The application, never the model, assigns S1..Sn
by retrieval rank and retains document, filename, page, chunk, and similarity metadata. Document
text and the question remain untrusted data; embedded instructions cannot enable tools or alter
model/provider configuration.

Structured output contains only `answer`, `abstained`, and `citation_ids`. A non-abstained answer
must be non-empty, use at least one inline `[Sx]`, and match its unique `citation_ids` exactly. Every
ID must exist among retrieved chunks. An abstention must have empty model answer/citations and is
normalized locally. Provider safety refusal is a distinct integration error, not an abstention.

Source validation proves that cited IDs exist, were retrieved, were sent to generation, and were
referenced consistently. It does not mathematically prove that every generated statement is
semantically entailed by a cited chunk. Prompt constraints and controlled evaluation reduce risk;
they do not eliminate hallucination or prompt injection.

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
executed or used to fetch a URL. Generation receives no tool. The prompt explicitly treats question
and source content as data and ignores instructions found there. Provider output is schema-checked
and grounding-checked before it reaches the API response. Prompts, complete chunks, embeddings, raw
provider responses, API keys, and stack traces are not logged by these paths.

## Privacy and cost boundaries

With real providers, upload sends chunk text to the OpenAI Embeddings API. Query sends the question
to the Embeddings API, then the question plus at most four retrieved chunk texts to the Responses
API. The original PDF binary is neither retained nor sent to OpenAI by the application. Generation
sets `store=false`, but applicable provider data policies still need review before sensitive
documents are used.

A real upload incurs embedding calls in batches of at most 32. A real query creates one question
embedding and at most one generation capped at 700 output tokens. Adapters and application add no
retry. Real calls can incur cost.

## Evaluation and quality gates

The integration and API suites require the real PostgreSQL/pgvector service and an upgraded schema:

```bash
uv sync --frozen
uv run alembic upgrade head
uv run ruff check .
uv run ruff format --check .
uv run mypy src tests
uv run pytest
uv run alembic check
uv run python -m evals.run --mode deterministic
```

GitHub Actions runs these gates with the same PostgreSQL/pgvector image and requires no external
secrets. API/RAG tests inject deterministic 1536-dimensional embeddings and structured generation,
so CI never calls OpenAI. SQLite is not used as a database substitute.

The controlled evaluation dataset has exactly eight cases: three directly answerable, one
multi-chunk, two unanswerable, one production/staging retrieval trap, and one injection-like
document. Its synthetic two-PDF corpus is generated from versioned page fixtures. Deterministic mode
uses real extraction, chunking, PostgreSQL/pgvector retrieval, source mapping, grounding validation,
and controlled provider doubles. It checks Evidence Recall@4, grounded success, abstention,
citation validity, the trap, and the adversarial contract. This mode validates plumbing and
invariants; it does not measure real OpenAI model quality.

The final deterministic v1.0 gate reports:

- dataset: 8 cases;
- Evidence Recall@4: 1.00 (7/7 gold evidence items retrieved);
- grounded answerable cases: 6/6;
- unanswerable abstention: 2/2;
- citation validity: 100%, with zero invalid source IDs;
- production/staging retrieval trap: PASS;
- injection-like case: PASS (`BRAVO-17` retained and `COMPROMISED` absent).

Optional live evaluation uses the same corpus and dataset and can incur cost. It is intentionally
manual and outside the public CI; it is not a v1.0 release gate:

```bash
uv run python -m evals.run --mode live
```

## Limitations

- Textual PDFs only; scanned/image-only documents require OCR and are rejected.
- Grounding validation proves source identity and citation consistency, not formal semantic
  entailment of every generated statement.
- Hallucination and document prompt-injection risks are mitigated, not eliminated.
- There is no authentication or tenant isolation.
- Exact retrieval and synchronous ingestion target a small corpus, not high-throughput workloads.

## Outside v1.0

OCR, additional document formats, authentication, asynchronous workers, object storage, hybrid or
approximate retrieval, reranking, agents, a frontend, and multimodal processing are not included.
