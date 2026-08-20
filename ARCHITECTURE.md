# DocIntel Lite Architecture

## Purpose and boundaries

DocIntel Lite is a modular monolith for document-grounded question answering. This document records
the implemented v1.0 architecture and its deliberate boundaries.

## Milestone map

| Area | Decision | State |
| --- | --- | --- |
| Application shape | One modular FastAPI service with synchronous SQLAlchemy | Implemented in M1 |
| Source format | Textual PDF only; no OCR | Implemented in M1 |
| Binary retention | Do not persist the uploaded PDF | Implemented in M1 |
| Text model | Preserve normalized text page by page, including empty pages | Implemented in M1 |
| Persistence | PostgreSQL 18.6 with pgvector 0.8.6 enabled | Implemented in M1 |
| Chunking | Page-bounded 600-token chunks with up to 100-token overlap | Implemented in M2 |
| Embeddings | OpenAI `text-embedding-3-small`, 1536 dimensions, batches of 32 | Implemented in M2 |
| Retrieval | Exact cosine search with `top_k = 4` | Implemented in M2 |
| Answer model | `gpt-5.6-terra` through the Responses API | Implemented in M3 |
| Answer policy | Document-grounded answers with abstention when evidence is insufficient | Implemented in M3 |
| Citations | Source IDs created and controlled by the application | Implemented in M3 |
| Orchestration | No agents, tools, or function calling | Implemented constraint |
| Evaluation | Controlled deterministic harness and optional live path | Implemented in M3 |
| Hardening/release | Reproducibility, security review, and final release gates | Complete in M4 |

## Implemented in M1

The HTTP layer owns request validation and explicit response schemas. The document layer owns exact
hashing, PDF parsing, deterministic normalization, and transactional ingestion. The database layer
owns SQLAlchemy mappings and synchronous session lifecycle. Alembic is the sole schema-change path.

An upload is bounded before parsing. Its filename is sanitized as untrusted display metadata and is
never converted to a local path. pypdf extracts only textual page content. Unicode NFC, newline
canonicalization, inappropriate control-character removal, trailing line-whitespace trimming, and
blank-line bounding are deterministic. Empty pages are persisted to retain the PDF's real page
sequence, while a document whose aggregate normalized text is empty is rejected.

SHA-256 is computed over the original bytes. The unique database constraint is authoritative under
concurrent requests; both a pre-check and the constraint-race path return the existing document ID.
The binary input is not persisted. A document and all of its pages share one transaction, and the
foreign key cascades deletion.

The `vector` extension is installed now so the deployment prerequisite is observable through
readiness and integration tests. No chunk table, vector column, similarity query, or approximate
nearest-neighbor index exists in M1.

## Implemented in M2

The upload pipeline now completes indexing synchronously. Normalized pages are split independently
with `cl100k_base`; chunks never cross pages, contain at most 600 tokens, and may repeat up to the
last 100 tokens of the preceding chunk on that page. Paragraph boundaries are preferred when they
fit, large paragraphs use direct token windows, and `chunk_index` is stable and document-global.
The 250-chunk document limit is enforced before any provider call.

The embedding boundary is one small synchronous protocol with a production OpenAI adapter and a
deterministic fake used only through dependency injection in tests. The v1 adapter is fixed to
`text-embedding-3-small`, 1536 dimensions, and batches of 32. It sets a 30-second timeout, disables
automatic retries, omits the dimensions parameter, and validates returned quantity, indices, order,
dimension, and finite values. No API key is needed until a real embedding operation is attempted.

External calls finish before persistence begins. Document metadata, every page, and every validated
chunk then share one transaction. `document_chunks.embedding` is `vector(1536)`; a composite foreign
key ensures each chunk page exists for the same document. Existing unindexed M1 rows are neither
backfilled nor made searchable.

Retrieval is internal and has no HTTP endpoint. It validates the complete requested document set,
embeds the question once, filters to explicitly selected indexed documents, and executes exact
pgvector cosine distance ordered by distance, document ID, and chunk index. It returns at most four
typed results with document, filename, page, chunk, content, rank, and similarity provenance. There
is no ANN index, threshold, fallback, reranking, query rewriting, or source-ID assignment.

## Implemented in M3

The stateless query path is:

```text
strict document selection
  -> exact top-4 retrieval
  -> deterministic S1..Sn source mapping
  -> prompt v1 with untrusted JSON context
  -> structured Responses API generation
  -> abstention/citation validation
  -> answer plus retrieved-source metadata
```

Generation uses the synchronous OpenAI SDK boundary and `gpt-5.6-terra`. The request fixes
`reasoning.effort=none`, `max_output_tokens=700`, `store=false`, a 30-second timeout, and zero SDK
retries. It omits temperature, tools, conversation state, previous responses, streaming, and
background mode. The adapter parses a strict Pydantic Structured Output containing only answer,
abstention, and citation IDs. Provider refusal remains a 502 integration outcome and is never
relabelled as documentary abstention.

Static grounding instructions never contain question or source text. The user input is
deterministically serialized JSON with only question and source ID/text. S1..Sn follow retrieval
rank, and every source returned by the API is the exact retrieved chunk sent to generation. The
application owns filename, page, chunk, and similarity metadata; generation cannot supply or alter
it.

Post-generation validation rejects unknown or duplicate IDs, inline/list mismatches, uncited
non-abstained answers, factual abstentions, and abstentions with citations. Valid documentary
abstention is normalized to one stable phrase. Empty retrieval abstains locally without a generation
call. Questions and answers remain ephemeral; M3 adds no database table or migration.

The eight-case evaluation uses two synthetic textual PDFs. Deterministic/CI mode exercises real PDF
extraction, chunking, vector persistence, exact cosine retrieval, source mapping, and validation with
controlled provider doubles. Optional live mode uses the same corpus with OpenAI providers and is
manual/cost-bearing. No LLM judge or external evaluation framework is used.

Retrieval relevance, source validity, and semantic grounding are distinct. Exact retrieval and
Evidence Recall@4 measure whether gold evidence appears. Source validation proves that cited chunks
were retrieved and sent. It cannot prove semantic entailment for every sentence; prompting and
end-to-end evaluation mitigate but do not eliminate hallucination or prompt injection.

## Completed in M4

The v1.0 hardening gate freezes the public API, schema, models, chunking, retrieval, grounding, and
evaluation contracts described above. Installation is verified from the frozen `uv.lock`; Alembic
is verified both against the current database and from an empty PostgreSQL database through
`0001_document_ingestion` and `0002_document_chunks`. M4 adds no migration, table, provider,
endpoint, or retrieval algorithm.

The release checks cover the real PostgreSQL/pgvector integration, deterministic evaluation,
OpenAPI surface, sanitized error paths, ignored local secrets, and secret-free CI. Live OpenAI
evaluation remains an explicit manual operation and is not required for a reproducible release.

## Trade-offs

- **Exact search instead of ANN:** exact cosine ordering is deterministic and appropriate for the
  bounded v1 corpus. It avoids index tuning and recall trade-offs at the cost of poorer scaling.
- **Synchronous ingestion instead of workers:** the request has simple all-or-nothing semantics and
  no processing state. It also holds the client connection during provider calls and is unsuitable
  for high-throughput ingestion.
- **Direct RAG instead of a framework:** small provider, retrieval, and grounding boundaries keep
  data flow and failure behavior inspectable. More elaborate orchestration would require explicit
  justification in a later version.
- **Application-owned sources instead of model-owned citations:** the model can select only S1..Sn;
  document and page metadata always comes from retrieved database rows. This validates provenance
  identity, but it is not a proof of semantic entailment.

## Security posture

M1 deliberately narrows the input surface to bounded, non-encrypted textual PDFs. It does not
execute embedded content and does not retain the original binary. M3 treats both document and
question as untrusted data, separates them from application instructions, exposes no generation
tool, validates provider output, and keeps source metadata application-controlled. The original PDF
is not sent to OpenAI; real indexing sends chunk text, while real query sends the question to
embeddings and question plus up to four chunks to Responses. `store=false` is set, but provider data
policies still apply. Authentication, multi-user isolation, OCR, object storage, and cloud
deployment remain outside the v1.0 system boundary.
