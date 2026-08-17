from dataclasses import dataclass

from docintel.documents.extraction import (
    DocumentIngestionError,
    DocumentLimitsExceededError,
    DocumentTextNotExtractableError,
    ExtractedPage,
)
from docintel.indexing.chunking import MAX_CHUNKS_PER_DOCUMENT, ChunkDraft, chunk_pages
from docintel.providers.embeddings import (
    EMBEDDING_DIMENSIONS,
    EMBEDDING_MODEL,
    EmbeddingProvider,
    EmbeddingProviderFailure,
    EmbeddingProviderUnconfigured,
    validate_embedding_vectors,
)


class EmbeddingProviderApiError(DocumentIngestionError):
    status_code = 502
    code = "embedding_provider_error"
    message = "The embedding provider could not index the document."


class EmbeddingProviderUnconfiguredApiError(DocumentIngestionError):
    status_code = 503
    code = "embedding_provider_unconfigured"
    message = "The embedding provider is not configured."


@dataclass(frozen=True, slots=True)
class IndexedChunkDraft:
    chunk: ChunkDraft
    embedding: list[float]


def prepare_index(
    pages: list[ExtractedPage],
    provider: EmbeddingProvider,
) -> list[IndexedChunkDraft]:
    chunks = chunk_pages(pages)
    if len(chunks) > MAX_CHUNKS_PER_DOCUMENT:
        raise DocumentLimitsExceededError
    if not chunks:
        raise DocumentTextNotExtractableError

    if provider.model != EMBEDDING_MODEL or provider.dimensions != EMBEDDING_DIMENSIONS:
        raise EmbeddingProviderApiError

    try:
        vectors = provider.embed([chunk.content for chunk in chunks])
        validated_vectors = validate_embedding_vectors(vectors, expected_count=len(chunks))
    except EmbeddingProviderUnconfigured as error:
        raise EmbeddingProviderUnconfiguredApiError from error
    except EmbeddingProviderFailure as error:
        raise EmbeddingProviderApiError from error

    return [
        IndexedChunkDraft(chunk=chunk, embedding=embedding)
        for chunk, embedding in zip(chunks, validated_vectors, strict=True)
    ]
