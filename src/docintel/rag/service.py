import uuid
from collections.abc import Sequence
from dataclasses import dataclass

from sqlalchemy.orm import Session

from docintel.providers.embeddings import EMBEDDING_DIMENSIONS, EMBEDDING_MODEL, EmbeddingProvider
from docintel.providers.generation import (
    GENERATION_MODEL,
    GenerationProvider,
    GenerationProviderFailure,
    GenerationProviderRefusal,
    GenerationProviderUnconfigured,
)
from docintel.rag.prompt import GROUNDING_INSTRUCTIONS, build_generation_input
from docintel.rag.sources import (
    CANONICAL_ABSTENTION_ANSWER,
    GroundedSource,
    SourceValidationError,
    map_retrieval_sources,
    validate_grounded_output,
)
from docintel.retrieval.service import (
    RETRIEVAL_TOP_K,
    DocumentsNotSearchableError,
    RetrievalError,
    RetrievalProviderError,
    RetrievalProviderUnconfigured,
    retrieve_chunks,
    validate_retrieval_request,
)


class RAGError(Exception):
    status_code = 500
    code = "query_failed"
    message = "The document query could not be completed."

    def response_body(self) -> dict[str, str]:
        return {"code": self.code, "message": self.message}


class DocumentsNotFoundError(RAGError):
    status_code = 404
    code = "documents_not_found"
    message = "One or more selected documents do not exist."


class DocumentsNotIndexedError(RAGError):
    status_code = 409
    code = "documents_not_indexed"
    message = "One or more selected documents are not indexed with the v1 model."


class EmbeddingProviderUnconfiguredQueryError(RAGError):
    status_code = 503
    code = "embedding_provider_unconfigured"
    message = "The embedding provider is not configured."


class GenerationProviderUnconfiguredQueryError(RAGError):
    status_code = 503
    code = "generation_provider_unconfigured"
    message = "The generation provider is not configured."


class EmbeddingProviderQueryError(RAGError):
    status_code = 502
    code = "embedding_provider_error"
    message = "The embedding provider could not process the question."


class GenerationProviderQueryError(RAGError):
    status_code = 502
    code = "generation_provider_error"
    message = "The generation provider could not answer the question."


class GenerationProviderRefusalQueryError(RAGError):
    status_code = 502
    code = "generation_provider_refusal"
    message = "The generation provider refused the request."


class InvalidGroundedResponseError(RAGError):
    status_code = 502
    code = "invalid_grounded_response"
    message = "The generation provider returned an invalid grounded response."


@dataclass(frozen=True, slots=True)
class RetrievalMetadata:
    top_k: int
    chunks_returned: int


@dataclass(frozen=True, slots=True)
class RAGResult:
    question: str
    answer: str
    abstained: bool
    citation_ids: tuple[str, ...]
    sources: tuple[GroundedSource, ...]
    retrieval: RetrievalMetadata


def run_rag_query(
    session: Session,
    *,
    question: str,
    document_ids: Sequence[uuid.UUID],
    embedding_provider: EmbeddingProvider,
    generation_provider: GenerationProvider,
) -> RAGResult:
    try:
        normalized_question, requested_ids = validate_retrieval_request(
            session,
            question=question,
            document_ids=document_ids,
        )
    except DocumentsNotSearchableError as error:
        if error.missing:
            raise DocumentsNotFoundError from error
        raise DocumentsNotIndexedError from error
    except RetrievalError as error:
        raise RAGError from error

    if not embedding_provider.is_configured:
        raise EmbeddingProviderUnconfiguredQueryError
    # Check generation configuration before paying for the query embedding.
    if not generation_provider.is_configured:
        raise GenerationProviderUnconfiguredQueryError
    if (
        embedding_provider.model != EMBEDDING_MODEL
        or embedding_provider.dimensions != EMBEDDING_DIMENSIONS
    ):
        raise EmbeddingProviderQueryError
    if generation_provider.model != GENERATION_MODEL:
        raise GenerationProviderQueryError

    try:
        retrieval_results = retrieve_chunks(
            session,
            question=normalized_question,
            document_ids=requested_ids,
            embedding_provider=embedding_provider,
        )
    except DocumentsNotSearchableError as error:
        if error.missing:
            raise DocumentsNotFoundError from error
        raise DocumentsNotIndexedError from error
    except RetrievalProviderUnconfigured as error:
        raise EmbeddingProviderUnconfiguredQueryError from error
    except RetrievalProviderError as error:
        raise EmbeddingProviderQueryError from error
    except RetrievalError as error:
        raise RAGError from error

    sources = map_retrieval_sources(retrieval_results)
    retrieval = RetrievalMetadata(
        top_k=RETRIEVAL_TOP_K,
        chunks_returned=len(sources),
    )
    if not sources:
        return RAGResult(
            question=normalized_question,
            answer=CANONICAL_ABSTENTION_ANSWER,
            abstained=True,
            citation_ids=(),
            sources=(),
            retrieval=retrieval,
        )

    generation_input = build_generation_input(normalized_question, sources)
    try:
        output = generation_provider.generate(
            instructions=GROUNDING_INSTRUCTIONS,
            input_text=generation_input,
        )
    except GenerationProviderUnconfigured as error:
        raise GenerationProviderUnconfiguredQueryError from error
    except GenerationProviderRefusal as error:
        raise GenerationProviderRefusalQueryError from error
    except GenerationProviderFailure as error:
        raise GenerationProviderQueryError from error

    try:
        validated = validate_grounded_output(output, sources)
    except SourceValidationError as error:
        raise InvalidGroundedResponseError from error
    return RAGResult(
        question=normalized_question,
        answer=validated.answer,
        abstained=validated.abstained,
        citation_ids=validated.citation_ids,
        sources=tuple(sources),
        retrieval=retrieval,
    )
