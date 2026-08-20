from typing import Annotated

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from docintel.api.schemas import ErrorResponse, QueryRequest, QueryResponse
from docintel.config import get_settings
from docintel.db.session import get_db
from docintel.providers.embeddings import EmbeddingProvider, OpenAIEmbeddingProvider
from docintel.providers.generation import GenerationProvider, OpenAIGenerationProvider
from docintel.rag.service import run_rag_query

router = APIRouter(prefix="/api/v1", tags=["query"])
DatabaseSession = Annotated[Session, Depends(get_db)]


def get_query_embedding_provider() -> EmbeddingProvider:
    settings = get_settings()
    return OpenAIEmbeddingProvider(api_key=settings.openai_api_key)


def get_generation_provider() -> GenerationProvider:
    settings = get_settings()
    return OpenAIGenerationProvider(
        api_key=settings.openai_api_key,
        model=settings.openai_generation_model,
    )


EmbeddingProviderDependency = Annotated[
    EmbeddingProvider,
    Depends(get_query_embedding_provider),
]
GenerationProviderDependency = Annotated[
    GenerationProvider,
    Depends(get_generation_provider),
]


@router.post(
    "/query",
    response_model=QueryResponse,
    responses={
        404: {"model": ErrorResponse},
        409: {"model": ErrorResponse},
        502: {"model": ErrorResponse},
        503: {"model": ErrorResponse},
    },
)
def query_documents(
    request: QueryRequest,
    session: DatabaseSession,
    embedding_provider: EmbeddingProviderDependency,
    generation_provider: GenerationProviderDependency,
) -> QueryResponse:
    result = run_rag_query(
        session,
        question=request.question,
        document_ids=request.document_ids,
        embedding_provider=embedding_provider,
        generation_provider=generation_provider,
    )
    return QueryResponse.from_result(result)
