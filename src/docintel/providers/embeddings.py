import math
from collections.abc import Sequence
from typing import Protocol, cast

from openai import OpenAI
from pydantic import SecretStr

EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIMENSIONS = 1536
EMBEDDING_BATCH_SIZE = 32
EMBEDDING_TIMEOUT_SECONDS = 30.0


class EmbeddingProvider(Protocol):
    model: str
    dimensions: int

    def embed(self, texts: Sequence[str]) -> list[list[float]]: ...


class EmbeddingProviderFailure(Exception):
    """A sanitized provider or response-contract failure."""


class EmbeddingProviderUnconfigured(EmbeddingProviderFailure):
    """The real provider was requested without credentials."""


class _EmbeddingDatum(Protocol):
    @property
    def index(self) -> int: ...

    @property
    def embedding(self) -> object: ...


class _EmbeddingResponse(Protocol):
    @property
    def data(self) -> Sequence[_EmbeddingDatum]: ...


class _EmbeddingsResource(Protocol):
    def create(
        self,
        *,
        input: Sequence[str],
        model: str,
        encoding_format: str,
    ) -> _EmbeddingResponse: ...


class _EmbeddingClient(Protocol):
    @property
    def embeddings(self) -> _EmbeddingsResource: ...


def validate_embedding_vectors(
    vectors: Sequence[Sequence[float]],
    *,
    expected_count: int,
) -> list[list[float]]:
    if len(vectors) != expected_count:
        raise EmbeddingProviderFailure("The provider returned an unexpected vector count.")

    validated: list[list[float]] = []
    for vector in vectors:
        if len(vector) != EMBEDDING_DIMENSIONS:
            raise EmbeddingProviderFailure("The provider returned an invalid vector dimension.")
        try:
            validated_vector = [float(component) for component in vector]
        except (OverflowError, TypeError, ValueError) as error:
            raise EmbeddingProviderFailure(
                "The provider returned a non-numeric vector component."
            ) from error
        if not all(math.isfinite(component) for component in validated_vector):
            raise EmbeddingProviderFailure("The provider returned a non-finite vector component.")
        validated.append(validated_vector)
    return validated


def embedding_to_pgvector(vector: Sequence[float]) -> str:
    validated = validate_embedding_vectors([vector], expected_count=1)[0]
    return f"[{','.join(format(component, '.17g') for component in validated)}]"


class OpenAIEmbeddingProvider:
    model = EMBEDDING_MODEL
    dimensions = EMBEDDING_DIMENSIONS

    def __init__(
        self,
        *,
        api_key: SecretStr | None,
        client: _EmbeddingClient | None = None,
    ) -> None:
        self._api_key = api_key
        self._client = client

    def _get_client(self) -> _EmbeddingClient:
        if self._client is not None:
            return self._client
        api_key = self._api_key.get_secret_value() if self._api_key is not None else ""
        if not api_key:
            raise EmbeddingProviderUnconfigured("The embedding provider is not configured.")
        try:
            sdk_client = OpenAI(
                api_key=api_key,
                max_retries=0,
                timeout=EMBEDDING_TIMEOUT_SECONDS,
            )
        except Exception as error:
            raise EmbeddingProviderFailure(
                "The embedding provider could not be initialized."
            ) from error
        self._client = cast(_EmbeddingClient, sdk_client)
        return self._client

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []

        vectors: list[list[float]] = []
        for offset in range(0, len(texts), EMBEDDING_BATCH_SIZE):
            batch = list(texts[offset : offset + EMBEDDING_BATCH_SIZE])
            try:
                response = self._get_client().embeddings.create(
                    input=batch,
                    model=self.model,
                    encoding_format="float",
                )
            except EmbeddingProviderFailure:
                raise
            except Exception as error:
                # Provider exception text can contain request payloads or credentials.
                raise EmbeddingProviderFailure("The embedding provider request failed.") from error

            try:
                data = list(response.data)
                if len(data) != len(batch):
                    raise EmbeddingProviderFailure(
                        "The provider returned an unexpected vector count."
                    )

                ordered: list[Sequence[float] | None] = [None] * len(batch)
                for item in data:
                    if (
                        not isinstance(item.index, int)
                        or isinstance(item.index, bool)
                        or not 0 <= item.index < len(batch)
                    ):
                        raise EmbeddingProviderFailure(
                            "The provider returned an invalid vector index."
                        )
                    if ordered[item.index] is not None:
                        raise EmbeddingProviderFailure(
                            "The provider returned a duplicate vector index."
                        )
                    if not isinstance(item.embedding, Sequence) or isinstance(
                        item.embedding, (str, bytes)
                    ):
                        raise EmbeddingProviderFailure(
                            "The provider returned an invalid embedding payload."
                        )
                    ordered[item.index] = item.embedding

                if any(vector is None for vector in ordered):
                    raise EmbeddingProviderFailure("The provider omitted an embedding vector.")
                batch_vectors = cast(list[Sequence[float]], ordered)
                vectors.extend(validate_embedding_vectors(batch_vectors, expected_count=len(batch)))
            except EmbeddingProviderFailure:
                raise
            except (AttributeError, OverflowError, TypeError, ValueError) as error:
                raise EmbeddingProviderFailure(
                    "The provider returned an invalid embedding response."
                ) from error

        return vectors
