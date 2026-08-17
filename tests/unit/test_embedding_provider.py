from collections.abc import Callable, Sequence
from dataclasses import dataclass
from unittest.mock import patch

import httpx2
import openai
import pytest
from pydantic import SecretStr

from docintel.providers.embeddings import (
    EMBEDDING_BATCH_SIZE,
    EMBEDDING_DIMENSIONS,
    EMBEDDING_MODEL,
    EMBEDDING_TIMEOUT_SECONDS,
    EmbeddingProviderFailure,
    EmbeddingProviderUnconfigured,
    OpenAIEmbeddingProvider,
)
from tests.fakes import controlled_vector


@dataclass
class StubDatum:
    index: int
    embedding: object


@dataclass
class StubResponse:
    data: list[StubDatum]


ResponseFactory = Callable[[list[str]], list[StubDatum]]


class RecordingEmbeddings:
    def __init__(
        self,
        *,
        response_factory: ResponseFactory | None = None,
        error: Exception | None = None,
    ) -> None:
        self.calls: list[dict[str, object]] = []
        self._response_factory = response_factory or self._default_response
        self._error = error

    @staticmethod
    def _default_response(texts: list[str]) -> list[StubDatum]:
        return [
            StubDatum(index=index, embedding=controlled_vector(float(text)))
            for index, text in enumerate(texts)
        ]

    def create(
        self,
        *,
        input: Sequence[str],
        model: str,
        encoding_format: str,
        **extra: object,
    ) -> StubResponse:
        self.calls.append(
            {
                "input": list(input),
                "model": model,
                "encoding_format": encoding_format,
                **extra,
            }
        )
        if self._error is not None:
            raise self._error
        return StubResponse(data=self._response_factory(list(input)))


@dataclass
class StubClient:
    embeddings: RecordingEmbeddings


def make_provider(resource: RecordingEmbeddings) -> OpenAIEmbeddingProvider:
    return OpenAIEmbeddingProvider(api_key=None, client=StubClient(resource))


def test_empty_input_returns_empty_without_request() -> None:
    resource = RecordingEmbeddings()
    assert make_provider(resource).embed([]) == []
    assert resource.calls == []


def test_model_is_fixed_and_dimensions_parameter_is_absent() -> None:
    resource = RecordingEmbeddings()
    make_provider(resource).embed(["1"])
    assert resource.calls == [
        {
            "input": ["1"],
            "model": EMBEDDING_MODEL,
            "encoding_format": "float",
        }
    ]
    assert "dimensions" not in resource.calls[0]


@pytest.mark.parametrize(
    ("count", "expected_batch_sizes"),
    [
        (1, [1]),
        (EMBEDDING_BATCH_SIZE, [32]),
        (EMBEDDING_BATCH_SIZE + 1, [32, 1]),
        (250, [32, 32, 32, 32, 32, 32, 32, 26]),
    ],
)
def test_batching_and_original_order(count: int, expected_batch_sizes: list[int]) -> None:
    resource = RecordingEmbeddings()
    texts = [str(index + 1) for index in range(count)]
    vectors = make_provider(resource).embed(texts)
    batch_sizes: list[int] = []
    for call in resource.calls:
        batch_input = call["input"]
        assert isinstance(batch_input, list)
        batch_sizes.append(len(batch_input))
    assert batch_sizes == expected_batch_sizes
    assert [vector[0] for vector in vectors] == [float(text) for text in texts]


def test_out_of_order_response_is_reordered_by_index() -> None:
    def reverse_response(texts: list[str]) -> list[StubDatum]:
        return [
            StubDatum(index=index, embedding=controlled_vector(float(texts[index])))
            for index in reversed(range(len(texts)))
        ]

    vectors = make_provider(RecordingEmbeddings(response_factory=reverse_response)).embed(
        ["1", "2", "3"]
    )
    assert [vector[0] for vector in vectors] == [1.0, 2.0, 3.0]


def test_incorrect_vector_quantity_is_rejected() -> None:
    resource = RecordingEmbeddings(
        response_factory=lambda _texts: [StubDatum(0, controlled_vector(1.0))]
    )
    with pytest.raises(EmbeddingProviderFailure):
        make_provider(resource).embed(["1", "2"])


def test_duplicate_or_missing_index_is_rejected() -> None:
    resource = RecordingEmbeddings(
        response_factory=lambda _texts: [
            StubDatum(0, controlled_vector(1.0)),
            StubDatum(0, controlled_vector(2.0)),
        ]
    )
    with pytest.raises(EmbeddingProviderFailure):
        make_provider(resource).embed(["1", "2"])


def test_out_of_range_index_is_rejected() -> None:
    resource = RecordingEmbeddings(
        response_factory=lambda _texts: [StubDatum(1, controlled_vector(1.0))]
    )
    with pytest.raises(EmbeddingProviderFailure):
        make_provider(resource).embed(["1"])


@pytest.mark.parametrize("dimension", [1535, 1537])
def test_invalid_dimensions_are_rejected(dimension: int) -> None:
    resource = RecordingEmbeddings(
        response_factory=lambda _texts: [StubDatum(0, [1.0] * dimension)]
    )
    with pytest.raises(EmbeddingProviderFailure):
        make_provider(resource).embed(["1"])


@pytest.mark.parametrize("component", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_components_are_rejected(component: float) -> None:
    vector = controlled_vector(1.0)
    vector[10] = component
    resource = RecordingEmbeddings(response_factory=lambda _texts: [StubDatum(0, vector)])
    with pytest.raises(EmbeddingProviderFailure):
        make_provider(resource).embed(["1"])


def provider_errors() -> list[Exception]:
    request = httpx2.Request("POST", "https://api.openai.com/v1/embeddings")
    return [
        openai.AuthenticationError(
            "authentication failed",
            response=httpx2.Response(401, request=request),
            body=None,
        ),
        openai.RateLimitError(
            "rate limited",
            response=httpx2.Response(429, request=request),
            body=None,
        ),
        openai.APITimeoutError(request=request),
        openai.InternalServerError(
            "provider failed",
            response=httpx2.Response(500, request=request),
            body=None,
        ),
    ]


@pytest.mark.parametrize("provider_error", provider_errors())
def test_sdk_errors_are_sanitized(provider_error: Exception) -> None:
    resource = RecordingEmbeddings(error=provider_error)
    with pytest.raises(EmbeddingProviderFailure, match="provider request failed") as captured:
        make_provider(resource).embed(["1"])
    assert str(provider_error) not in str(captured.value)


def test_client_has_explicit_timeout_and_no_retries() -> None:
    resource = RecordingEmbeddings()
    client = StubClient(resource)
    with patch("docintel.providers.embeddings.OpenAI", return_value=client) as client_factory:
        provider = OpenAIEmbeddingProvider(api_key=SecretStr("configured-test-key"))
        provider.embed(["1"])
    client_factory.assert_called_once_with(
        api_key="configured-test-key",
        max_retries=0,
        timeout=EMBEDDING_TIMEOUT_SECONDS,
    )


def test_missing_api_key_fails_only_when_embedding_is_requested() -> None:
    provider = OpenAIEmbeddingProvider(api_key=None)
    with pytest.raises(EmbeddingProviderUnconfigured):
        provider.embed(["1"])
    assert provider.model == EMBEDDING_MODEL
    assert provider.dimensions == EMBEDDING_DIMENSIONS
