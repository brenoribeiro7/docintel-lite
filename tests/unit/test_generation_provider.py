from collections.abc import Sequence
from dataclasses import dataclass
from unittest.mock import patch

import httpx2
import openai
import pytest
from pydantic import SecretStr, ValidationError

from docintel.config import Settings
from docintel.providers.generation import (
    GENERATION_MAX_OUTPUT_TOKENS,
    GENERATION_MODEL,
    GENERATION_TIMEOUT_SECONDS,
    GenerationOutput,
    GenerationProviderFailure,
    GenerationProviderRefusal,
    GenerationProviderUnconfigured,
    OpenAIGenerationProvider,
)


@dataclass
class StubContent:
    type: str


@dataclass
class StubOutputItem:
    type: str
    content: Sequence[StubContent]


@dataclass
class StubParsedResponse:
    status: str | None = "completed"
    output: Sequence[StubOutputItem] = ()
    output_parsed: object = None


class RecordingResponses:
    def __init__(
        self,
        *,
        response: StubParsedResponse | None = None,
        error: Exception | None = None,
    ) -> None:
        self.response = response or StubParsedResponse(
            output_parsed=GenerationOutput(
                answer="Grounded [S1]",
                abstained=False,
                citation_ids=["S1"],
            )
        )
        self.error = error
        self.calls: list[dict[str, object]] = []

    def parse(
        self,
        *,
        model: str,
        instructions: str,
        input: str,
        reasoning: dict[str, str],
        max_output_tokens: int,
        store: bool,
        text_format: type[GenerationOutput],
    ) -> StubParsedResponse:
        self.calls.append(
            {
                "model": model,
                "instructions": instructions,
                "input": input,
                "reasoning": reasoning,
                "max_output_tokens": max_output_tokens,
                "store": store,
                "text_format": text_format,
            }
        )
        if self.error is not None:
            raise self.error
        return self.response


@dataclass
class StubClient:
    responses: RecordingResponses


def make_provider(resource: RecordingResponses) -> OpenAIGenerationProvider:
    return OpenAIGenerationProvider(api_key=None, client=StubClient(resource))


def test_responses_parse_uses_fixed_structured_contract() -> None:
    resource = RecordingResponses()
    output = make_provider(resource).generate(instructions="rules", input_text='{"data":true}')
    assert output.answer == "Grounded [S1]"
    assert resource.calls == [
        {
            "model": GENERATION_MODEL,
            "instructions": "rules",
            "input": '{"data":true}',
            "reasoning": {"effort": "none"},
            "max_output_tokens": GENERATION_MAX_OUTPUT_TOKENS,
            "store": False,
            "text_format": GenerationOutput,
        }
    ]
    call = resource.calls[0]
    for forbidden_parameter in (
        "temperature",
        "tools",
        "previous_response_id",
        "conversation",
        "background",
        "stream",
    ):
        assert forbidden_parameter not in call


def test_provider_refusal_is_distinct_from_document_abstention() -> None:
    response = StubParsedResponse(
        output=[StubOutputItem(type="message", content=[StubContent(type="refusal")])]
    )
    with pytest.raises(GenerationProviderRefusal):
        make_provider(RecordingResponses(response=response)).generate(
            instructions="rules",
            input_text="data",
        )


@pytest.mark.parametrize(
    "response",
    [
        StubParsedResponse(status="completed", output_parsed=None),
        StubParsedResponse(status="incomplete", output_parsed=None),
    ],
)
def test_missing_or_incomplete_output_is_rejected(response: StubParsedResponse) -> None:
    with pytest.raises(GenerationProviderFailure):
        make_provider(RecordingResponses(response=response)).generate(
            instructions="rules",
            input_text="data",
        )


def provider_errors() -> list[Exception]:
    request = httpx2.Request("POST", "https://api.openai.com/v1/responses")
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
        ValueError("malformed structured output"),
    ]


@pytest.mark.parametrize("provider_error", provider_errors())
def test_sdk_and_parsing_errors_are_sanitized(provider_error: Exception) -> None:
    resource = RecordingResponses(error=provider_error)
    with pytest.raises(GenerationProviderFailure, match="provider request failed") as captured:
        make_provider(resource).generate(instructions="rules", input_text="private content")
    assert str(provider_error) not in str(captured.value)


def test_client_has_explicit_timeout_and_no_retries() -> None:
    resource = RecordingResponses()
    client = StubClient(resource)
    with patch("docintel.providers.generation.OpenAI", return_value=client) as client_factory:
        provider = OpenAIGenerationProvider(api_key=SecretStr("configured-test-key"))
        provider.generate(instructions="rules", input_text="data")
    client_factory.assert_called_once_with(
        api_key="configured-test-key",
        max_retries=0,
        timeout=GENERATION_TIMEOUT_SECONDS,
    )


def test_missing_api_key_fails_only_when_generation_is_requested() -> None:
    provider = OpenAIGenerationProvider(api_key=None)
    assert not provider.is_configured
    with pytest.raises(GenerationProviderUnconfigured):
        provider.generate(instructions="rules", input_text="data")


def test_generation_model_configuration_is_fixed() -> None:
    assert Settings().openai_generation_model == GENERATION_MODEL
    with pytest.raises(ValidationError):
        Settings.model_validate({"OPENAI_GENERATION_MODEL": "other-model"})
    with pytest.raises(ValueError, match="Only gpt-5.6-terra"):
        OpenAIGenerationProvider(api_key=None, model="other-model")


def test_structured_output_rejects_extra_fields_and_type_coercion() -> None:
    with pytest.raises(ValidationError):
        GenerationOutput.model_validate(
            {"answer": "", "abstained": "true", "citation_ids": [], "extra": "field"}
        )
