from collections.abc import Sequence
from typing import Protocol, cast

from openai import OpenAI
from pydantic import BaseModel, ConfigDict, SecretStr

GENERATION_MODEL = "gpt-5.6-terra"
GENERATION_MAX_OUTPUT_TOKENS = 700
GENERATION_TIMEOUT_SECONDS = 30.0


class GenerationOutput(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    answer: str
    abstained: bool
    citation_ids: list[str]


class GenerationProvider(Protocol):
    model: str

    @property
    def is_configured(self) -> bool: ...

    def generate(self, *, instructions: str, input_text: str) -> GenerationOutput: ...


class GenerationProviderFailure(Exception):
    """A sanitized remote-provider or response-contract failure."""


class GenerationProviderUnconfigured(GenerationProviderFailure):
    """The real provider was requested without credentials."""


class GenerationProviderRefusal(GenerationProviderFailure):
    """The provider refused the request for a safety reason."""


class _ResponseContent(Protocol):
    @property
    def type(self) -> str: ...


class _ResponseOutputItem(Protocol):
    @property
    def type(self) -> str: ...

    @property
    def content(self) -> Sequence[_ResponseContent]: ...


class _ParsedResponse(Protocol):
    @property
    def status(self) -> str | None: ...

    @property
    def output(self) -> Sequence[_ResponseOutputItem]: ...

    @property
    def output_parsed(self) -> object: ...


class _ResponsesResource(Protocol):
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
    ) -> _ParsedResponse: ...


class _GenerationClient(Protocol):
    @property
    def responses(self) -> _ResponsesResource: ...


def _contains_refusal(response: _ParsedResponse) -> bool:
    for item in response.output:
        if item.type != "message":
            continue
        if any(content.type == "refusal" for content in item.content):
            return True
    return False


class OpenAIGenerationProvider:
    def __init__(
        self,
        *,
        api_key: SecretStr | None,
        model: str = GENERATION_MODEL,
        client: _GenerationClient | None = None,
    ) -> None:
        if model != GENERATION_MODEL:
            raise ValueError(f"Only {GENERATION_MODEL} is supported by the v1 generation adapter.")
        self.model = model
        self._api_key = api_key
        self._client = client

    @property
    def is_configured(self) -> bool:
        if self._client is not None:
            return True
        return bool(self._api_key and self._api_key.get_secret_value())

    def _get_client(self) -> _GenerationClient:
        if self._client is not None:
            return self._client
        api_key = self._api_key.get_secret_value() if self._api_key is not None else ""
        if not api_key:
            raise GenerationProviderUnconfigured("The generation provider is not configured.")
        try:
            sdk_client = OpenAI(
                api_key=api_key,
                max_retries=0,
                timeout=GENERATION_TIMEOUT_SECONDS,
            )
        except Exception as error:
            raise GenerationProviderFailure(
                "The generation provider could not be initialized."
            ) from error
        self._client = cast(_GenerationClient, sdk_client)
        return self._client

    def generate(self, *, instructions: str, input_text: str) -> GenerationOutput:
        try:
            response = self._get_client().responses.parse(
                model=self.model,
                instructions=instructions,
                input=input_text,
                reasoning={"effort": "none"},
                max_output_tokens=GENERATION_MAX_OUTPUT_TOKENS,
                store=False,
                text_format=GenerationOutput,
            )
        except GenerationProviderFailure:
            raise
        except Exception as error:
            # Provider exceptions can contain parts of the request or credentials.
            raise GenerationProviderFailure("The generation provider request failed.") from error

        if _contains_refusal(response):
            raise GenerationProviderRefusal("The generation provider refused the request.")
        if response.status != "completed":
            raise GenerationProviderFailure("The generation provider response was incomplete.")
        parsed = response.output_parsed
        if not isinstance(parsed, GenerationOutput):
            raise GenerationProviderFailure(
                "The generation provider returned no structured output."
            )
        return parsed
