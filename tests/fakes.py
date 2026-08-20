from collections.abc import Callable, Mapping, Sequence

from docintel.providers.embeddings import (
    EMBEDDING_DIMENSIONS,
    EMBEDDING_MODEL,
    EmbeddingProviderFailure,
)
from docintel.providers.generation import (
    GENERATION_MODEL,
    GenerationOutput,
    GenerationProviderFailure,
)


def controlled_vector(first: float, second: float = 0.0) -> list[float]:
    return [first, second, *([0.0] * (EMBEDDING_DIMENSIONS - 2))]


class FakeEmbeddingProvider:
    model = EMBEDDING_MODEL
    dimensions = EMBEDDING_DIMENSIONS

    def __init__(
        self,
        *,
        vectors: Mapping[str, Sequence[float]] | None = None,
        default_vector: Sequence[float] | None = None,
        failure: EmbeddingProviderFailure | None = None,
    ) -> None:
        self._vectors = dict(vectors or {})
        self._default_vector = list(default_vector or controlled_vector(1.0))
        self.failure = failure
        self.calls: list[list[str]] = []

    @property
    def is_configured(self) -> bool:
        return True

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        self.calls.append(list(texts))
        if self.failure is not None:
            raise self.failure
        return [list(self._vectors.get(text, self._default_vector)) for text in texts]


GenerationCallback = Callable[[str, str], GenerationOutput]


class FakeGenerationProvider:
    model = GENERATION_MODEL

    def __init__(
        self,
        *,
        output: GenerationOutput | None = None,
        callback: GenerationCallback | None = None,
        failure: GenerationProviderFailure | None = None,
        configured: bool = True,
    ) -> None:
        self.output = output or GenerationOutput(answer="", abstained=True, citation_ids=[])
        self.callback = callback
        self.failure = failure
        self.configured = configured
        self.calls: list[dict[str, str]] = []

    @property
    def is_configured(self) -> bool:
        return self.configured

    def generate(self, *, instructions: str, input_text: str) -> GenerationOutput:
        self.calls.append({"instructions": instructions, "input_text": input_text})
        if self.failure is not None:
            raise self.failure
        if self.callback is not None:
            return self.callback(instructions, input_text)
        return self.output
