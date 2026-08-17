from collections.abc import Mapping, Sequence

from docintel.providers.embeddings import (
    EMBEDDING_DIMENSIONS,
    EMBEDDING_MODEL,
    EmbeddingProviderFailure,
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

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        self.calls.append(list(texts))
        if self.failure is not None:
            raise self.failure
        return [list(self._vectors.get(text, self._default_vector)) for text in texts]
