import pytest

from docintel.documents.extraction import DocumentLimitsExceededError, ExtractedPage
from docintel.indexing.service import EmbeddingProviderApiError, prepare_index
from docintel.providers.embeddings import EMBEDDING_DIMENSIONS
from tests.fakes import FakeEmbeddingProvider
from tests.fixtures.tokens import text_with_token_count


def test_more_than_250_chunks_fails_before_provider() -> None:
    provider = FakeEmbeddingProvider()
    with pytest.raises(DocumentLimitsExceededError):
        prepare_index([ExtractedPage(1, text_with_token_count(125_101))], provider)
    assert provider.calls == []


def test_fake_provider_response_is_validated_by_indexing_boundary() -> None:
    provider = FakeEmbeddingProvider(default_vector=[1.0] * (EMBEDDING_DIMENSIONS - 1))
    with pytest.raises(EmbeddingProviderApiError):
        prepare_index([ExtractedPage(1, "text")], provider)
