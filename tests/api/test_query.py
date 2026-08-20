import uuid
from collections.abc import AsyncIterator
from datetime import UTC, datetime

import httpx
import pytest
from sqlalchemy.orm import Session

from docintel.api.routes.documents import get_embedding_provider
from docintel.api.routes.query import get_generation_provider, get_query_embedding_provider
from docintel.app import app
from docintel.db.models import Document, DocumentPage
from docintel.providers.embeddings import EmbeddingProviderFailure, OpenAIEmbeddingProvider
from docintel.providers.generation import (
    GenerationOutput,
    GenerationProviderFailure,
    GenerationProviderRefusal,
    OpenAIGenerationProvider,
)
from docintel.rag.prompt import GROUNDING_INSTRUCTIONS
from docintel.rag.sources import CANONICAL_ABSTENTION_ANSWER
from tests.fakes import FakeEmbeddingProvider, FakeGenerationProvider
from tests.fixtures.pdfs import make_text_pdf

pytestmark = [pytest.mark.database, pytest.mark.anyio]


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.fixture
def fake_embedding_provider() -> FakeEmbeddingProvider:
    return FakeEmbeddingProvider()


@pytest.fixture
def fake_generation_provider() -> FakeGenerationProvider:
    return FakeGenerationProvider()


@pytest.fixture
async def client(
    database_session: Session,
    fake_embedding_provider: FakeEmbeddingProvider,
    fake_generation_provider: FakeGenerationProvider,
) -> AsyncIterator[httpx.AsyncClient]:
    del database_session
    app.dependency_overrides[get_embedding_provider] = lambda: fake_embedding_provider
    app.dependency_overrides[get_query_embedding_provider] = lambda: fake_embedding_provider
    app.dependency_overrides[get_generation_provider] = lambda: fake_generation_provider
    transport = httpx.ASGITransport(app=app)
    try:
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://testserver",
        ) as test_client:
            yield test_client
    finally:
        app.dependency_overrides.clear()


async def upload_document(
    client: httpx.AsyncClient,
    *,
    pages: list[str | None] | None = None,
) -> dict[str, object]:
    response = await client.post(
        "/api/v1/documents",
        files={
            "file": (
                "query.pdf",
                make_text_pdf(pages or ["The production port is 8443."]),
                "application/pdf",
            )
        },
    )
    assert response.status_code == 201
    body = response.json()
    assert isinstance(body, dict)
    return body


def add_document_without_chunks(database_session: Session, *, indexed: bool) -> Document:
    identifier = uuid.uuid4()
    document = Document(
        id=identifier,
        filename="empty-index.pdf",
        media_type="application/pdf",
        size_bytes=100,
        content_hash=identifier.hex.ljust(64, "0"),
        page_count=1,
        extracted_char_count=4,
        embedding_model="text-embedding-3-small" if indexed else None,
        embedding_dimensions=1536 if indexed else None,
        indexed_at=datetime.now(UTC) if indexed else None,
        pages=[
            DocumentPage(
                document_id=identifier,
                page_number=1,
                content="text",
                char_count=4,
            )
        ],
    )
    database_session.add(document)
    database_session.commit()
    return document


async def test_answerable_query_returns_grounded_real_source(
    client: httpx.AsyncClient,
    fake_generation_provider: FakeGenerationProvider,
) -> None:
    document = await upload_document(client)
    fake_generation_provider.output = GenerationOutput(
        answer="The production service uses port 8443 [S1].",
        abstained=False,
        citation_ids=["S1"],
    )

    response = await client.post(
        "/api/v1/query",
        json={"question": "What port is used?", "document_ids": [document["id"]]},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["question"] == "What port is used?"
    assert body["answer"] == "The production service uses port 8443 [S1]."
    assert body["abstained"] is False
    assert body["citation_ids"] == ["S1"]
    assert body["retrieval"] == {"top_k": 4, "chunks_returned": 1}
    assert body["sources"][0]["source_id"] == "S1"
    assert body["sources"][0]["document_id"] == document["id"]
    assert body["sources"][0]["filename"] == "query.pdf"
    assert body["sources"][0]["page_number"] == 1
    assert body["sources"][0]["chunk_index"] == 0
    assert body["sources"][0]["text"] == "The production port is 8443."
    assert isinstance(body["sources"][0]["chunk_id"], str)
    assert fake_generation_provider.calls[0]["instructions"] == GROUNDING_INSTRUCTIONS


async def test_multiple_retrieved_sources_remain_in_rank_order(
    client: httpx.AsyncClient,
    fake_generation_provider: FakeGenerationProvider,
) -> None:
    document = await upload_document(client, pages=["RTO is 2 hours.", "RPO is 15 minutes."])
    fake_generation_provider.output = GenerationOutput(
        answer="RTO is 2 hours [S1] and RPO is 15 minutes [S2].",
        abstained=False,
        citation_ids=["S1", "S2"],
    )
    response = await client.post(
        "/api/v1/query",
        json={"question": "What are RTO and RPO?", "document_ids": [document["id"]]},
    )
    assert response.status_code == 200
    assert [source["source_id"] for source in response.json()["sources"]] == ["S1", "S2"]
    assert [source["page_number"] for source in response.json()["sources"]] == [1, 2]


async def test_sources_include_retrieved_chunks_not_cited_by_generation(
    client: httpx.AsyncClient,
    fake_generation_provider: FakeGenerationProvider,
) -> None:
    document = await upload_document(client, pages=["Context.", "The answer is on page two."])
    fake_generation_provider.output = GenerationOutput(
        answer="The answer is on page two [S2].",
        abstained=False,
        citation_ids=["S2"],
    )
    response = await client.post(
        "/api/v1/query",
        json={"question": "Where is the answer?", "document_ids": [document["id"]]},
    )
    assert response.status_code == 200
    assert response.json()["citation_ids"] == ["S2"]
    assert [source["source_id"] for source in response.json()["sources"]] == ["S1", "S2"]


async def test_provider_abstention_is_canonicalized(
    client: httpx.AsyncClient,
    fake_generation_provider: FakeGenerationProvider,
) -> None:
    document = await upload_document(client)
    fake_generation_provider.output = GenerationOutput(answer="  ", abstained=True, citation_ids=[])
    response = await client.post(
        "/api/v1/query",
        json={"question": "Who is the CEO?", "document_ids": [document["id"]]},
    )
    assert response.status_code == 200
    assert response.json()["answer"] == CANONICAL_ABSTENTION_ANSWER
    assert response.json()["abstained"] is True
    assert response.json()["citation_ids"] == []
    assert len(response.json()["sources"]) == 1


async def test_empty_retrieval_abstains_without_generation(
    client: httpx.AsyncClient,
    database_session: Session,
    fake_generation_provider: FakeGenerationProvider,
) -> None:
    document = add_document_without_chunks(database_session, indexed=True)
    response = await client.post(
        "/api/v1/query",
        json={"question": "Question", "document_ids": [str(document.id)]},
    )
    assert response.status_code == 200
    assert response.json() == {
        "question": "Question",
        "answer": CANONICAL_ABSTENTION_ANSWER,
        "abstained": True,
        "citation_ids": [],
        "sources": [],
        "retrieval": {"top_k": 4, "chunks_returned": 0},
    }
    assert fake_generation_provider.calls == []


async def test_missing_document_returns_404(client: httpx.AsyncClient) -> None:
    response = await client.post(
        "/api/v1/query",
        json={"question": "Question", "document_ids": [str(uuid.uuid4())]},
    )
    assert response.status_code == 404
    assert response.json()["code"] == "documents_not_found"


async def test_ingested_document_returns_409(
    client: httpx.AsyncClient,
    database_session: Session,
) -> None:
    document = add_document_without_chunks(database_session, indexed=False)
    response = await client.post(
        "/api/v1/query",
        json={"question": "Question", "document_ids": [str(document.id)]},
    )
    assert response.status_code == 409
    assert response.json()["code"] == "documents_not_indexed"


async def test_partially_invalid_selection_fails_without_embedding(
    client: httpx.AsyncClient,
    fake_embedding_provider: FakeEmbeddingProvider,
) -> None:
    document = await upload_document(client)
    fake_embedding_provider.calls.clear()
    response = await client.post(
        "/api/v1/query",
        json={
            "question": "Question",
            "document_ids": [document["id"], str(uuid.uuid4())],
        },
    )
    assert response.status_code == 404
    assert fake_embedding_provider.calls == []


async def test_embedding_provider_unconfigured_returns_503(
    client: httpx.AsyncClient,
) -> None:
    document = await upload_document(client)
    app.dependency_overrides[get_query_embedding_provider] = lambda: OpenAIEmbeddingProvider(
        api_key=None
    )
    response = await client.post(
        "/api/v1/query",
        json={"question": "Question", "document_ids": [document["id"]]},
    )
    assert response.status_code == 503
    assert response.json()["code"] == "embedding_provider_unconfigured"


async def test_generation_provider_unconfigured_prevents_query_embedding(
    client: httpx.AsyncClient,
    fake_embedding_provider: FakeEmbeddingProvider,
) -> None:
    document = await upload_document(client)
    fake_embedding_provider.calls.clear()
    app.dependency_overrides[get_generation_provider] = lambda: OpenAIGenerationProvider(
        api_key=None
    )
    response = await client.post(
        "/api/v1/query",
        json={"question": "Question", "document_ids": [document["id"]]},
    )
    assert response.status_code == 503
    assert response.json()["code"] == "generation_provider_unconfigured"
    assert fake_embedding_provider.calls == []


async def test_document_selection_precedes_provider_configuration(
    client: httpx.AsyncClient,
    fake_embedding_provider: FakeEmbeddingProvider,
) -> None:
    fake_embedding_provider.calls.clear()
    app.dependency_overrides[get_generation_provider] = lambda: OpenAIGenerationProvider(
        api_key=None
    )
    response = await client.post(
        "/api/v1/query",
        json={"question": "Question", "document_ids": [str(uuid.uuid4())]},
    )
    assert response.status_code == 404
    assert response.json()["code"] == "documents_not_found"
    assert fake_embedding_provider.calls == []


async def test_embedding_provider_failure_returns_502(
    client: httpx.AsyncClient,
    fake_embedding_provider: FakeEmbeddingProvider,
) -> None:
    document = await upload_document(client)
    fake_embedding_provider.failure = EmbeddingProviderFailure("private embedding detail")
    response = await client.post(
        "/api/v1/query",
        json={"question": "Question", "document_ids": [document["id"]]},
    )
    assert response.status_code == 502
    assert response.json()["code"] == "embedding_provider_error"


@pytest.mark.parametrize(
    ("failure", "expected_code"),
    [
        (GenerationProviderFailure("private provider detail"), "generation_provider_error"),
        (GenerationProviderRefusal("private refusal detail"), "generation_provider_refusal"),
    ],
)
async def test_generation_provider_failures_are_sanitized(
    client: httpx.AsyncClient,
    fake_generation_provider: FakeGenerationProvider,
    failure: GenerationProviderFailure,
    expected_code: str,
) -> None:
    document = await upload_document(client)
    fake_generation_provider.failure = failure
    response = await client.post(
        "/api/v1/query",
        json={"question": "Question", "document_ids": [document["id"]]},
    )
    assert response.status_code == 502
    assert response.json()["code"] == expected_code
    assert "private" not in response.text


@pytest.mark.parametrize(
    "output",
    [
        GenerationOutput(answer="Invented [S9]", abstained=False, citation_ids=["S9"]),
        GenerationOutput(answer="Mismatch [S1]", abstained=False, citation_ids=["S2"]),
        GenerationOutput(answer="No citation", abstained=False, citation_ids=["S1"]),
        GenerationOutput(answer="", abstained=False, citation_ids=[]),
    ],
)
async def test_invalid_grounded_responses_return_502(
    client: httpx.AsyncClient,
    fake_generation_provider: FakeGenerationProvider,
    output: GenerationOutput,
) -> None:
    document = await upload_document(client)
    fake_generation_provider.output = output
    response = await client.post(
        "/api/v1/query",
        json={"question": "Question", "document_ids": [document["id"]]},
    )
    assert response.status_code == 502
    assert response.json()["code"] == "invalid_grounded_response"


@pytest.mark.parametrize(
    "payload",
    [
        {"question": "", "document_ids": [str(uuid.uuid4())]},
        {"question": "x" * 1_001, "document_ids": [str(uuid.uuid4())]},
        {"question": "Question", "document_ids": []},
        {
            "question": "Question",
            "document_ids": [str(identifier := uuid.uuid4()), str(identifier)],
        },
        {
            "question": "Question",
            "document_ids": [str(uuid.uuid4()) for _index in range(11)],
        },
    ],
)
async def test_query_request_validation_returns_422(
    client: httpx.AsyncClient,
    payload: dict[str, object],
) -> None:
    response = await client.post("/api/v1/query", json=payload)
    assert response.status_code == 422
