import uuid
from collections.abc import AsyncIterator

import httpx
import pytest
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from docintel.app import app
from docintel.db.models import DocumentPage
from docintel.documents.extraction import MAX_FILE_BYTES
from tests.fixtures.pdfs import make_text_pdf

pytestmark = [pytest.mark.database, pytest.mark.anyio]


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.fixture
async def client(database_session: Session) -> AsyncIterator[httpx.AsyncClient]:
    del database_session
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as test_client:
        yield test_client


async def test_healthz_is_independent_of_external_services(client: httpx.AsyncClient) -> None:
    response = await client.get("/healthz")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


async def test_readyz_with_healthy_database(client: httpx.AsyncClient) -> None:
    response = await client.get("/readyz")
    assert response.status_code == 200
    assert response.json() == {"status": "ready"}


async def test_valid_upload_persists_page_metadata(client: httpx.AsyncClient) -> None:
    response = await client.post(
        "/api/v1/documents",
        files={"file": ("lesson.pdf", make_text_pdf(["One", None, "Three"]), "application/pdf")},
    )
    assert response.status_code == 201
    body = response.json()
    assert body["filename"] == "lesson.pdf"
    assert body["page_count"] == 3
    assert body["extracted_char_count"] == 8
    assert body["status"] == "INGESTED"
    assert body["embedding_model"] is None
    assert body["embedding_dimensions"] is None
    assert body["indexed_at"] is None


async def test_duplicate_upload_returns_existing_document_id(client: httpx.AsyncClient) -> None:
    pdf = make_text_pdf(["Duplicate"])
    first = await client.post(
        "/api/v1/documents",
        files={"file": ("first.pdf", pdf, "application/pdf")},
    )
    duplicate = await client.post(
        "/api/v1/documents",
        files={"file": ("second.pdf", pdf, "application/pdf")},
    )
    assert first.status_code == 201
    assert duplicate.status_code == 409
    assert duplicate.json() == {
        "code": "document_already_exists",
        "message": "A document with the same content already exists.",
        "document_id": first.json()["id"],
    }


async def test_large_upload_returns_413(client: httpx.AsyncClient) -> None:
    response = await client.post(
        "/api/v1/documents",
        files={"file": ("large.pdf", b"x" * (MAX_FILE_BYTES + 1), "application/pdf")},
    )
    assert response.status_code == 413
    assert response.json()["code"] == "document_too_large"


@pytest.mark.parametrize(
    ("filename", "media_type"),
    [("document.txt", "application/pdf"), ("document.pdf", "text/plain")],
)
async def test_invalid_type_returns_415(
    client: httpx.AsyncClient,
    filename: str,
    media_type: str,
) -> None:
    response = await client.post(
        "/api/v1/documents",
        files={"file": (filename, make_text_pdf(["Text"]), media_type)},
    )
    assert response.status_code == 415
    assert response.json()["code"] == "unsupported_document_type"


async def test_invalid_pdf_returns_422(client: httpx.AsyncClient) -> None:
    response = await client.post(
        "/api/v1/documents",
        files={"file": ("invalid.pdf", b"%PDF-1.4 broken", "application/pdf")},
    )
    assert response.status_code == 422
    assert response.json()["code"] == "invalid_pdf"


async def test_pdf_without_text_returns_422(client: httpx.AsyncClient) -> None:
    response = await client.post(
        "/api/v1/documents",
        files={"file": ("blank.pdf", make_text_pdf([None]), "application/pdf")},
    )
    assert response.status_code == 422
    assert response.json()["code"] == "document_text_not_extractable"


async def test_get_documents_returns_metadata_only(client: httpx.AsyncClient) -> None:
    upload = await client.post(
        "/api/v1/documents",
        files={"file": ("metadata.pdf", make_text_pdf(["Private page text"]), "application/pdf")},
    )
    response = await client.get("/api/v1/documents")
    assert response.status_code == 200
    assert len(response.json()) == 1
    item = response.json()[0]
    assert item["id"] == upload.json()["id"]
    assert "content" not in item
    assert "pages" not in item


async def test_delete_document_cascades_pages(
    client: httpx.AsyncClient,
    database_session: Session,
) -> None:
    upload = await client.post(
        "/api/v1/documents",
        files={"file": ("delete.pdf", make_text_pdf(["One", "Two"]), "application/pdf")},
    )
    document_id = upload.json()["id"]
    response = await client.delete(f"/api/v1/documents/{document_id}")
    assert response.status_code == 204
    database_session.expire_all()
    assert database_session.scalar(select(func.count(DocumentPage.document_id))) == 0


async def test_delete_missing_document_returns_404(client: httpx.AsyncClient) -> None:
    response = await client.delete(f"/api/v1/documents/{uuid.uuid4()}")
    assert response.status_code == 404
    assert response.json()["code"] == "document_not_found"
