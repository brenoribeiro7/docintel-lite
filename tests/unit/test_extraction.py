import pytest

from docintel.documents.extraction import (
    MAX_FILE_BYTES,
    DocumentLimitsExceededError,
    DocumentTextNotExtractableError,
    DocumentTooLargeError,
    EncryptedPdfError,
    InvalidPdfError,
    extract_pdf,
    validate_upload_size,
)
from docintel.documents.service import sha256_digest
from tests.fixtures.pdfs import make_encrypted_pdf, make_text_pdf


def test_sha256_is_deterministic() -> None:
    expected = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    assert sha256_digest(b"abc") == expected
    assert sha256_digest(b"abc") == sha256_digest(b"abc")


def test_valid_text_is_extracted_per_page() -> None:
    extracted = extract_pdf(make_text_pdf(["First page", "Second page"]))
    assert [page.page_number for page in extracted.pages] == [1, 2]
    assert [page.content for page in extracted.pages] == ["First page", "Second page"]
    assert extracted.extracted_char_count == len("First pageSecond page")


def test_empty_document_is_rejected() -> None:
    with pytest.raises(DocumentTextNotExtractableError):
        extract_pdf(make_text_pdf([None]))


def test_empty_intermediate_page_is_preserved() -> None:
    extracted = extract_pdf(make_text_pdf(["First", None, "Third"]))
    assert [(page.page_number, page.content) for page in extracted.pages] == [
        (1, "First"),
        (2, ""),
        (3, "Third"),
    ]


def test_page_limit_is_enforced() -> None:
    with pytest.raises(DocumentLimitsExceededError):
        extract_pdf(make_text_pdf(["text"] * 101))


def test_normalized_character_limit_is_enforced() -> None:
    with pytest.raises(DocumentLimitsExceededError):
        extract_pdf(make_text_pdf(["a" * 500_001]))


def test_file_size_limit_is_enforced_before_parsing() -> None:
    with pytest.raises(DocumentTooLargeError):
        validate_upload_size(b"x" * (MAX_FILE_BYTES + 1))


def test_encrypted_pdf_is_rejected() -> None:
    with pytest.raises(EncryptedPdfError):
        extract_pdf(make_encrypted_pdf())


@pytest.mark.parametrize("content", [b"", b"not a pdf", b"%PDF-1.4 broken"])
def test_invalid_pdf_is_rejected(content: bytes) -> None:
    with pytest.raises(InvalidPdfError):
        extract_pdf(content)
