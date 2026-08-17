from dataclasses import dataclass
from io import BytesIO

from pypdf import PdfReader

from docintel.documents.normalization import normalize_text

MAX_FILE_BYTES = 10 * 1024 * 1024
MAX_PAGES = 100
MAX_NORMALIZED_CHARACTERS = 500_000
PDF_MEDIA_TYPE = "application/pdf"


class DocumentIngestionError(Exception):
    status_code = 422
    code = "document_ingestion_failed"
    message = "The document could not be ingested."

    def __init__(self, message: str | None = None) -> None:
        super().__init__(message or self.message)
        self.message = message or self.message

    def response_body(self) -> dict[str, str]:
        return {"code": self.code, "message": self.message}


class MissingFileError(DocumentIngestionError):
    code = "file_required"
    message = "A PDF file is required."


class DocumentTooLargeError(DocumentIngestionError):
    status_code = 413
    code = "document_too_large"
    message = "The document exceeds the 10 MiB upload limit."


class UnsupportedDocumentTypeError(DocumentIngestionError):
    status_code = 415
    code = "unsupported_document_type"
    message = "Only uploads named .pdf with media type application/pdf are supported."


class InvalidFilename(DocumentIngestionError):
    code = "invalid_filename"
    message = "The upload filename is empty or unsafe."


class InvalidPdfError(DocumentIngestionError):
    code = "invalid_pdf"
    message = "The file is not a structurally valid PDF."


class EncryptedPdfError(DocumentIngestionError):
    code = "encrypted_pdf_unsupported"
    message = "Encrypted PDFs are not supported."


class DocumentLimitsExceededError(DocumentIngestionError):
    code = "document_limits_exceeded"
    message = "The document exceeds the page or normalized text limit."


class DocumentTextNotExtractableError(DocumentIngestionError):
    code = "document_text_not_extractable"
    message = "The PDF contains no extractable text."


@dataclass(frozen=True, slots=True)
class ExtractedPage:
    page_number: int
    content: str

    @property
    def char_count(self) -> int:
        return len(self.content)


@dataclass(frozen=True, slots=True)
class ExtractedDocument:
    pages: list[ExtractedPage]
    extracted_char_count: int

    @property
    def page_count(self) -> int:
        return len(self.pages)


def validate_upload_size(file_bytes: bytes) -> None:
    if len(file_bytes) > MAX_FILE_BYTES:
        raise DocumentTooLargeError


def extract_pdf(file_bytes: bytes) -> ExtractedDocument:
    validate_upload_size(file_bytes)
    if not file_bytes or not file_bytes.startswith(b"%PDF-"):
        raise InvalidPdfError

    try:
        reader = PdfReader(BytesIO(file_bytes), strict=True)
        if reader.is_encrypted:
            raise EncryptedPdfError

        page_count = len(reader.pages)
        if page_count == 0:
            raise InvalidPdfError
        if page_count > MAX_PAGES:
            raise DocumentLimitsExceededError

        extracted_pages: list[ExtractedPage] = []
        total_characters = 0
        for page_number, page in enumerate(reader.pages, start=1):
            content = normalize_text(page.extract_text() or "")
            total_characters += len(content)
            if total_characters > MAX_NORMALIZED_CHARACTERS:
                raise DocumentLimitsExceededError
            # Empty pages remain explicit so page numbers always match the source PDF.
            extracted_pages.append(ExtractedPage(page_number=page_number, content=content))
    except DocumentIngestionError:
        raise
    except Exception as error:
        # The parser boundary is fail-closed and never exposes parser details to clients.
        raise InvalidPdfError from error

    if total_characters == 0:
        raise DocumentTextNotExtractableError
    return ExtractedDocument(
        pages=extracted_pages,
        extracted_char_count=total_characters,
    )
