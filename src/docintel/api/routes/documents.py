import uuid
from typing import Annotated

from fastapi import APIRouter, Depends, File, Response, UploadFile, status
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session

from docintel.api.schemas import DocumentResponse, DuplicateDocumentResponse, ErrorResponse
from docintel.db.session import get_db
from docintel.documents.extraction import (
    MAX_FILE_BYTES,
    PDF_MEDIA_TYPE,
    InvalidFilename,
    MissingFileError,
    UnsupportedDocumentTypeError,
    validate_upload_size,
)
from docintel.documents.normalization import InvalidFilenameError, sanitize_filename
from docintel.documents.service import delete_document, ingest_document, list_documents

router = APIRouter(prefix="/api/v1/documents", tags=["documents"])
DatabaseSession = Annotated[Session, Depends(get_db)]


@router.post(
    "",
    response_model=DocumentResponse,
    status_code=status.HTTP_201_CREATED,
    responses={
        409: {"model": DuplicateDocumentResponse},
        413: {"model": ErrorResponse},
        415: {"model": ErrorResponse},
        422: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
    },
)
async def upload_document(
    session: DatabaseSession,
    file: Annotated[UploadFile | None, File()] = None,
) -> DocumentResponse:
    if file is None:
        raise MissingFileError

    try:
        # Reading one byte beyond the cap rejects oversized uploads without retaining them.
        file_bytes = await file.read(MAX_FILE_BYTES + 1)
    finally:
        await file.close()
    validate_upload_size(file_bytes)

    media_type = (file.content_type or "").split(";", maxsplit=1)[0].strip().casefold()
    if media_type != PDF_MEDIA_TYPE:
        raise UnsupportedDocumentTypeError

    try:
        filename = sanitize_filename(file.filename or "")
    except InvalidFilenameError as error:
        raise InvalidFilename from error
    if not filename.casefold().endswith(".pdf"):
        raise UnsupportedDocumentTypeError

    document = ingest_document(
        session,
        filename=filename,
        media_type=PDF_MEDIA_TYPE,
        file_bytes=file_bytes,
    )
    return DocumentResponse.from_model(document)


@router.get("", response_model=list[DocumentResponse])
def get_documents(session: DatabaseSession) -> list[DocumentResponse]:
    return [DocumentResponse.from_model(document) for document in list_documents(session)]


@router.delete(
    "/{document_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    responses={404: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
)
def remove_document(document_id: uuid.UUID, session: DatabaseSession) -> Response:
    if not delete_document(session, document_id):
        return JSONResponse(
            content={"code": "document_not_found", "message": "Document not found."},
            status_code=status.HTTP_404_NOT_FOUND,
        )
    return Response(status_code=status.HTTP_204_NO_CONTENT)
