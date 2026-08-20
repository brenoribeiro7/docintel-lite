from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError

from docintel.api.routes.documents import router as documents_router
from docintel.api.routes.query import router as query_router
from docintel.api.schemas import HealthResponse, ReadinessResponse
from docintel.db.session import engine
from docintel.documents.extraction import DocumentIngestionError
from docintel.rag.service import RAGError

app = FastAPI(
    title="DocIntel Lite",
    version="1.0.0",
    docs_url=None,
    redoc_url=None,
    openapi_url=None,
)
app.include_router(documents_router)
app.include_router(query_router)


@app.exception_handler(DocumentIngestionError)
async def handle_ingestion_error(
    _request: Request,
    error: DocumentIngestionError,
) -> JSONResponse:
    return JSONResponse(status_code=error.status_code, content=error.response_body())


@app.exception_handler(RAGError)
async def handle_rag_error(
    _request: Request,
    error: RAGError,
) -> JSONResponse:
    return JSONResponse(status_code=error.status_code, content=error.response_body())


@app.get("/healthz", response_model=HealthResponse)
def healthz() -> HealthResponse:
    return HealthResponse(status="ok")


@app.get(
    "/readyz",
    response_model=ReadinessResponse,
    responses={503: {"model": ReadinessResponse}},
)
def readyz() -> ReadinessResponse | JSONResponse:
    try:
        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))
            vector_version = connection.scalar(
                text("SELECT extversion FROM pg_extension WHERE extname = 'vector'")
            )
    except SQLAlchemyError:
        vector_version = None
    if vector_version is None:
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={"status": "not_ready"},
        )
    return ReadinessResponse(status="ready")
