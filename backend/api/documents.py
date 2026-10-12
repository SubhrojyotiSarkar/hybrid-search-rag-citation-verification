"""Document management endpoints.

Errors are returned as {"detail": {"code": ..., "message": ...}} so the frontend can show
a readable message and branch on a stable code. Stack traces are never returned.
"""

from fastapi import APIRouter, Depends, HTTPException, Request, Response, UploadFile, status

from backend.config import get_settings
from backend.ingestion.pipeline import IngestionError, IngestionPipeline
from backend.schemas import Chunk, DocumentRecord
from backend.storage.repositories import DocumentRepository

router = APIRouter(prefix="/api/documents", tags=["documents"])


def get_repository(request: Request) -> DocumentRepository:
    return DocumentRepository(request.app.state.db)


def get_pipeline(
    request: Request, repository: DocumentRepository = Depends(get_repository)
) -> IngestionPipeline:
    return IngestionPipeline(get_settings(), repository, request.app.state.tokenizer)


def api_error(status_code: int, code: str, message: str, document_id: str | None = None) -> HTTPException:
    detail = {"code": code, "message": message}
    if document_id:
        detail["document_id"] = document_id
    return HTTPException(status_code=status_code, detail=detail)


def _not_found(document_id: str) -> HTTPException:
    return api_error(404, "document_not_found", f"No document with id '{document_id}'.")


# Sync (def) endpoints run in FastAPI's thread pool, so CPU-bound ingestion does not
# block the event loop.
@router.post("/upload", response_model=DocumentRecord, status_code=status.HTTP_201_CREATED)
def upload_document(file: UploadFile, pipeline: IngestionPipeline = Depends(get_pipeline)) -> DocumentRecord:
    # Read at most one byte past the limit: enough to detect an oversized file without
    # loading an arbitrarily large upload into memory.
    content = file.file.read(pipeline.settings.max_upload_bytes + 1)
    try:
        return pipeline.ingest(file.filename or "", content)
    except IngestionError as exc:
        raise api_error(exc.status_code, exc.code, exc.message, exc.document_id) from exc
    except Exception as exc:
        raise api_error(500, "internal_error", "Unexpected error while processing the document.") from exc


@router.get("", response_model=list[DocumentRecord])
def list_documents(repository: DocumentRepository = Depends(get_repository)) -> list[DocumentRecord]:
    return repository.list_all()


@router.get("/{document_id}", response_model=DocumentRecord)
def get_document(document_id: str, repository: DocumentRepository = Depends(get_repository)) -> DocumentRecord:
    document = repository.get(document_id)
    if document is None:
        raise _not_found(document_id)
    return document


@router.get("/{document_id}/chunks", response_model=list[Chunk])
def list_document_chunks(
    document_id: str, repository: DocumentRepository = Depends(get_repository)
) -> list[Chunk]:
    if repository.get(document_id) is None:
        raise _not_found(document_id)
    return repository.list_chunks(document_id)


@router.delete("/{document_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_document(document_id: str, pipeline: IngestionPipeline = Depends(get_pipeline)) -> Response:
    if not pipeline.delete(document_id):
        raise _not_found(document_id)
    return Response(status_code=status.HTTP_204_NO_CONTENT)
