"""Document management endpoints.

Errors are returned as {"detail": {"code": ..., "message": ...}} so the frontend can show
a readable message and branch on a stable code. Stack traces are never returned.
"""

from fastapi import APIRouter, Depends, Response, UploadFile, status
from fastapi.responses import FileResponse

from backend.api.dependencies import (
    api_error,
    get_file_principal,
    get_pipeline,
    get_principal,
    get_repository,
    require_admin,
)

# Reading needs any logged-in user; changing documents needs an admin.
READ = [Depends(get_principal)]
ADMIN = [Depends(require_admin)]
from backend.config import get_settings
from backend.ingestion.pipeline import IngestionError, IngestionPipeline
from backend.schemas import Chunk, DocumentRecord
from backend.storage.repositories import DocumentRepository

router = APIRouter(prefix="/api/documents", tags=["documents"])


def _not_found(document_id: str):
    return api_error(404, "document_not_found", f"No document with id '{document_id}'.")


# Sync (def) endpoints run in FastAPI's thread pool, so CPU-bound ingestion does not
# block the event loop.
@router.post("/upload", response_model=DocumentRecord, status_code=status.HTTP_201_CREATED, dependencies=ADMIN)
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


@router.get("", response_model=list[DocumentRecord], dependencies=READ)
def list_documents(repository: DocumentRepository = Depends(get_repository)) -> list[DocumentRecord]:
    return repository.list_all()


@router.get("/{document_id}", response_model=DocumentRecord, dependencies=READ)
def get_document(document_id: str, repository: DocumentRepository = Depends(get_repository)) -> DocumentRecord:
    document = repository.get(document_id)
    if document is None:
        raise _not_found(document_id)
    return document


@router.get("/{document_id}/chunks", response_model=list[Chunk], dependencies=READ)
def list_document_chunks(
    document_id: str, repository: DocumentRepository = Depends(get_repository)
) -> list[Chunk]:
    if repository.get(document_id) is None:
        raise _not_found(document_id)
    return repository.list_chunks(document_id)


@router.post("/{document_id}/reprocess", response_model=DocumentRecord, dependencies=ADMIN)
def reprocess_document(document_id: str, pipeline: IngestionPipeline = Depends(get_pipeline)) -> DocumentRecord:
    """Re-extract and re-chunk a stored PDF in place (same document_id), e.g. after the
    chunking logic changed. If it fails, the existing chunks stay searchable."""
    try:
        return pipeline.reprocess(document_id)
    except IngestionError as exc:
        raise api_error(exc.status_code, exc.code, exc.message, exc.document_id) from exc


@router.get("/{document_id}/file", dependencies=[Depends(get_file_principal)])
def get_document_file(document_id: str, repository: DocumentRepository = Depends(get_repository)) -> FileResponse:
    """The original PDF, served inline so browsers can open it at a page (…/file#page=4)."""
    document = repository.get(document_id)
    path = get_settings().upload_dir / f"{document_id}.pdf"
    if document is None or not path.is_file():
        raise _not_found(document_id)
    return FileResponse(
        path, media_type="application/pdf", filename=document.filename, content_disposition_type="inline"
    )


@router.delete("/{document_id}", status_code=status.HTTP_204_NO_CONTENT, dependencies=ADMIN)
def delete_document(document_id: str, pipeline: IngestionPipeline = Depends(get_pipeline)) -> Response:
    if not pipeline.delete(document_id):
        raise _not_found(document_id)
    return Response(status_code=status.HTTP_204_NO_CONTENT)
