"""FastAPI dependencies shared by the routers.

Long-lived services (database, tokenizer, embedder, FAISS store) are created once at
startup and kept on app.state; these functions hand them to request handlers.
"""

from fastapi import Depends, HTTPException, Request

from backend.config import get_settings
from backend.ingestion.pipeline import IngestionPipeline
from backend.retrieval.dense_retriever import DenseRetriever
from backend.retrieval.embeddings import Embedder
from backend.retrieval.vector_store import FaissVectorStore
from backend.storage.repositories import DocumentRepository


def api_error(status_code: int, code: str, message: str, document_id: str | None = None) -> HTTPException:
    """Errors are returned as {"detail": {"code", "message"}}; never stack traces."""
    detail = {"code": code, "message": message}
    if document_id:
        detail["document_id"] = document_id
    return HTTPException(status_code=status_code, detail=detail)


def get_repository(request: Request) -> DocumentRepository:
    return DocumentRepository(request.app.state.db)


def get_retrieval_services(request: Request) -> tuple[Embedder, FaissVectorStore]:
    embedder = request.app.state.embedder
    store = request.app.state.vector_store
    if embedder is None or store is None:
        raise api_error(
            503, "retrieval_unavailable",
            "The embedding model or search index is unavailable. Check the backend logs.",
        )
    return embedder, store


def get_pipeline(
    request: Request,
    repository: DocumentRepository = Depends(get_repository),
    services: tuple[Embedder, FaissVectorStore] = Depends(get_retrieval_services),
) -> IngestionPipeline:
    embedder, store = services
    return IngestionPipeline(get_settings(), repository, request.app.state.tokenizer, embedder, store)


def get_dense_retriever(
    repository: DocumentRepository = Depends(get_repository),
    services: tuple[Embedder, FaissVectorStore] = Depends(get_retrieval_services),
) -> DenseRetriever:
    embedder, store = services
    return DenseRetriever(embedder, store, repository)
