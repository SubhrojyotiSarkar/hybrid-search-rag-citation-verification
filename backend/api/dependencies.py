"""FastAPI dependencies shared by the routers.

Long-lived services (database, tokenizer, embedder, FAISS store, BM25 index) are created
once at startup and kept on app.state; these functions hand them to request handlers.
"""

from collections.abc import Callable

from fastapi import Depends, HTTPException, Request

from backend.config import get_settings
from backend.generation.answer_generator import AnswerGenerator
from backend.generation.llm_client import LLMClient
from backend.ingestion.pipeline import IngestionPipeline
from backend.retrieval.hybrid_search import RetrievalServices, Retriever, build_retriever
from backend.schemas import DocumentStatus
from backend.storage.repositories import DocumentRepository


def api_error(status_code: int, code: str, message: str, document_id: str | None = None) -> HTTPException:
    """Errors are returned as {"detail": {"code", "message"}}; never stack traces."""
    detail = {"code": code, "message": message}
    if document_id:
        detail["document_id"] = document_id
    return HTTPException(status_code=status_code, detail=detail)


def get_repository(request: Request) -> DocumentRepository:
    return DocumentRepository(request.app.state.db)


def get_retrieval_services(request: Request) -> RetrievalServices:
    services = request.app.state.retrieval
    if services is None:
        raise api_error(
            503, "retrieval_unavailable",
            "The embedding model or search index is unavailable. Check the backend logs.",
        )
    return services


def get_pipeline(
    request: Request,
    repository: DocumentRepository = Depends(get_repository),
    services: RetrievalServices = Depends(get_retrieval_services),
) -> IngestionPipeline:
    return IngestionPipeline(
        get_settings(), repository, request.app.state.tokenizer,
        services.embedder, services.vector_store, services.bm25_index,
    )


def get_retriever_factory(
    repository: DocumentRepository = Depends(get_repository),
    services: RetrievalServices = Depends(get_retrieval_services),
) -> Callable[[str], Retriever]:
    """Returns a function mode -> retriever ("dense", "bm25" or "hybrid")."""
    settings = get_settings()
    return lambda mode: build_retriever(mode, services, repository, settings)


def get_llm_client(request: Request) -> LLMClient:
    llm = request.app.state.llm
    if llm is None:
        raise api_error(
            503, "llm_not_configured",
            "No LLM is configured. Set GEMINI_API_KEY (and LLM_PROVIDER=gemini) in .env and restart the backend.",
        )
    return llm


def get_answer_generator(
    retriever_factory: Callable[[str], Retriever] = Depends(get_retriever_factory),
    llm: LLMClient = Depends(get_llm_client),
) -> AnswerGenerator:
    settings = get_settings()
    return AnswerGenerator(retriever_factory(settings.retrieval_mode), llm, settings.retrieval_top_k)


def validate_document_ids(document_ids: list[str] | None, repository: DocumentRepository) -> None:
    """Reject unknown or not-yet-searchable documents explicitly instead of ignoring them."""
    if document_ids is None:
        return
    if not document_ids:
        raise api_error(400, "no_documents_selected", "Select at least one document.")
    unusable = []
    for document_id in document_ids:
        document = repository.get(document_id)
        if document is None or document.status != DocumentStatus.INDEXED:
            unusable.append(document_id)
    if unusable:
        raise api_error(
            400, "documents_not_searchable",
            f"These documents do not exist or are not indexed: {', '.join(unusable)}",
        )
