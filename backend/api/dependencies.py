"""FastAPI dependencies shared by the routers.

Long-lived services (database, tokenizer, embedder, FAISS store, BM25 index) are created
once at startup and kept on app.state; these functions hand them to request handlers.
"""

from collections.abc import Callable

from fastapi import Depends, HTTPException, Request

from backend.auth import Principal
from backend.config import get_settings
from backend.generation.answer_generator import AnswerGenerator
from backend.generation.factory import build_answer_generator
from backend.generation.llm_client import LLMClient
from backend.ingestion.pipeline import IngestionPipeline
from backend.logging_config import request_id_var
from backend.retrieval.hybrid_search import RetrievalServices, Retriever, build_retriever
from backend.schemas import DocumentStatus
from backend.storage.repositories import DocumentRepository


def api_error(status_code: int, code: str, message: str, document_id: str | None = None) -> HTTPException:
    """Errors are returned as {"detail": {"code", "message"}}; never stack traces."""
    detail = {"code": code, "message": message}
    if document_id:
        detail["document_id"] = document_id
    if status_code >= 500:
        # Lets a user report an error that can be matched to its log lines.
        detail["request_id"] = request_id_var.get()
    return HTTPException(status_code=status_code, detail=detail)


# ---- authentication -----------------------------------------------------------------------

# With login disabled (local development), every request acts as this admin.
LOCAL_ADMIN = Principal("local", "admin")


def _not_authenticated(message: str = "Log in to use this application.") -> HTTPException:
    return HTTPException(status_code=401, detail={"code": "not_authenticated", "message": message},
                         headers={"WWW-Authenticate": "Bearer"})


def _resolve(request: Request, token: str, scope: str) -> Principal:
    auth = request.app.state.auth
    principal = auth.tokens.verify(token, scope) if token else None
    # A valid token of a user who has since been removed (or whose role changed) is refused.
    if principal is None or auth.users.list_users().get(principal.username) != principal.role:
        raise _not_authenticated("Your session is invalid or has expired. Log in again.")
    request.state.principal = principal  # read by the access log
    return principal


def get_principal(request: Request) -> Principal:
    """The logged-in user (any role). Requires `Authorization: Bearer <access token>`."""
    if not request.app.state.auth.enabled:
        return LOCAL_ADMIN
    header = request.headers.get("Authorization", "")
    if not header.lower().startswith("bearer "):
        raise _not_authenticated()
    return _resolve(request, header[7:].strip(), "api")


def require_admin(principal: Principal = Depends(get_principal)) -> Principal:
    if not principal.is_admin:
        raise api_error(403, "admin_required", "This action needs an admin account.")
    return principal


def get_file_principal(request: Request) -> Principal:
    """For PDF downloads: an access token header, or a file-scoped token in ?token=.
    Browser links cannot send headers; a leaked file token can only download PDFs."""
    if not request.app.state.auth.enabled:
        return LOCAL_ADMIN
    header = request.headers.get("Authorization", "")
    if header.lower().startswith("bearer "):
        return _resolve(request, header[7:].strip(), "api")
    return _resolve(request, request.query_params.get("token", ""), "file")


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
    repository: DocumentRepository = Depends(get_repository),
    services: RetrievalServices = Depends(get_retrieval_services),
    llm: LLMClient = Depends(get_llm_client),
) -> AnswerGenerator:
    return build_answer_generator(get_settings(), services, repository, llm)


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
