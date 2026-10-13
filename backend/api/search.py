"""Retrieval-only endpoint: inspect what the retriever returns, without calling the LLM.

Used for debugging in the UI now, and by the retrieval experiments in Phase 2-3.
"""

import time

from fastapi import APIRouter, Depends

from backend.api.dependencies import api_error, get_dense_retriever, get_repository
from backend.config import get_settings
from backend.retrieval.dense_retriever import DenseRetriever
from backend.schemas import DocumentStatus, SearchRequest, SearchResponse
from backend.storage.repositories import DocumentRepository

router = APIRouter(prefix="/api/retrieval", tags=["retrieval"])


def validate_document_ids(document_ids: list[str] | None, repository: DocumentRepository) -> None:
    """Reject unknown or not-yet-searchable documents explicitly instead of ignoring them."""
    if document_ids is None:
        return
    if not document_ids:
        raise api_error(400, "no_documents_selected", "Select at least one document to search.")
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


@router.post("/search", response_model=SearchResponse)
def search(
    request: SearchRequest,
    retriever: DenseRetriever = Depends(get_dense_retriever),
    repository: DocumentRepository = Depends(get_repository),
) -> SearchResponse:
    validate_document_ids(request.document_ids, repository)
    top_k = request.top_k or get_settings().dense_top_k

    started = time.perf_counter()
    results = retriever.search(request.query, top_k, request.document_ids)
    latency_ms = (time.perf_counter() - started) * 1000

    return SearchResponse(
        query=request.query, retriever=retriever.name, top_k=top_k,
        results=results, latency_ms=round(latency_ms, 2),
    )
