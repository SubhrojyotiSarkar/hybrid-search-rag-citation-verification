"""Retrieval-only endpoint: inspect what a retriever returns, without calling the LLM.

Used for debugging in the UI, and by the retrieval experiments that compare dense, BM25
and hybrid retrieval.
"""

import time
from collections.abc import Callable

from fastapi import APIRouter, Depends

from backend.api.dependencies import get_repository, get_retriever_factory, validate_document_ids
from backend.config import get_settings
from backend.retrieval.hybrid_search import Retriever
from backend.schemas import SearchRequest, SearchResponse
from backend.storage.repositories import DocumentRepository

router = APIRouter(prefix="/api/retrieval", tags=["retrieval"])


@router.post("/search", response_model=SearchResponse)
def search(
    request: SearchRequest,
    retriever_factory: Callable[[str], Retriever] = Depends(get_retriever_factory),
    repository: DocumentRepository = Depends(get_repository),
) -> SearchResponse:
    validate_document_ids(request.document_ids, repository)
    settings = get_settings()
    retriever = retriever_factory(request.retriever or settings.retrieval_mode)
    top_k = request.top_k or settings.retrieval_top_k

    started = time.perf_counter()
    results = retriever.search(request.query, top_k, request.document_ids)
    latency_ms = (time.perf_counter() - started) * 1000

    return SearchResponse(
        query=request.query, retriever=retriever.name, top_k=top_k,
        results=results, latency_ms=round(latency_ms, 2),
    )
