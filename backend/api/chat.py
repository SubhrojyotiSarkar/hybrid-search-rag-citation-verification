"""Question answering endpoint."""

import logging

from fastapi import APIRouter, Depends

from backend.api.dependencies import api_error, get_answer_generator, get_repository, validate_document_ids
from backend.generation.answer_generator import AnswerGenerator
from backend.generation.llm_client import LLMError
from backend.schemas import AskRequest, AskResponse
from backend.storage.repositories import DocumentRepository

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/chat", tags=["chat"])

# HTTP status per LLMError code; anything unlisted is a 502 (bad response from upstream).
LLM_ERROR_STATUS = {
    "llm_rate_limited": 429,
    "llm_unavailable": 503,
    "llm_auth_failed": 503,
    "llm_permission_denied": 503,
    "llm_model_not_found": 503,
    "llm_refused": 422,
}


@router.post("/ask", response_model=AskResponse)
def ask(
    request: AskRequest,
    generator: AnswerGenerator = Depends(get_answer_generator),
    repository: DocumentRepository = Depends(get_repository),
) -> AskResponse:
    validate_document_ids(request.document_ids, repository)
    try:
        return generator.answer(request.question, request.document_ids)
    except LLMError as exc:
        logger.warning("Answer generation failed: %s (%s)", exc.code, exc.message)
        raise api_error(LLM_ERROR_STATUS.get(exc.code, 502), exc.code, exc.message) from exc
    except Exception as exc:
        logger.exception("Unexpected error while answering")
        raise api_error(500, "internal_error", "Unexpected error while answering the question.") from exc
