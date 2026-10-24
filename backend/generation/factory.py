"""Builds the answer pipeline from settings, so the API and the evaluation runner use the
exact same configuration (retriever, verifier, recovery policy)."""

from backend.config import Settings
from backend.generation.answer_generator import AnswerGenerator, RecoveryConfig
from backend.generation.llm_client import LLMClient
from backend.retrieval.hybrid_search import RetrievalServices, build_retriever
from backend.storage.repositories import DocumentRepository
from backend.verification.citation_verifier import CitationVerifier


def build_answer_generator(
    settings: Settings, services: RetrievalServices, repository: DocumentRepository, llm: LLMClient
) -> AnswerGenerator:
    verifier = CitationVerifier(llm, settings.verification_max_claims_per_call) if settings.verification_enabled else None
    recovery = RecoveryConfig(
        max_attempts=settings.max_recovery_attempts,
        recovery_top_k=settings.recovery_top_k,
        policy=settings.unsupported_claim_policy,
    )
    retriever = build_retriever(settings.retrieval_mode, services, repository, settings)
    return AnswerGenerator(retriever, llm, settings.retrieval_top_k, verifier, recovery)
