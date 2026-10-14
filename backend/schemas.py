"""Shared Pydantic contracts between the backend modules, the API, and the frontend.

Keeping every request/response shape in one module prevents the API, the pipeline and
the UI from drifting apart. Citation and claim models are added here in later steps.
"""

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, Field, field_validator

MAX_QUERY_CHARS = 2000


def _strip_nonblank(value: str) -> str:
    value = value.strip()
    if not value:
        raise ValueError("must not be blank")
    return value


class HealthResponse(BaseModel):
    status: str  # "ok", or "degraded" when retrieval is unavailable
    app_version: str
    embedding_model: str
    retrieval_ready: bool
    retrieval_mode: str
    indexed_vectors: int
    bm25_chunks: int
    reranker_model: str
    reranker_loaded: bool  # False until the first reranked query (lazy loading)
    llm_provider: str
    llm_model: str
    llm_configured: bool


class DocumentStatus(StrEnum):
    PROCESSING = "processing"  # ingestion started
    CHUNKED = "chunked"  # text extracted and chunked, not yet searchable
    INDEXED = "indexed"  # embeddings stored; available for retrieval
    FAILED = "failed"  # see error_code / error_message


class PageText(BaseModel):
    page_number: int = Field(ge=1)  # physical position in the file, 1-based
    # Printed page number from the PDF's page labels (e.g. "186" in a journal article
    # whose first page is 183, or "iv" in front matter). None if the PDF defines none.
    page_label: str | None = None
    text: str
    is_empty: bool


class Chunk(BaseModel):
    chunk_id: str
    document_id: str
    filename: str
    page_number: int = Field(ge=1)  # physical page; used to open the PDF at the right place
    page_label: str | None = None  # printed page number; preferred for display in citations
    chunk_index: int = Field(ge=0)  # position within the document
    start_char: int = Field(ge=0)  # character span within the page's extracted text
    end_char: int = Field(gt=0)
    token_count: int = Field(gt=0)
    text: str
    # Caption + column headers of the table this chunk continues (None otherwise). Used for
    # indexing and in the LLM prompt; never part of the cited text or snippet.
    context: str | None = None


def index_text(context: str | None, text: str) -> str:
    """The text that is embedded and BM25-indexed for a chunk: context first, then text."""
    return f"{context}\n{text}" if context else text


class DocumentRecord(BaseModel):
    document_id: str
    filename: str
    sha256: str
    size_bytes: int
    status: DocumentStatus
    page_count: int | None = None
    empty_pages: list[int] = Field(default_factory=list)
    chunk_count: int = 0
    error_code: str | None = None
    error_message: str | None = None
    metadata: dict[str, str] = Field(default_factory=dict)
    created_at: str
    updated_at: str


RetrievalMode = Literal["dense", "bm25", "hybrid", "hybrid_rerank"]


class RetrievedChunk(BaseModel):
    chunk: Chunk
    # Retriever-specific: cosine similarity (dense), BM25 score (bm25), RRF score (hybrid),
    # cross-encoder logit (hybrid_rerank; unbounded, higher = more relevant).
    score: float
    rank: int = Field(ge=1)
    retriever: str
    # Hybrid: each component retriever's rank and score for this chunk (a retriever that did
    # not return it is absent). Reranked results also include the pre-rerank "hybrid" entry.
    component_ranks: dict[str, int] = Field(default_factory=dict)
    component_scores: dict[str, float] = Field(default_factory=dict)


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=MAX_QUERY_CHARS)
    document_ids: list[str] | None = None  # None searches every indexed document
    top_k: int | None = Field(default=None, ge=1, le=50)
    retriever: RetrievalMode | None = None  # None uses RETRIEVAL_MODE

    @field_validator("query")
    @classmethod
    def _strip_query(cls, value: str) -> str:
        return _strip_nonblank(value)


class SearchResponse(BaseModel):
    query: str
    retriever: str
    top_k: int
    results: list[RetrievedChunk]
    latency_ms: float


# ---- Answer generation ------------------------------------------------------------------
# GeneratedClaim / GeneratedAnswer are what the LLM must return (its JSON schema). They are
# kept deliberately simple: everything else is computed and validated by our own code.

AnswerStatus = Literal["answered", "partially_answered", "insufficient_evidence"]


class GeneratedClaim(BaseModel):
    text: str = Field(description="One self-contained factual statement, checkable on its own.")
    source_labels: list[str] = Field(
        description='Labels of the passages that directly support this claim, e.g. ["S1", "S3"].'
    )


class GeneratedAnswer(BaseModel):
    answer_status: AnswerStatus
    claims: list[GeneratedClaim]
    missing_information: str = Field(
        description="What the question asks that the sources do not cover; empty string if nothing."
    )


class VerificationStatus(StrEnum):
    UNVERIFIED = "unverified"  # no verifier has checked this claim (Phase 1)
    SUPPORTED = "supported"
    CONTRADICTED = "contradicted"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    VERIFICATION_FAILED = "verification_failed"  # verifier error: NOT the same as unsupported


class OverallStatus(StrEnum):
    """Deliberately distinct states: only the verifier (Phase 4) can produce VERIFIED."""

    UNVERIFIED = "unverified"  # generated with citations, not yet checked
    VERIFIED = "verified_within_provided_evidence"
    UNRESOLVED_CLAIMS = "contains_unresolved_claims"
    VERIFICATION_FAILED = "verification_failed"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"


class Citation(BaseModel):
    source_label: str
    chunk_id: str
    document_id: str
    filename: str
    page_number: int
    page_label: str | None = None
    snippet: str


class Claim(BaseModel):
    claim_id: str
    text: str
    citations: list[Citation]
    # Labels the model cited that were never provided to it (hallucinated citations).
    invalid_source_labels: list[str] = Field(default_factory=list)
    verification_status: VerificationStatus = VerificationStatus.UNVERIFIED
    verification_reason: str | None = None


class SourcePassage(BaseModel):
    """A passage that was given to the LLM as evidence, for the source viewer."""

    source_label: str
    chunk: Chunk
    retrieval_score: float
    retrieval_rank: int


class ProcessingMetadata(BaseModel):
    retriever: str
    top_k: int
    llm_model: str | None = None
    retrieval_ms: float
    generation_ms: float | None = None
    total_ms: float
    input_tokens: int | None = None
    output_tokens: int | None = None


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=MAX_QUERY_CHARS)
    document_ids: list[str] | None = None  # None searches every indexed document

    @field_validator("question")
    @classmethod
    def _strip_question(cls, value: str) -> str:
        return _strip_nonblank(value)


class AskResponse(BaseModel):
    question: str
    answer: str
    answer_status: AnswerStatus
    missing_information: str
    claims: list[Claim]
    sources: list[SourcePassage]
    overall_status: OverallStatus
    metadata: ProcessingMetadata
