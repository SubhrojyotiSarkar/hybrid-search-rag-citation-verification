"""Shared Pydantic contracts between the backend modules, the API, and the frontend.

Keeping every request/response shape in one module prevents the API, the pipeline and
the UI from drifting apart. Citation and claim models are added here in later steps.
"""

from enum import StrEnum

from pydantic import BaseModel, Field, field_validator

MAX_QUERY_CHARS = 2000


class HealthResponse(BaseModel):
    status: str  # "ok", or "degraded" when retrieval is unavailable
    app_version: str
    embedding_model: str
    retrieval_ready: bool
    indexed_vectors: int
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


class RetrievedChunk(BaseModel):
    chunk: Chunk
    score: float  # retriever-specific; for dense retrieval, cosine similarity
    rank: int = Field(ge=1)
    retriever: str


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=MAX_QUERY_CHARS)
    document_ids: list[str] | None = None  # None searches every indexed document
    top_k: int | None = Field(default=None, ge=1, le=50)

    @field_validator("query")
    @classmethod
    def _strip_query(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("query must not be blank")
        return value


class SearchResponse(BaseModel):
    query: str
    retriever: str
    top_k: int
    results: list[RetrievedChunk]
    latency_ms: float
