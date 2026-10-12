"""Shared Pydantic contracts between the backend modules, the API, and the frontend.

Keeping every request/response shape in one module prevents the API, the pipeline and
the UI from drifting apart. Citation and claim models are added here in later steps.
"""

from enum import StrEnum

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str
    app_version: str
    embedding_model: str
    llm_provider: str
    llm_model: str
    llm_configured: bool


class DocumentStatus(StrEnum):
    PROCESSING = "processing"  # ingestion started
    CHUNKED = "chunked"  # text extracted and chunked, not yet searchable
    INDEXED = "indexed"  # embeddings stored; available for retrieval
    FAILED = "failed"  # see error_code / error_message


class PageText(BaseModel):
    page_number: int = Field(ge=1)  # 1-based, as shown in PDF viewers
    text: str
    is_empty: bool


class Chunk(BaseModel):
    chunk_id: str
    document_id: str
    filename: str
    page_number: int = Field(ge=1)
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
