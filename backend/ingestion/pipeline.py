"""Ingestion orchestration: validate -> store -> extract -> chunk -> persist.

Every failure after the document record exists is recorded on that record (status FAILED
with a reason code), so problems are visible in the document list rather than silent.
"""

import hashlib
import logging
import uuid
from pathlib import Path

from backend.config import Settings
from backend.ingestion.chunker import Tokenizer, chunk_pages
from backend.ingestion.pdf_loader import PDFExtractionError, extract_pdf
from backend.schemas import DocumentRecord
from backend.storage.repositories import DocumentRepository, DuplicateHashError

logger = logging.getLogger(__name__)

MAX_FILENAME_LENGTH = 255


class IngestionError(Exception):
    """An ingestion failure that should be reported to the user. Carries an HTTP status."""

    status_code = 400

    def __init__(self, code: str, message: str, document_id: str | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.document_id = document_id


class InvalidUploadError(IngestionError):
    status_code = 400


class UploadTooLargeError(IngestionError):
    status_code = 413


class DuplicateDocumentError(IngestionError):
    status_code = 409


class ExtractionFailedError(IngestionError):
    status_code = 422


def new_document_id() -> str:
    return f"doc_{uuid.uuid4().hex[:12]}"


class IngestionPipeline:
    def __init__(self, settings: Settings, repository: DocumentRepository, tokenizer: Tokenizer) -> None:
        self.settings = settings
        self.repository = repository
        self.tokenizer = tokenizer

    def file_path(self, document_id: str) -> Path:
        # Stored under the generated ID, never the user's filename (avoids path traversal).
        return self.settings.upload_dir / f"{document_id}.pdf"

    def ingest(self, filename: str, content: bytes) -> DocumentRecord:
        filename = self._validate(filename, content)
        sha256 = hashlib.sha256(content).hexdigest()

        existing = self.repository.get_by_sha256(sha256)
        if existing:
            raise self._duplicate_error(existing.document_id, existing.filename)

        document_id = new_document_id()
        try:
            self.repository.create(document_id, filename, sha256, len(content))
        except DuplicateHashError:
            # Another upload of the same file won the race between the check and the insert.
            existing = self.repository.get_by_sha256(sha256)
            raise self._duplicate_error(existing.document_id if existing else None, filename)

        try:
            self.file_path(document_id).write_bytes(content)
            extraction = extract_pdf(content, self.settings.min_page_chars)

            if all(page.is_empty for page in extraction.pages):
                message = (
                    f"No extractable text found on any of the {extraction.page_count} page(s). "
                    "The PDF may be scanned images, which need OCR (not supported yet)."
                )
                self.repository.mark_failed(
                    document_id, "no_extractable_text", message,
                    extraction.page_count, extraction.empty_page_numbers,
                )
                raise ExtractionFailedError("no_extractable_text", message, document_id)

            chunks = chunk_pages(
                extraction.pages, document_id, filename, self.tokenizer,
                self.settings.chunk_size_tokens, self.settings.chunk_overlap_tokens,
            )
            self.repository.save_chunks(
                document_id, chunks, extraction.page_count,
                extraction.empty_page_numbers, extraction.metadata,
            )
        except PDFExtractionError as exc:
            self.repository.mark_failed(document_id, exc.code, exc.message)
            raise ExtractionFailedError(exc.code, exc.message, document_id) from exc
        except IngestionError:
            raise
        except Exception:
            logger.exception("Unexpected ingestion failure for %s", document_id)
            self.repository.mark_failed(
                document_id, "internal_error", "Unexpected error while processing the document."
            )
            raise

        logger.info("Ingested %s (%s): %d chunks", document_id, filename, len(chunks))
        return self.repository.get(document_id)

    def delete(self, document_id: str) -> bool:
        """Remove a document, its chunks and its stored file. Returns False if it did not exist."""
        deleted = self.repository.delete(document_id)
        if deleted:
            self.file_path(document_id).unlink(missing_ok=True)
        return deleted

    def _validate(self, filename: str, content: bytes) -> str:
        """Check name, size and PDF signature; return the filename stripped of any path."""
        name = Path(filename.replace("\\", "/")).name.strip()
        if not name or len(name) > MAX_FILENAME_LENGTH:
            raise InvalidUploadError("invalid_filename", "The file needs a valid name.")
        if not name.lower().endswith(".pdf"):
            raise InvalidUploadError("not_a_pdf", "Only PDF files are supported.")
        if len(content) > self.settings.max_upload_bytes:
            raise UploadTooLargeError(
                "file_too_large", f"The file exceeds the {self.settings.max_upload_mb} MB limit."
            )
        if not content:
            raise InvalidUploadError("empty_file", "The uploaded file is empty.")
        if not content.startswith(b"%PDF-"):
            raise InvalidUploadError("not_a_pdf", "The file does not look like a PDF.")
        return name

    @staticmethod
    def _duplicate_error(document_id: str | None, filename: str) -> DuplicateDocumentError:
        return DuplicateDocumentError(
            "duplicate_document",
            f"This file has already been uploaded as '{filename}'. Delete it first to re-ingest.",
            document_id,
        )
