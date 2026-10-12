"""Page-by-page text extraction with PyMuPDF.

Extraction is deliberately conservative: text is whitespace-normalized but otherwise kept
as extracted, so citation snippets match what the user sees in the PDF.
"""

import logging
import re
from dataclasses import dataclass

import pymupdf

from backend.schemas import PageText

logger = logging.getLogger(__name__)


class PDFExtractionError(Exception):
    """The PDF could not be read at all (corrupt, encrypted, or no pages)."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


@dataclass
class ExtractionResult:
    page_count: int
    pages: list[PageText]
    metadata: dict[str, str]

    @property
    def empty_page_numbers(self) -> list[int]:
        return [page.page_number for page in self.pages if page.is_empty]


def normalize_text(text: str) -> str:
    """Collapse runs of spaces and blank lines; drop soft hyphens and NUL characters."""
    text = text.replace("­", "").replace("\x00", "")
    text = re.sub(r"[ \t\f\v ]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_pdf(content: bytes, min_page_chars: int = 20) -> ExtractionResult:
    """Extract normalized text for every page and flag pages with no usable text."""
    try:
        document = pymupdf.open(stream=content, filetype="pdf")
    except Exception as exc:
        raise PDFExtractionError("unreadable_pdf", "The file could not be opened as a PDF.") from exc

    with document:
        if document.needs_pass:
            raise PDFExtractionError("encrypted_pdf", "The PDF is password-protected.")
        if document.page_count == 0:
            raise PDFExtractionError("empty_pdf", "The PDF contains no pages.")

        pages: list[PageText] = []
        for index, page in enumerate(document):
            try:
                raw_text = page.get_text("text")
            except Exception:
                # One damaged page should not discard the rest of the document.
                logger.warning("Text extraction failed on page %d", index + 1, exc_info=True)
                raw_text = ""
            text = normalize_text(raw_text)
            visible_chars = len(re.sub(r"\s", "", text))
            pages.append(
                PageText(page_number=index + 1, text=text, is_empty=visible_chars < min_page_chars)
            )

        metadata = {key: str(value) for key, value in (document.metadata or {}).items() if value}
        return ExtractionResult(page_count=document.page_count, pages=pages, metadata=metadata)
