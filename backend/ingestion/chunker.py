"""Split page text into overlapping, token-bounded chunks that keep their source location.

Chunks never cross page boundaries, so every chunk cites exactly one page. Chunk text is
sliced from the original page string using the tokenizer's character offsets, so the
stored text is an exact substring of the extracted page text.
"""

import threading
from bisect import bisect_right
from dataclasses import dataclass
from typing import Protocol

from backend.schemas import Chunk, PageText


class Tokenizer(Protocol):
    def token_offsets(self, text: str) -> list[tuple[int, int]]:
        """Return the (start_char, end_char) span of every token, excluding special tokens."""
        ...


class HuggingFaceTokenizer:
    """The embedding model's tokenizer, so chunk sizes are measured in the units it reads.

    Loaded lazily on first use: constructing it does not download anything.
    """

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self._tokenizer = None
        self._lock = threading.Lock()

    def _load(self):
        if self._tokenizer is None:
            with self._lock:
                if self._tokenizer is None:
                    from transformers import AutoTokenizer

                    self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        return self._tokenizer

    def token_offsets(self, text: str) -> list[tuple[int, int]]:
        # verbose=False silences the "sequence longer than max length" warning: we only
        # need offsets here, and every chunk is kept below the model's limit.
        encoding = self._load()(
            text, add_special_tokens=False, return_offsets_mapping=True, verbose=False
        )
        return [(start, end) for start, end in encoding["offset_mapping"] if end > start]


@dataclass(frozen=True)
class TextSpan:
    start_char: int
    end_char: int
    token_count: int


def split_into_spans(
    text: str, tokenizer: Tokenizer, chunk_size: int, overlap: int
) -> list[TextSpan]:
    """Slide a window of `chunk_size` tokens over the text, stepping by chunk_size - overlap.

    Window edges are snapped back to the start of a word so words are never cut in half;
    this can make a chunk slightly shorter, or an overlap slightly longer, than configured.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if not 0 <= overlap < chunk_size:
        raise ValueError("overlap must be >= 0 and smaller than chunk_size")

    offsets = tokenizer.token_offsets(text)
    n_tokens = len(offsets)
    if n_tokens == 0:
        return []

    # A token starts a new word when there is a gap (whitespace) before it. Sub-word pieces
    # and attached punctuation follow the previous token directly.
    word_starts = [0] + [i for i in range(1, n_tokens) if offsets[i][0] > offsets[i - 1][1]]

    def snap_back(position: int, floor: int) -> int:
        """Largest word start <= position and > floor; position itself if there is none."""
        idx = bisect_right(word_starts, position) - 1
        if idx >= 0 and word_starts[idx] > floor:
            return word_starts[idx]
        return position

    spans: list[TextSpan] = []
    start = 0
    while True:
        end = min(start + chunk_size, n_tokens)
        if end < n_tokens:
            end = snap_back(end, start)
        spans.append(TextSpan(offsets[start][0], offsets[end - 1][1], end - start))
        if end == n_tokens:
            break
        # max(..., start + 1) guarantees progress, so the loop always terminates.
        start = snap_back(max(end - overlap, start + 1), start)
    return spans


def make_chunk_id(document_id: str, page_number: int, chunk_index: int) -> str:
    return f"{document_id}_p{page_number:04d}_c{chunk_index:04d}"


def chunk_pages(
    pages: list[PageText],
    document_id: str,
    filename: str,
    tokenizer: Tokenizer,
    chunk_size: int,
    overlap: int,
) -> list[Chunk]:
    """Chunk every non-empty page; chunk_index counts across the whole document."""
    chunks: list[Chunk] = []
    for page in pages:
        if page.is_empty:
            continue
        for span in split_into_spans(page.text, tokenizer, chunk_size, overlap):
            index = len(chunks)
            chunks.append(
                Chunk(
                    chunk_id=make_chunk_id(document_id, page.page_number, index),
                    document_id=document_id,
                    filename=filename,
                    page_number=page.page_number,
                    page_label=page.page_label,
                    chunk_index=index,
                    start_char=span.start_char,
                    end_char=span.end_char,
                    token_count=span.token_count,
                    text=page.text[span.start_char : span.end_char],
                )
            )
    return chunks
