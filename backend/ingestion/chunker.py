"""Split page text into overlapping, token-bounded chunks that keep their source location.

Chunks never cross page boundaries, so every chunk cites exactly one page. Chunk text is
sliced from the original page string using the tokenizer's character offsets, so the
stored text is an exact substring of the extracted page text.

Tables: when a table is split, only the first chunk holds its caption and column headers;
later chunks would be bare numbers ("GOOGL 0.06248 0.03752 ...") with no indication of
which table, dataset or column they belong to. Such chunks get the caption and headers as
`context`, which is used for indexing and prompting but is not part of the cited text.
"""

import re
import threading
from bisect import bisect_right
from dataclasses import dataclass
from typing import Protocol

from backend.schemas import Chunk, PageText

# Upper bound on the context prefix, in embedding-model tokens. Chunks on pages with
# tables are shortened by the context length so prefix + chunk still fit the model.
MAX_CONTEXT_TOKENS = 48
MAX_HEADER_LINES = 16

# "Table 7", "Table 7: ...", "Table 7. ..." on its own line; not prose like "Table 7 shows".
_CAPTION_PATTERN = re.compile(r"^(?:Table|TABLE)\s+\d+[A-Za-z]?(?:\s*$|\s*[:.—–-]\s*)")
# A table cell that is only a number (allows signs, decimals, %, scientific notation).
_NUMERIC_LINE = re.compile(r"^[-+−]?[\d.,]*\d[\d.,]*(?:[eE][-+]?\d+)?%?$")


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
    """Chunk every non-empty page; chunk_index counts across the whole document.

    Chunks that continue a table (start inside it without containing its caption) get the
    table's caption and header row as `context`.
    """
    chunks: list[Chunk] = []
    for page in pages:
        if page.is_empty:
            continue
        regions = [
            TableRegion(r.start_char, r.end_char, r.caption_line, truncate_to_tokens(r.context, tokenizer, MAX_CONTEXT_TOKENS))
            for r in find_table_regions(page.text)
        ]
        # Reserve room for the longest context on this page, so context + chunk never
        # exceeds chunk_size tokens (the embedding model's input budget).
        reserve = max((len(tokenizer.token_offsets(r.context)) + 1 for r in regions), default=0)
        page_chunk_size = max(chunk_size - reserve, 2)
        page_overlap = min(overlap, page_chunk_size - 1)

        for span in split_into_spans(page.text, tokenizer, page_chunk_size, page_overlap):
            text = page.text[span.start_char : span.end_char]
            region = next((r for r in regions if r.start_char <= span.start_char < r.end_char), None)
            context = region.context if region and region.caption_line not in text else None
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
                    text=text,
                    context=context,
                )
            )
    return chunks


@dataclass(frozen=True)
class TableRegion:
    start_char: int  # start of the caption line
    end_char: int  # where the table ends: next caption, prose, or end of page
    caption_line: str
    context: str  # caption + column headers, e.g. "Table 7 SVR results ... Stock Linear kernel RMSE MAPE"


def _is_numeric(line: str) -> bool:
    return bool(_NUMERIC_LINE.match(line.replace(" ", "")))


def _is_prose(line: str) -> bool:
    """A full sentence-like line, i.e. body text after the table rather than a cell."""
    return len(line) >= 60 and len(line.split()) >= 8 and not line.lower().startswith("note")


def find_table_regions(text: str) -> list[TableRegion]:
    """Locate tables by caption line and collect their caption + header lines.

    Header lines are the non-numeric lines after the caption, up to (not including) the
    first row label, recognised as a non-numeric line immediately followed by a number.
    """
    lines: list[tuple[int, str]] = []  # (start offset, stripped line)
    offset = 0
    for raw in text.splitlines(keepends=True):
        if raw.strip():
            lines.append((offset + len(raw) - len(raw.lstrip()), raw.strip()))
        offset += len(raw)

    caption_indexes = [i for i, (_, line) in enumerate(lines) if _CAPTION_PATTERN.match(line)]
    regions: list[TableRegion] = []
    for position, i in enumerate(caption_indexes):
        next_caption = caption_indexes[position + 1] if position + 1 < len(caption_indexes) else len(lines)
        header = [lines[i][1]]
        j = i + 1
        while j < next_caption and len(header) < MAX_HEADER_LINES:
            line = lines[j][1]
            following = lines[j + 1][1] if j + 1 < len(lines) else ""
            if _is_numeric(line) or (len(header) > 1 and _is_numeric(following)):
                break
            header.append(line)
            j += 1
        end_index = next((k for k in range(j, next_caption) if _is_prose(lines[k][1])), next_caption)
        end_char = lines[end_index][0] if end_index < len(lines) else len(text)
        regions.append(TableRegion(lines[i][0], end_char, lines[i][1], " ".join(header)))
    return regions


def truncate_to_tokens(text: str, tokenizer: Tokenizer, max_tokens: int) -> str:
    offsets = tokenizer.token_offsets(text)
    if len(offsets) <= max_tokens:
        return text
    return text[: offsets[max_tokens - 1][1]]
