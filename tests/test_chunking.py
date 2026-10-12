import pytest

from backend.ingestion.chunker import chunk_pages, split_into_spans
from backend.schemas import PageText
from tests.conftest import WhitespaceTokenizer

TOKENIZER = WhitespaceTokenizer()


def numbered_words(count: int) -> str:
    return " ".join(f"w{i}" for i in range(count))


def test_empty_text_produces_no_spans():
    assert split_into_spans("   ", TOKENIZER, chunk_size=10, overlap=2) == []


def test_short_text_is_a_single_chunk():
    text = "Regularization reduces overfitting."
    spans = split_into_spans(text, TOKENIZER, chunk_size=10, overlap=2)

    assert len(spans) == 1
    assert text[spans[0].start_char : spans[0].end_char] == text
    assert spans[0].token_count == 3


def test_long_text_windows_respect_size_and_overlap():
    text = numbered_words(25)
    spans = split_into_spans(text, TOKENIZER, chunk_size=10, overlap=3)
    chunk_words = [text[s.start_char : s.end_char].split() for s in spans]

    assert all(len(words) <= 10 for words in chunk_words)
    # Each chunk starts 7 (= 10 - 3) words after the previous one.
    assert [words[0] for words in chunk_words] == ["w0", "w7", "w14", "w21"]
    # Consecutive chunks share exactly `overlap` words.
    for previous, current in zip(chunk_words, chunk_words[1:]):
        assert previous[-3:] == current[:3]
    # Every word is covered, ending with the last one.
    covered = {word for words in chunk_words for word in words}
    assert covered == set(text.split())


def test_chunk_text_is_exact_substring_of_page():
    text = "Line one has words.\nLine two, with punctuation!\n\nNew paragraph here."
    for span in split_into_spans(text, TOKENIZER, chunk_size=4, overlap=1):
        assert text[span.start_char : span.end_char] in text
        assert not text[span.start_char : span.end_char].startswith(" ")


def test_windows_snap_to_word_starts():
    # Simulate sub-word tokens: "regularization" -> "regular" + "ization" (no gap between).
    class SubwordTokenizer:
        def token_offsets(self, text):
            return [(0, 3), (4, 11), (11, 18), (19, 24)]  # "the" "regular" "ization" "helps"

    text = "the regularization helps"
    spans = split_into_spans(text, SubwordTokenizer(), chunk_size=2, overlap=0)
    pieces = [text[s.start_char : s.end_char] for s in spans]

    # The window ending after "regular" is pulled back so the word stays whole.
    assert pieces == ["the", "regularization", "helps"]


def test_overlap_must_be_smaller_than_chunk_size():
    with pytest.raises(ValueError):
        split_into_spans("some text", TOKENIZER, chunk_size=5, overlap=5)


def test_chunk_pages_skips_empty_pages_and_keeps_metadata():
    pages = [
        PageText(page_number=1, text=numbered_words(12), is_empty=False),
        PageText(page_number=2, text="", is_empty=True),
        PageText(page_number=3, text="Final page text.", is_empty=False),
    ]
    chunks = chunk_pages(pages, "doc_abc", "notes.pdf", TOKENIZER, chunk_size=10, overlap=2)

    assert [c.page_number for c in chunks] == [1, 1, 3]
    assert [c.chunk_index for c in chunks] == [0, 1, 2]
    assert chunks[2].chunk_id == "doc_abc_p0003_c0002"
    assert len({c.chunk_id for c in chunks}) == len(chunks)
    assert all(c.document_id == "doc_abc" and c.filename == "notes.pdf" for c in chunks)
    assert chunks[2].text == "Final page text."
