import pytest

from backend.ingestion.chunker import chunk_pages, find_table_regions, split_into_spans
from backend.schemas import PageText, index_text
from tests.conftest import WhitespaceTokenizer

# Mirrors how PyMuPDF extracts the paper's Table 7: one cell per line, caption first.
TABLE_7_PAGE = "\n".join(
    ["Results are discussed below for every stock in the sample considered here.",
     "Table 7",
     "SVR results, using the optimal parameter in Table 4 on the daily price test data.",
     "Stock", "Linear kernel", "Radial kernel", "Polynomial kernel",
     "RMSE", "MAPE", "RMSE", "MAPE", "RMSE", "MAPE"]
    + [cell for stock in ["BBAS3", "PETR4", "VALE5", "ALPA4", "DIRR3", "LEVE3", "BAC"]
       for cell in [stock, "0.16427", "0.20141", "1.13716", "1.06158", "19.1481", "32.3253"]]
    + ["GOOGL", "0.06248", "0.03752", "1.97066", "1.30933", "1.18256", "0.72193",
       "XOM", "0.07517", "0.06040", "1.08478", "0.87865", "6.87381", "4.13189",
       "The comparison of errors produced by the SVR and random walk models shows a clear pattern here."]
)

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


def test_table_region_has_caption_and_headers_but_no_rows():
    [region] = find_table_regions(TABLE_7_PAGE)

    assert region.caption_line == "Table 7"
    assert region.context == (
        "Table 7 SVR results, using the optimal parameter in Table 4 on the daily price test data. "
        "Stock Linear kernel Radial kernel Polynomial kernel RMSE MAPE RMSE MAPE RMSE MAPE"
    )
    # The table starts at its caption and ends where body text resumes.
    assert TABLE_7_PAGE[region.start_char:].startswith("Table 7")
    assert TABLE_7_PAGE[region.end_char:].startswith("The comparison of errors")


def test_prose_mentioning_a_table_is_not_a_caption():
    assert find_table_regions("As Table 7 shows, the linear kernel performs best on most stocks.") == []


def test_caption_with_colon_on_one_line_is_detected():
    text = "Table 3: Error rates per model\nModel\nError\nSVR\n0.12\nRF\n0.15"
    [region] = find_table_regions(text)

    assert region.context == "Table 3: Error rates per model Model Error"


def test_chunks_continuing_a_table_get_its_context():
    # Regression for "GOOGL linear kernel RMSE 0.06248 MAPE 0.03752": the GOOGL row used to
    # land in a chunk with no caption or headers, so nothing said it was Table 7.
    page = PageText(page_number=12, page_label="194", text=TABLE_7_PAGE, is_empty=False)
    chunks = chunk_pages([page], "doc_t", "paper.pdf", TOKENIZER, chunk_size=60, overlap=5)

    [googl] = [c for c in chunks if "GOOGL\n0.06248\n0.03752" in c.text]
    assert "Table 7" not in googl.text
    assert googl.context.startswith("Table 7 SVR results")
    assert "Linear kernel" in googl.context and "RMSE" in googl.context
    # The chunk that already contains the caption gets no duplicate context.
    [first] = [c for c in chunks if "Table 7" in c.text]
    assert first.context is None
    # Body text after the table is not labelled as part of it.
    [after] = [c for c in chunks if c.text.startswith("The comparison")] or [None]
    assert after is None or after.context is None


def test_context_plus_chunk_fits_the_token_budget():
    page = PageText(page_number=1, text=TABLE_7_PAGE, is_empty=False)
    chunks = chunk_pages([page], "doc_t", "paper.pdf", TOKENIZER, chunk_size=60, overlap=5)

    for chunk in chunks:
        assert len(TOKENIZER.token_offsets(index_text(chunk.context, chunk.text))) <= 60
    # The stored text is still an exact slice of the page.
    assert all(TABLE_7_PAGE[c.start_char:c.end_char] == c.text for c in chunks)


def test_pages_without_tables_are_chunked_as_before():
    text = numbered_words(25)
    page = PageText(page_number=1, text=text, is_empty=False)
    chunks = chunk_pages([page], "doc_t", "x.pdf", TOKENIZER, chunk_size=10, overlap=3)

    assert [c.text.split()[0] for c in chunks] == ["w0", "w7", "w14", "w21"]
    assert all(c.context is None for c in chunks)


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
