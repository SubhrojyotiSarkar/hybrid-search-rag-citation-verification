import pytest

from backend.ingestion.pdf_loader import PDFExtractionError, extract_pdf, normalize_text
from tests.conftest import make_pdf


def test_extracts_text_per_page_with_one_based_numbers():
    content = make_pdf(["Page one discusses L2 regularization.", "Page two covers dropout layers."])
    result = extract_pdf(content)

    assert result.page_count == 2
    assert [p.page_number for p in result.pages] == [1, 2]
    assert "L2 regularization" in result.pages[0].text
    assert "dropout" in result.pages[1].text
    assert result.empty_page_numbers == []


def test_printed_page_labels_are_kept_alongside_physical_numbers():
    content = make_pdf(["First page of the article.", "Second page of the article."], first_page_label=183)
    result = extract_pdf(content)

    assert [(p.page_number, p.page_label) for p in result.pages] == [(1, "183"), (2, "184")]


def test_pdf_without_page_labels_has_no_label():
    result = extract_pdf(make_pdf(["Some text on the only page."]))

    assert result.pages[0].page_label is None


def test_blank_page_is_flagged_empty():
    content = make_pdf(["Some real content on the first page.", ""])
    result = extract_pdf(content)

    assert result.empty_page_numbers == [2]
    assert not result.pages[0].is_empty


def test_garbage_bytes_raise_extraction_error():
    with pytest.raises(PDFExtractionError) as excinfo:
        extract_pdf(b"%PDF-1.7 this is not really a pdf")
    assert excinfo.value.code == "unreadable_pdf"


def test_normalize_text_collapses_whitespace():
    assert normalize_text("  a  \t b \n\n\n\n c­ d ") == "a b\n\nc d"
