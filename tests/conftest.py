import re

import pymupdf
import pytest
from fastapi.testclient import TestClient

from backend.config import get_settings

FAKE_API_KEY = "sk-test-not-a-real-key"


class WhitespaceTokenizer:
    """Deterministic test tokenizer: one token per whitespace-separated word."""

    def token_offsets(self, text: str) -> list[tuple[int, int]]:
        return [(m.start(), m.end()) for m in re.finditer(r"\S+", text)]


def make_pdf(pages: list[str]) -> bytes:
    """Build an in-memory PDF with one page per string ('' gives a blank page)."""
    document = pymupdf.open()
    for text in pages:
        page = document.new_page()
        if text:
            page.insert_textbox(pymupdf.Rect(50, 50, 550, 800), text, fontsize=10)
    data = document.tobytes()
    document.close()
    return data


@pytest.fixture
def client(tmp_path, monkeypatch):
    """A TestClient with an isolated temp data directory and the whitespace tokenizer."""
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("ANTHROPIC_API_KEY", FAKE_API_KEY)
    get_settings.cache_clear()

    from backend.main import app

    with TestClient(app) as test_client:
        # Swap the real (downloaded) tokenizer for a fast offline one.
        app.state.tokenizer = WhitespaceTokenizer()
        yield test_client
    get_settings.cache_clear()
