import hashlib
import re

import numpy as np
import pymupdf
import pytest
from fastapi.testclient import TestClient

from backend.config import get_settings

FAKE_API_KEY = "sk-test-not-a-real-key"


class WhitespaceTokenizer:
    """Deterministic test tokenizer: one token per whitespace-separated word."""

    def token_offsets(self, text: str) -> list[tuple[int, int]]:
        return [(m.start(), m.end()) for m in re.finditer(r"\S+", text)]


class FakeEmbedder:
    """Deterministic bag-of-words embedder: texts sharing words get similar vectors.

    Uses md5 (not Python's hash(), which is randomized per process) so vectors are
    identical across "restarts" within a test.
    """

    model_name = "fake-bag-of-words"
    dimension = 64

    def encode(self, texts: list[str]) -> np.ndarray:
        vectors = np.zeros((len(texts), self.dimension), dtype=np.float32)
        for row, text in enumerate(texts):
            for word in re.findall(r"[a-z0-9]+", text.lower()):
                bucket = int(hashlib.md5(word.encode()).hexdigest(), 16) % self.dimension
                vectors[row, bucket] += 1.0
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        return vectors / np.where(norms == 0, 1.0, norms)


def make_pdf(pages: list[str], first_page_label: int | None = None) -> bytes:
    """Build an in-memory PDF with one page per string ('' gives a blank page).

    first_page_label adds printed page labels starting at that number, like a journal
    article extracted from an issue (physical page 1 printed as e.g. 183).
    """
    document = pymupdf.open()
    for text in pages:
        page = document.new_page()
        if text:
            page.insert_textbox(pymupdf.Rect(50, 50, 550, 800), text, fontsize=10)
    if first_page_label is not None:
        document.set_page_labels([{"startpage": 0, "style": "D", "firstpagenum": first_page_label}])
    data = document.tobytes()
    document.close()
    return data


@pytest.fixture
def app_factory(tmp_path, monkeypatch):
    """Build apps that share one temp data directory, so a test can simulate a restart."""
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("ANTHROPIC_API_KEY", FAKE_API_KEY)
    get_settings.cache_clear()

    from backend.main import create_app

    def factory():
        return create_app(embedder=FakeEmbedder(), tokenizer=WhitespaceTokenizer())

    yield factory
    get_settings.cache_clear()


@pytest.fixture
def client(app_factory):
    with TestClient(app_factory()) as test_client:
        yield test_client
