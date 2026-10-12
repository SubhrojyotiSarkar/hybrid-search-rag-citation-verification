import pytest
from fastapi.testclient import TestClient

from backend.config import get_settings

FAKE_API_KEY = "sk-test-not-a-real-key"


@pytest.fixture
def client(tmp_path, monkeypatch):
    """A TestClient whose data directory is an isolated temp folder (never the real data/)."""
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("ANTHROPIC_API_KEY", FAKE_API_KEY)
    get_settings.cache_clear()

    from backend.main import app

    with TestClient(app) as test_client:
        yield test_client
    get_settings.cache_clear()
