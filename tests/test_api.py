from backend.config import get_settings
from tests.conftest import FAKE_API_KEY


def test_health_reports_configuration(client):
    response = client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["embedding_model"] == "sentence-transformers/all-MiniLM-L6-v2"
    assert body["llm_configured"] is True


def test_health_never_exposes_api_key(client):
    response = client.get("/health")

    assert FAKE_API_KEY not in response.text


def test_startup_creates_data_directories(client):
    settings = get_settings()

    assert settings.upload_dir.is_dir()
    assert settings.index_dir.is_dir()
