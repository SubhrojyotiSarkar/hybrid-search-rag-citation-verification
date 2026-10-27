"""Request IDs, access logging and the catch-all error handler."""

import logging

from fastapi.testclient import TestClient

from backend.logging_config import RequestIdFilter, new_request_id, request_id_var
from tests.conftest import FakeLLM
from tests.test_verification_flow import ML_PDF


def test_every_response_carries_a_request_id(client):
    response = client.get("/health")

    assert len(response.headers["X-Request-ID"]) == 12


def test_safe_supplied_request_id_is_kept_and_unsafe_one_replaced(client):
    assert client.get("/health", headers={"X-Request-ID": "trace-42"}).headers["X-Request-ID"] == "trace-42"
    unsafe = client.get("/health", headers={"X-Request-ID": "bad id\nwith newline"}).headers["X-Request-ID"]
    assert unsafe != "bad id\nwith newline" and len(unsafe) == 12


def test_unhandled_exception_becomes_clean_json_with_request_id(app_factory):
    app = app_factory()

    @app.get("/boom")
    def boom():
        raise RuntimeError("secret internal detail")

    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.get("/boom")

    assert response.status_code == 500
    detail = response.json()["detail"]
    assert detail["code"] == "internal_error"
    assert detail["request_id"] == response.headers["X-Request-ID"]
    assert "secret internal detail" not in response.text and "Traceback" not in response.text


def test_server_errors_from_endpoints_include_the_request_id(app_factory):
    from backend.generation.llm_client import LLMError

    llm = FakeLLM(error=LLMError("llm_unavailable", "Down.", retryable=True))
    with TestClient(app_factory(llm_client=llm)) as client:
        # A document is needed: with nothing retrieved the pipeline abstains without an LLM call.
        client.post("/api/documents/upload", files={"file": ("ml.pdf", ML_PDF, "application/pdf")})
        response = client.post("/api/chat/ask", json={"question": "What does L2 do?"}, headers={"X-Request-ID": "req-7"})

    assert response.status_code == 503
    assert response.json()["detail"]["request_id"] == "req-7"


def test_client_errors_do_not_need_a_request_id(client):
    detail = client.get("/api/documents/doc_missing").json()["detail"]

    assert "request_id" not in detail


def test_log_records_get_the_current_request_id():
    record = logging.LogRecord("x", logging.INFO, __file__, 1, "msg", None, None)
    token = request_id_var.set("abc123")
    try:
        RequestIdFilter().filter(record)
    finally:
        request_id_var.reset(token)

    assert record.request_id == "abc123"


def test_tests_never_read_the_developers_env_file():
    # Regression: a personal .env with AUTH_ENABLED=true once broke ~100 tests.
    import backend.config

    assert backend.config.ENV_FILE is None


def test_new_request_id_rejects_overlong_ids():
    assert new_request_id("x" * 65) != "x" * 65
