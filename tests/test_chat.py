"""/api/chat/ask end to end, with the fake embedder and a scripted fake LLM."""

import re

import pytest
from fastapi.testclient import TestClient

from backend.config import get_settings
from backend.generation.llm_client import LLMError
from backend.schemas import GeneratedAnswer, GeneratedClaim
from tests.conftest import FakeLLM, make_pdf

ML_PDF = make_pdf(
    [
        "L2 regularization adds a penalty on squared weights to reduce overfitting.",
        "Dropout randomly disables units during training.",
    ],
    first_page_label=183,
)
COOKING_PDF = make_pdf(["Knead the bread dough and let it rise for two hours."])


@pytest.fixture(autouse=True)
def generation_only(monkeypatch):
    """These tests cover generation and the API contract; verification is tested in
    tests/test_verification_flow.py."""
    monkeypatch.setenv("VERIFICATION_ENABLED", "false")
    get_settings.cache_clear()


def answer_citing_first_source(user_prompt: str) -> GeneratedAnswer:
    first_label = re.search(r'label="(S\d+)"', user_prompt).group(1)
    return GeneratedAnswer(
        answer_status="answered",
        claims=[GeneratedClaim(text="L2 regularization penalizes squared weights.", source_labels=[first_label])],
        missing_information="",
    )


@pytest.fixture
def make_client(app_factory):
    """Yield a function that starts a client with a given fake LLM and uploads documents."""
    clients = []

    def start(llm, pdfs=(("ml.pdf", ML_PDF),)):
        client = TestClient(app_factory(llm_client=llm))
        client.__enter__()
        clients.append(client)
        ids = []
        for name, content in pdfs:
            response = client.post("/api/documents/upload", files={"file": (name, content, "application/pdf")})
            ids.append(response.json()["document_id"])
        return client, ids

    yield start
    for client in clients:
        client.__exit__(None, None, None)


def ask(client, question: str, **extra):
    return client.post("/api/chat/ask", json={"question": question, **extra})


def test_answer_has_claims_with_resolved_citations(make_client):
    llm = FakeLLM(output=answer_citing_first_source)
    client, [doc_id] = make_client(llm)

    response = ask(client, "What does L2 regularization do?")

    assert response.status_code == 200
    body = response.json()
    assert body["overall_status"] == "unverified"  # never "verified" without a verifier
    assert body["answer_status"] == "answered"
    [claim] = body["claims"]
    [citation] = claim["citations"]
    assert citation["document_id"] == doc_id
    assert citation["filename"] == "ml.pdf"
    assert citation["page_label"] in {"183", "184"}
    assert claim["verification_status"] == "unverified"
    assert f"[{citation['source_label']}]" in body["answer"]
    # The cited chunk is one of the passages shown in the source viewer.
    assert citation["chunk_id"] in {s["chunk"]["chunk_id"] for s in body["sources"]}
    assert body["metadata"]["llm_model"] == "fake-llm"
    assert body["metadata"]["retriever"] == "hybrid"
    assert body["metadata"]["input_tokens"] == 120


def test_hallucinated_label_is_flagged(make_client):
    llm = FakeLLM(output=GeneratedAnswer(
        answer_status="answered",
        claims=[GeneratedClaim(text="A claim.", source_labels=["S42"])],
        missing_information="",
    ))
    client, _ = make_client(llm)

    [claim] = ask(client, "Anything?").json()["claims"]

    assert claim["citations"] == []
    assert claim["invalid_source_labels"] == ["S42"]


def test_model_abstention_is_reported_as_insufficient_evidence(make_client):
    llm = FakeLLM(output=GeneratedAnswer(
        answer_status="insufficient_evidence", claims=[],
        missing_information="The documents do not discuss quantum computing.",
    ))
    client, _ = make_client(llm)

    body = ask(client, "How do quantum computers work?").json()

    assert body["overall_status"] == "insufficient_evidence"
    assert body["claims"] == []
    assert "do not discuss quantum computing" in body["answer"]


def test_no_documents_abstains_without_calling_the_llm(make_client):
    llm = FakeLLM(output=answer_citing_first_source)
    client, _ = make_client(llm, pdfs=())

    body = ask(client, "What does L2 regularization do?").json()

    assert body["overall_status"] == "insufficient_evidence"
    assert llm.calls == []


def test_document_filter_limits_passages_sent_to_the_llm(make_client):
    llm = FakeLLM(output=answer_citing_first_source)
    client, [_, cooking_id] = make_client(llm, pdfs=(("ml.pdf", ML_PDF), ("cooking.pdf", COOKING_PDF)))

    body = ask(client, "regularization", document_ids=[cooking_id]).json()

    assert {s["chunk"]["document_id"] for s in body["sources"]} == {cooking_id}
    assert "ml.pdf" not in llm.calls[0]["user"]


def test_prompt_contains_system_rules_and_passages(make_client):
    llm = FakeLLM(output=answer_citing_first_source)
    client, _ = make_client(llm)

    ask(client, "What does L2 regularization do?")

    call = llm.calls[0]
    assert "untrusted" in call["system"]
    assert "<sources>" in call["user"] and "What does L2 regularization do?" in call["user"]
    assert call["schema"] is GeneratedAnswer


@pytest.mark.parametrize(
    ("code", "status"),
    [("llm_rate_limited", 429), ("llm_unavailable", 503), ("llm_auth_failed", 503),
     ("llm_invalid_output", 502), ("llm_truncated", 502), ("llm_refused", 422)],
)
def test_llm_errors_map_to_clear_http_errors(make_client, code, status):
    client, _ = make_client(FakeLLM(error=LLMError(code, "Readable message.")))

    response = ask(client, "What does L2 regularization do?")

    assert response.status_code == status
    assert response.json()["detail"] == {"code": code, "message": "Readable message."}


def test_unexpected_errors_do_not_leak_details(make_client):
    client, _ = make_client(FakeLLM(error=RuntimeError("secret internal detail")))

    response = ask(client, "What does L2 regularization do?")

    assert response.status_code == 500
    assert "secret internal detail" not in response.text


def test_missing_api_key_returns_llm_not_configured(app_factory, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "")
    get_settings.cache_clear()

    with TestClient(app_factory()) as client:
        assert client.get("/health").json()["llm_configured"] is False
        response = ask(client, "anything")

    assert response.status_code == 503
    assert response.json()["detail"]["code"] == "llm_not_configured"


def test_placeholder_api_key_counts_as_not_configured(app_factory, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "your-api-key-here")
    get_settings.cache_clear()

    with TestClient(app_factory()) as client:
        assert client.get("/health").json()["llm_configured"] is False


def test_unsupported_provider_disables_llm_without_crashing(app_factory, monkeypatch):
    # e.g. an old .env that still says LLM_PROVIDER=anthropic
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    get_settings.cache_clear()

    with TestClient(app_factory()) as client:
        health = client.get("/health").json()
        response = ask(client, "anything")

    assert health["llm_configured"] is False
    assert response.status_code == 503
    assert response.json()["detail"]["code"] == "llm_not_configured"


def test_health_reports_gemini_provider_and_model(app_factory):
    with TestClient(app_factory()) as client:
        health = client.get("/health").json()

    assert (health["llm_provider"], health["llm_model"]) == ("gemini", "gemini-3.8-flash")


def test_blank_question_is_rejected(make_client):
    client, _ = make_client(FakeLLM(output=answer_citing_first_source))

    assert ask(client, "   ").status_code == 422


def test_original_pdf_is_served_inline(make_client):
    client, [doc_id] = make_client(FakeLLM(output=answer_citing_first_source))

    response = client.get(f"/api/documents/{doc_id}/file")

    assert response.status_code == 200
    assert response.headers["content-type"] == "application/pdf"
    assert response.headers["content-disposition"].startswith("inline")
    assert response.content.startswith(b"%PDF-")
