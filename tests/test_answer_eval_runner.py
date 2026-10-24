"""The answer-evaluation runner: record format, before/after claims, error and rate-limit handling."""

import json

from fastapi.testclient import TestClient

from backend.config import get_settings
from backend.evaluation.citation_metrics import compute_answer_metrics, iter_claims
from backend.evaluation.dataset import EvalQuestion
from backend.evaluation.run_answer_eval import RATE_LIMIT_PAUSE_SECONDS, evaluate_question
from backend.generation.factory import build_answer_generator
from backend.generation.llm_client import LLMError
from backend.storage.repositories import DocumentRepository
from tests.conftest import FakeLLM
from tests.test_verification_flow import ML_PDF, OVERSTATED, SUPPORTED, answer_with, scripted_verifier

QUESTION = EvalQuestion(id="q1", type="factual", question="What does L2 regularization do?", document="ml.pdf",
                        evidence=[{"quote": "squared magnitude"}], should_not_claim=[OVERSTATED])


def run_question(app_factory, monkeypatch, llm, **env):
    for name, value in env.items():
        monkeypatch.setenv(name, str(value))
    get_settings.cache_clear()
    app = app_factory(llm_client=llm)
    with TestClient(app) as client:
        client.post("/api/documents/upload", files={"file": ("ml.pdf", ML_PDF, "application/pdf")})
        generator = build_answer_generator(get_settings(), app.state.retrieval, DocumentRepository(app.state.db), llm)
        return evaluate_question(generator, QUESTION)


def test_record_keeps_first_attempt_claims_before_verification(app_factory, monkeypatch):
    llm = FakeLLM(output=answer_with(SUPPORTED, OVERSTATED),
                  verification=scripted_verifier({"always": "insufficient_evidence"}))
    record = run_question(app_factory, monkeypatch, llm, MAX_RECOVERY_ATTEMPTS=0)

    json.dumps(record)  # the whole record must be serializable
    assert record["error"] is None
    assert record["should_not_claim"] == [OVERSTATED]
    [attempt] = record["attempts"]
    assert [c["text"] for c in attempt["claims"]] == [SUPPORTED, OVERSTATED]  # before verification removed one
    assert [c["text"] for c in record["response"]["claims"]] == [SUPPORTED]
    assert attempt["passages"] and "text" in attempt["passages"][0]

    run = {"questions": [record]}
    states = {row["claim"]["text"]: row["state"] for row in iter_claims(run)}
    assert states == {SUPPORTED: "shown", OVERSTATED: "removed"}
    assert compute_answer_metrics(run, {})["answers"]["required_correction"]["n"] == 1


def test_recovered_question_records_both_attempts(app_factory, monkeypatch):
    llm = FakeLLM(output=[answer_with(SUPPORTED, OVERSTATED), answer_with(SUPPORTED)],
                  verification=scripted_verifier({"always": "insufficient_evidence"}))
    record = run_question(app_factory, monkeypatch, llm)

    assert [a["number"] for a in record["attempts"]] == [1, 2]
    states = sorted((row["attempt"], row["state"]) for row in iter_claims({"questions": [record]}))
    assert states == [(1, "superseded"), (1, "superseded"), (2, "shown")]


def test_generation_errors_are_recorded_not_raised(app_factory, monkeypatch):
    llm = FakeLLM(error=LLMError("llm_unavailable", "Down."))
    record = run_question(app_factory, monkeypatch, llm)

    assert record["response"] is None and record["error"]["code"] == "llm_unavailable"


class FlakyGenerator:
    """Raises the given errors in order, one per answer() call."""

    last_attempts = []

    def __init__(self, *errors):
        self.errors = list(errors)
        self.calls = 0

    def answer(self, question):
        self.calls += 1
        raise self.errors.pop(0)


def test_temporary_failures_are_retried_after_a_pause():
    sleeps = []
    generator = FlakyGenerator(
        LLMError("llm_rate_limited", "Quota.", retryable=True),
        LLMError("llm_unavailable", "Overloaded (503).", retryable=True),
        LLMError("llm_bad_request", "Stop here."),
    )
    record = evaluate_question(generator, QUESTION, sleep=sleeps.append)

    assert sleeps == [RATE_LIMIT_PAUSE_SECONDS, RATE_LIMIT_PAUSE_SECONDS]
    assert generator.calls == 3
    assert record["error"]["code"] == "llm_bad_request"


def test_permanent_failures_are_not_retried():
    sleeps = []
    generator = FlakyGenerator(LLMError("llm_auth_failed", "Bad key."))
    record = evaluate_question(generator, QUESTION, sleep=sleeps.append)

    assert sleeps == [] and generator.calls == 1
    assert record["error"]["code"] == "llm_auth_failed"


def test_retries_are_bounded():
    sleeps = []
    generator = FlakyGenerator(*[LLMError("llm_unavailable", "Down.", retryable=True)] * 5)
    record = evaluate_question(generator, QUESTION, sleep=sleeps.append)

    assert generator.calls == 3  # first try + RATE_LIMIT_RETRIES (2)
    assert record["error"]["code"] == "llm_unavailable"
