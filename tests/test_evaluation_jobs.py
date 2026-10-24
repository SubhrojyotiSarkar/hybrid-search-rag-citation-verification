"""Background evaluation jobs: the manager itself and POST /api/evaluation/run end to end."""

import json
import threading
import time

import pytest
from fastapi.testclient import TestClient

from backend.config import get_settings
from backend.evaluation.jobs import JobConflictError, JobManager
from tests.conftest import FakeLLM
from tests.test_verification_flow import ML_PDF, SUPPORTED, answer_with, scripted_verifier


def wait_for(get_job, timeout=30.0) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        job = get_job()
        if job["status"] != "running":
            return job
        time.sleep(0.05)
    raise AssertionError("job did not finish in time")


# ---- JobManager ---------------------------------------------------------------------------

def test_job_succeeds_saves_and_reports_progress():
    manager, saved = JobManager(), []

    def work(progress):
        progress(1, 2, "half")
        progress(2, 2, "done")
        return {"run_id": "answers_20261024T000000Z"}

    job = manager.start("answers", {}, work, saved.append)
    final = wait_for(lambda: manager.get(job.job_id).model_dump())

    assert final["status"] == "succeeded" and final["run_id"] == "answers_20261024T000000Z"
    assert (final["done"], final["total"], final["message"]) == (2, 2, "done")
    assert saved == [{"run_id": "answers_20261024T000000Z"}]


def test_failed_job_reports_the_error():
    manager = JobManager()

    def work(progress):
        raise RuntimeError("index missing")

    job = manager.start("retrieval", {}, work, lambda record: None)
    final = wait_for(lambda: manager.get(job.job_id).model_dump())

    assert final["status"] == "failed" and "index missing" in final["error"]
    assert final["finished_at"]


def test_only_one_job_runs_at_a_time():
    manager, release = JobManager(), threading.Event()

    def blocking(progress):
        release.wait(5)
        return {"run_id": "retrieval_20261024T000000Z"}

    first = manager.start("retrieval", {}, blocking, lambda record: None)
    with pytest.raises(JobConflictError):
        manager.start("answers", {}, blocking, lambda record: None)
    release.set()
    wait_for(lambda: manager.get(first.job_id).model_dump())
    # Once finished, a new job may start.
    second = manager.start("answers", {}, lambda progress: {"run_id": "answers_20261024T000001Z"}, lambda r: None)
    assert wait_for(lambda: manager.get(second.job_id).model_dump())["status"] == "succeeded"


# ---- API ----------------------------------------------------------------------------------

def write_dataset(rows):
    path = get_settings().evaluation_dir / "datasets" / "questions.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")


QUESTIONS = [
    {"id": "q1", "type": "factual", "question": "What does L2 regularization add?", "document": "ml.pdf",
     "evidence": [{"quote": "squared magnitude"}], "reviewed": True},
    {"id": "q2", "type": "unanswerable", "question": "What is the capital of France?", "document": "ml.pdf", "evidence": []},
]


def started_client(app_factory, llm=None):
    client = TestClient(app_factory(llm_client=llm))
    client.__enter__()
    client.post("/api/documents/upload", files={"file": ("ml.pdf", ML_PDF, "application/pdf")})
    return client


def test_retrieval_job_end_to_end(app_factory):
    client = started_client(app_factory)
    try:
        write_dataset(QUESTIONS)
        job = client.post("/api/evaluation/run", json={"kind": "retrieval", "modes": ["dense", "bm25"]})
        assert job.status_code == 202
        final = wait_for(lambda: client.get(f"/api/evaluation/jobs/{job.json()['job_id']}").json())

        assert final["status"] == "succeeded", final["error"]
        run = client.get(f"/api/evaluation/runs/{final['run_id']}").json()
        assert [row["config"] for row in run["summary"]] == ["dense", "bm25"]
        assert run["dataset"]["evaluated"] == 1  # the unanswerable question has no evidence to rank
    finally:
        client.__exit__(None, None, None)


def test_answer_job_end_to_end(app_factory):
    llm = FakeLLM(output=answer_with(SUPPORTED), verification=scripted_verifier({}))
    client = started_client(app_factory, llm)
    try:
        write_dataset(QUESTIONS)
        job = client.post("/api/evaluation/run", json={"kind": "answers", "limit": 1}).json()
        final = wait_for(lambda: client.get(f"/api/evaluation/jobs/{job['job_id']}").json())

        assert final["status"] == "succeeded", final["error"]
        assert (final["done"], final["total"]) == (1, 1)
        run = client.get(f"/api/evaluation/runs/{final['run_id']}").json()
        assert run["metrics"]["answers"]["questions"] == 1
        assert client.get("/api/evaluation/jobs").json()[0]["job_id"] == job["job_id"]
    finally:
        client.__exit__(None, None, None)


def test_answer_job_requires_an_llm(app_factory, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "")
    get_settings.cache_clear()
    client = started_client(app_factory)
    try:
        write_dataset(QUESTIONS)
        response = client.post("/api/evaluation/run", json={"kind": "answers"})
        assert response.status_code == 503 and response.json()["detail"]["code"] == "llm_not_configured"
    finally:
        client.__exit__(None, None, None)


@pytest.mark.parametrize(("dataset", "code"), [(None, "invalid_dataset"), ("{broken", "invalid_dataset"), ("", "empty_dataset")])
def test_bad_datasets_are_rejected_before_starting(app_factory, dataset, code):
    client = started_client(app_factory)
    try:
        path = get_settings().evaluation_dir / "datasets" / "questions.jsonl"
        if dataset is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(dataset, encoding="utf-8")
        response = client.post("/api/evaluation/run", json={"kind": "retrieval"})
        assert response.status_code == 400 and response.json()["detail"]["code"] == code
    finally:
        client.__exit__(None, None, None)


def test_unknown_job_returns_404(client):
    assert client.get("/api/evaluation/jobs/nope").status_code == 404
