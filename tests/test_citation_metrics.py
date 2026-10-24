"""Answer/citation metrics, review storage and the evaluation API, on hand-built runs whose
expected values are computed by hand in the comments."""

import json

import pytest

from backend.config import get_settings
from backend.evaluation.citation_metrics import (
    claim_key,
    compute_answer_metrics,
    iter_claims,
    load_reviews,
    save_review,
)
from backend.generation.llm_client import SlidingWindowRateLimiter


def claim(claim_id, status, attempt=1, labels=("S1",), quote_failed=False):
    return {"claim_id": claim_id, "text": f"text of {claim_id}", "attempt": attempt, "verification_status": status,
            "quote_check_failed": quote_failed, "citations": [{"source_label": label} for label in labels]}


def record(qid, qtype, overall, attempts, shown=(), removed=(), metadata=None):
    meta = {"retrieval_ms": 20.0, "generation_ms": 1000.0, "verification_ms": 500.0, "total_ms": 1600.0,
            "llm_calls": 2, "input_tokens": 1000, "output_tokens": 200} | (metadata or {})
    return {
        "question_id": qid, "type": qtype, "question": f"Question {qid}?", "document": "paper.pdf",
        "error": None,
        "response": {"overall_status": overall, "claims": list(shown), "removed_claims": list(removed), "metadata": meta},
        "attempts": [{"number": n, "passages": [{"source_label": "S1", "text": "passage text", "page_label": "194"}],
                      "claims": claims} for n, claims in attempts],
    }


def sample_run() -> dict:
    # q1: verified at once.  q2: attempt 1 had an unsupported claim, recovered in attempt 2.
    # q3: unanswerable, correctly abstained.  q4: answerable but abstained.  q5: generation error.
    q1_a1 = [claim("claim_001", "supported"), claim("claim_002", "supported")]
    q2_a1 = [claim("claim_001", "supported"), claim("claim_002", "insufficient_evidence", quote_failed=True)]
    q2_a2 = [claim("claim_001", "supported", attempt=2)]
    return {
        "run_id": "answers_20261015T000000Z", "kind": "answers",
        "questions": [
            record("q1", "factual", "verified_within_provided_evidence", [(1, q1_a1)], shown=q1_a1),
            record("q2", "paraphrase", "verified_within_provided_evidence", [(1, q2_a1), (2, q2_a2)], shown=q2_a2,
                   metadata={"llm_calls": 4, "total_ms": 4000.0}),
            record("q3", "unanswerable", "insufficient_evidence", [(1, [])]),
            record("q4", "keyword", "insufficient_evidence", [(1, [])]),
            {"question_id": "q5", "type": "factual", "question": "?", "document": "paper.pdf",
             "error": {"code": "llm_unavailable", "message": "x"}, "response": None, "attempts": []},
        ],
    }


def test_claim_states_distinguish_shown_removed_and_superseded():
    run = sample_run()
    run["questions"][1]["response"]["removed_claims"] = [claim("claim_002", "insufficient_evidence")]
    states = {row["key"]: row["state"] for row in iter_claims(run)}

    assert states["q2/a2/claim_001"] == "shown"
    assert states["q2/a1/claim_002"] == "removed"
    assert states["q2/a1/claim_001"] == "superseded"  # from the attempt that was not chosen


def test_answer_level_metrics():
    metrics = compute_answer_metrics(sample_run(), reviews={})["answers"]

    assert (metrics["questions"], metrics["answered"], metrics["errors"]) == (5, 4, 1)
    assert metrics["unanswerable"] == {"n": 1, "correct_abstentions": 1, "rate": 1.0}
    assert metrics["answerable"] == {"n": 3, "false_abstentions": 1, "rate": pytest.approx(1 / 3, abs=1e-3)}
    # Only q2's first attempt had a claim the verifier rejected: 1 of 4 answered.
    assert metrics["required_correction"] == {"n": 1, "rate": 0.25}
    assert metrics["overall_status"] == {"verified_within_provided_evidence": 2, "insufficient_evidence": 2}


def test_verifier_judged_metrics_are_labelled_not_ground_truth():
    metrics = compute_answer_metrics(sample_run(), reviews={})["verifier_judged"]

    assert "not ground truth" in metrics["note"]
    # First-attempt claims: q1 (2) + q2 (2) = 4, of which 1 was not supported.
    assert metrics["first_attempt_claims"] == 4
    assert metrics["first_attempt_not_supported_rate"] == 0.25
    assert metrics["final_shown_claims"] == 3
    assert metrics["quote_check_downgrades"] == 1


def test_human_metrics_before_and_after_verification():
    v = "supported"
    reviews = {
        # q1: both shown claims really supported.
        "q1/a1/claim_001": {"human_verdict": v}, "q1/a1/claim_002": {"human_verdict": v},
        # q2 attempt 1: claim_001 fine; claim_002 truly unsupported (verifier was right).
        "q2/a1/claim_001": {"human_verdict": v}, "q2/a1/claim_002": {"human_verdict": "insufficient_evidence"},
        # q2 attempt 2: the human disagrees with the verifier (a verifier false positive).
        "q2/a2/claim_001": {"human_verdict": "contradicted"},
    }
    human = compute_answer_metrics(sample_run(), reviews)["human"]

    # Before = first-attempt claims: 4 reviewed, 3 supported.
    assert human["before_verification"] == {"claims": 4, "reviewed": 4, "citation_precision": 0.75, "unsupported_rate": 0.25}
    # After = shown claims (q1 a1 x2, q2 a2 x1): 3 reviewed, 2 supported.
    assert human["after_verification"]["citation_precision"] == pytest.approx(0.667, abs=1e-3)
    agreement = human["verifier_agreement"]
    # 5 pairs; agree on 4 (q1 x2, q2a1 x2), disagree on q2a2.
    assert agreement["n"] == 5 and agreement["accuracy"] == 0.8
    # Verifier said supported 4 times, human agreed 3 times; human-supported 3, all caught.
    assert agreement["supported_precision"] == 0.75 and agreement["supported_recall"] == 1.0
    assert (agreement["false_positives"], agreement["false_negatives"]) == (1, 0)
    assert agreement["confusion"]["supported"]["contradicted"] == 1


def test_verification_failed_claims_are_excluded_from_agreement():
    run = sample_run()
    run["questions"][0]["attempts"][0]["claims"][0]["verification_status"] = "verification_failed"
    reviews = {"q1/a1/claim_001": {"human_verdict": "supported"}}

    agreement = compute_answer_metrics(run, reviews)["human"]["verifier_agreement"]

    assert agreement["n"] == 0 and agreement["accuracy"] is None


def test_latency_summary():
    latency = compute_answer_metrics(sample_run(), reviews={})["latency"]

    assert latency["llm_calls"]["max"] == 4
    assert latency["total_ms"]["n"] == 4
    assert latency["total_ms"]["p50"] == 1600.0


def test_empty_run_gives_null_rates_not_errors():
    metrics = compute_answer_metrics({"questions": []}, {})

    assert metrics["answers"]["unanswerable"]["rate"] is None
    assert metrics["human"]["before_verification"]["citation_precision"] is None


def test_reviews_are_saved_updated_and_deleted(tmp_path):
    save_review(tmp_path, "answers_x", "q1/a1/claim_001", "supported", " ok ")
    save_review(tmp_path, "answers_x", "q1/a1/claim_001", "contradicted")
    reviews = load_reviews(tmp_path, "answers_x")
    assert reviews["q1/a1/claim_001"]["human_verdict"] == "contradicted"

    save_review(tmp_path, "answers_x", "q1/a1/claim_001", None)
    assert load_reviews(tmp_path, "answers_x") == {}


def test_invalid_verdict_is_rejected(tmp_path):
    with pytest.raises(ValueError):
        save_review(tmp_path, "answers_x", "k", "probably")


def test_claim_key_format():
    assert claim_key("q7", 2, "claim_003") == "q7/a2/claim_003"


# ---- rate limiter ---------------------------------------------------------------------------

def test_rate_limiter_waits_when_the_window_is_full():
    clock = {"t": 0.0}
    sleeps = []

    def sleep(seconds):
        sleeps.append(seconds)
        clock["t"] += seconds

    limiter = SlidingWindowRateLimiter(2, period=60, clock=lambda: clock["t"], sleep=sleep)
    limiter.acquire()
    clock["t"] = 10
    limiter.acquire()
    clock["t"] = 20
    limiter.acquire()  # third call within 60 s of the first: must wait until t = 60

    assert len(sleeps) == 1 and sleeps[0] == pytest.approx(40.05)
    assert limiter.waited_seconds == pytest.approx(40.05)


def test_rate_limiter_rejects_nonpositive_limits():
    with pytest.raises(ValueError):
        SlidingWindowRateLimiter(0)


# ---- API --------------------------------------------------------------------------------------

def write_run(run: dict) -> None:
    results = get_settings().eval_results_dir
    results.mkdir(parents=True, exist_ok=True)
    (results / f"{run['run_id']}.json").write_text(json.dumps(run), encoding="utf-8")


def test_evaluation_api_lists_runs_and_saves_reviews(client):
    run = sample_run()
    write_run(run)
    write_run({"run_id": "retrieval_20261014T000000Z", "timestamp": "x", "summary": [{"config": "bm25"}]})

    runs = {r["run_id"]: r for r in client.get("/api/evaluation/runs").json()}
    assert set(runs) == {"retrieval_20261014T000000Z", "answers_20261015T000000Z"}
    assert runs["answers_20261015T000000Z"]["kind"] == "answers"
    assert runs["answers_20261015T000000Z"]["reviewed_claims"] == 0
    assert runs["retrieval_20261014T000000Z"]["reviewed_claims"] is None

    detail = client.get(f"/api/evaluation/runs/{run['run_id']}").json()
    assert detail["metrics"]["human"]["reviewed_claims"] == 0
    row = next(r for r in detail["claim_rows"] if r["key"] == "q1/a1/claim_001")
    assert row["state"] == "shown" and row["passages"][0]["text"] == "passage text"

    saved = client.put(f"/api/evaluation/runs/{run['run_id']}/reviews",
                       json={"key": "q1/a1/claim_001", "human_verdict": "supported", "notes": "checked p194"}).json()
    assert saved["reviewed_claims"] == 1
    assert saved["metrics"]["human"]["before_verification"]["reviewed"] == 1
    # Reviews live outside the run file, so re-running never overwrites them.
    assert (get_settings().eval_reviews_dir / f"{run['run_id']}.json").is_file()


@pytest.mark.parametrize(("payload", "status"), [
    ({"key": "q1/a1/claim_001", "human_verdict": "maybe"}, 422),
    ({"key": "q9/a1/claim_001", "human_verdict": "supported"}, 404),
])
def test_bad_reviews_are_rejected(client, payload, status):
    run = sample_run()
    write_run(run)

    assert client.put(f"/api/evaluation/runs/{run['run_id']}/reviews", json=payload).status_code == status


@pytest.mark.parametrize(("run_id", "status"), [("answers_..secret", 400), ("answers_20990101T000000Z", 404)])
def test_unknown_or_malformed_run_ids_are_rejected(client, run_id, status):
    assert client.get(f"/api/evaluation/runs/{run_id}").status_code == status


def test_retrieval_runs_cannot_be_reviewed(client):
    write_run({"run_id": "retrieval_20261014T000000Z", "summary": []})

    response = client.put("/api/evaluation/runs/retrieval_20261014T000000Z/reviews",
                          json={"key": "q1/a1/claim_001", "human_verdict": "supported"})
    assert response.status_code == 400


def test_dataset_summary_counts_types(client):
    path = get_settings().evaluation_dir / "datasets" / "questions.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        {"id": "q1", "type": "keyword", "question": "Q?", "document": "d.pdf", "evidence": [{"quote": "abc"}], "reviewed": True},
        {"id": "q2", "type": "unanswerable", "question": "Q?", "document": "d.pdf", "evidence": []},
    ]
    path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")

    summary = client.get("/api/evaluation/dataset").json()

    assert summary["questions"] == 2 and summary["reviewed"] == 1
    assert summary["by_type"] == {"keyword": 1, "unanswerable": 1}
