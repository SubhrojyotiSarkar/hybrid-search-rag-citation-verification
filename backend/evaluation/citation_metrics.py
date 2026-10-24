"""Answer and citation metrics for an answer-evaluation run, plus human review handling.

Two kinds of numbers are kept strictly apart:

- verifier_judged: computed from the verifier's own verdicts. Available immediately, but
  NOT ground truth (the verifier grading its own pipeline).
- human: computed only from claims a person reviewed. A human verdict judges whether the
  claim's cited passages support it (not whether the claim is true in the world).

Definitions (claim level):
- citation precision = claims whose citations support them / claims judged
- unsupported-claim rate = 1 - citation precision
- before verification = the first attempt's claims as generated; after = claims shown to
  the user in the final answer
- verifier precision (supported) = human-supported / verifier-supported
- verifier recall (supported) = verifier-supported / human-supported
- correct abstention = an unanswerable question answered with insufficient_evidence
"""

import json
import statistics
from datetime import UTC, datetime
from pathlib import Path

VERDICTS = ("supported", "contradicted", "insufficient_evidence")
LATENCY_FIELDS = ("retrieval_ms", "generation_ms", "verification_ms", "total_ms", "llm_calls",
                  "input_tokens", "output_tokens")


def claim_key(question_id: str, attempt: int, claim_id: str) -> str:
    return f"{question_id}/a{attempt}/{claim_id}"


def iter_claims(run: dict):
    """Yield one row per claim of every attempt, with its final state in the answer:
    "shown", "removed" (dropped by verification) or "superseded" (from an attempt that
    was not chosen)."""
    for record in run["questions"]:
        response = record.get("response") or {}
        shown = {(c["attempt"], c["claim_id"]) for c in response.get("claims", [])}
        removed = {(c["attempt"], c["claim_id"]) for c in response.get("removed_claims", [])}
        for attempt in record.get("attempts", []):
            for claim in attempt["claims"]:
                position = (attempt["number"], claim["claim_id"])
                state = "shown" if position in shown else "removed" if position in removed else "superseded"
                yield {
                    "key": claim_key(record["question_id"], attempt["number"], claim["claim_id"]),
                    "question_id": record["question_id"],
                    "question_type": record["type"],
                    "attempt": attempt["number"],
                    "state": state,
                    "claim": claim,
                    "passages": {p["source_label"]: p for p in attempt.get("passages", [])},
                }


def _rate(numerator: int, denominator: int) -> float | None:
    return round(numerator / denominator, 3) if denominator else None


def _precision(rows: list[dict], reviews: dict) -> dict:
    judged = [r for r in rows if r["key"] in reviews]
    supported = sum(1 for r in judged if reviews[r["key"]]["human_verdict"] == "supported")
    return {
        "claims": len(rows), "reviewed": len(judged),
        "citation_precision": _rate(supported, len(judged)),
        "unsupported_rate": _rate(len(judged) - supported, len(judged)),
    }


def _latency(records: list[dict]) -> dict:
    summary = {}
    for field in LATENCY_FIELDS:
        values = sorted(r["response"]["metadata"][field] for r in records
                        if r.get("response") and r["response"]["metadata"].get(field) is not None)
        if not values:
            continue
        p95_index = min(len(values) - 1, round(0.95 * (len(values) - 1)))
        summary[field] = {"mean": round(statistics.fmean(values), 1), "p50": round(statistics.median(values), 1),
                          "p95": round(values[p95_index], 1), "max": round(values[-1], 1), "n": len(values)}
    return summary


def compute_answer_metrics(run: dict, reviews: dict[str, dict]) -> dict:
    records = run["questions"]
    answered = [r for r in records if r.get("response")]
    unanswerable = [r for r in answered if r["type"] == "unanswerable"]
    answerable = [r for r in answered if r["type"] != "unanswerable"]
    status_counts: dict[str, int] = {}
    for r in answered:
        status_counts[r["response"]["overall_status"]] = status_counts.get(r["response"]["overall_status"], 0) + 1

    def abstained(record: dict) -> bool:
        return record["response"]["overall_status"] == "insufficient_evidence"

    # An answer "required correction" if the verifier rejected any first-attempt claim.
    needed_correction = [
        r for r in answered
        if r.get("attempts") and any(c["verification_status"] in ("contradicted", "insufficient_evidence")
                                     for c in r["attempts"][0]["claims"])
    ]

    rows = list(iter_claims(run))
    first_attempt = [r for r in rows if r["attempt"] == 1]
    final_shown = [r for r in rows if r["state"] == "shown"]
    checked = [r for r in first_attempt if r["claim"]["verification_status"] in VERDICTS]

    # Verifier vs human, on reviewed claims the verifier actually judged (any attempt).
    pairs = [(r["claim"]["verification_status"], reviews[r["key"]]["human_verdict"])
             for r in rows if r["key"] in reviews and r["claim"]["verification_status"] in VERDICTS]
    confusion = {v: {h: 0 for h in VERDICTS} for v in VERDICTS}
    for verifier_verdict, human_verdict in pairs:
        confusion[verifier_verdict][human_verdict] += 1
    v_sup = sum(1 for v, _ in pairs if v == "supported")
    h_sup = sum(1 for _, h in pairs if h == "supported")
    both_sup = sum(1 for v, h in pairs if v == h == "supported")
    false_positive = sum(1 for v, h in pairs if v == "supported" and h != "supported")
    false_negative = sum(1 for v, h in pairs if v != "supported" and h == "supported")

    return {
        "answers": {
            "questions": len(records), "answered": len(answered),
            "errors": len(records) - len(answered),
            "overall_status": status_counts,
            "unanswerable": {"n": len(unanswerable), "correct_abstentions": sum(map(abstained, unanswerable)),
                             "rate": _rate(sum(map(abstained, unanswerable)), len(unanswerable))},
            "answerable": {"n": len(answerable), "false_abstentions": sum(map(abstained, answerable)),
                           "rate": _rate(sum(map(abstained, answerable)), len(answerable))},
            "required_correction": {"n": len(needed_correction), "rate": _rate(len(needed_correction), len(answered))},
        },
        "verifier_judged": {
            "note": "From the verifier's own verdicts: not ground truth.",
            "first_attempt_claims": len(first_attempt),
            "first_attempt_not_supported_rate": _rate(
                sum(1 for r in checked if r["claim"]["verification_status"] != "supported"), len(checked)),
            "final_shown_claims": len(final_shown),
            "verification_failed_claims": sum(1 for r in rows if r["claim"]["verification_status"] == "verification_failed"),
            "quote_check_downgrades": sum(1 for r in rows if r["claim"].get("quote_check_failed")),
        },
        "human": {
            "reviewed_claims": sum(1 for r in rows if r["key"] in reviews),
            "total_claims": len(rows),
            "before_verification": _precision(first_attempt, reviews),
            "after_verification": _precision(final_shown, reviews),
            "verifier_agreement": {
                "n": len(pairs),
                "accuracy": _rate(sum(1 for v, h in pairs if v == h), len(pairs)),
                "supported_precision": _rate(both_sup, v_sup),
                "supported_recall": _rate(both_sup, h_sup),
                "false_positives": false_positive,  # verifier said supported, human disagreed
                "false_negatives": false_negative,  # verifier rejected a claim the human found supported
                "confusion": confusion,  # confusion[verifier_verdict][human_verdict]
            },
        },
        "latency": _latency(answered),
    }


# ---- review storage -----------------------------------------------------------------------

def reviews_path(reviews_dir: Path, run_id: str) -> Path:
    return reviews_dir / f"{run_id}.json"


def load_reviews(reviews_dir: Path, run_id: str) -> dict[str, dict]:
    path = reviews_path(reviews_dir, run_id)
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8")).get("reviews", {})


def save_review(reviews_dir: Path, run_id: str, key: str, human_verdict: str | None, notes: str = "") -> dict[str, dict]:
    """Add, change or (with human_verdict=None) delete one review. Written atomically."""
    if human_verdict is not None and human_verdict not in VERDICTS:
        raise ValueError(f"human_verdict must be one of {VERDICTS}")
    reviews = load_reviews(reviews_dir, run_id)
    if human_verdict is None:
        reviews.pop(key, None)
    else:
        reviews[key] = {"human_verdict": human_verdict, "notes": notes.strip(),
                        "reviewed_at": datetime.now(UTC).isoformat(timespec="seconds")}
    reviews_dir.mkdir(parents=True, exist_ok=True)
    path = reviews_path(reviews_dir, run_id)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps({"run_id": run_id, "reviews": dict(sorted(reviews.items()))}, indent=2), encoding="utf-8")
    tmp.replace(path)
    return reviews
