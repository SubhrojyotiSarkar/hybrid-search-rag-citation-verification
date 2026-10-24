"""Run every dataset question through the full answer pipeline and record the results.

Run from the project root (uses DATA_DIR's indexes and the configured Gemini key):
    .\\.venv\\Scripts\\python.exe -m backend.evaluation.run_answer_eval
    .\\.venv\\Scripts\\python.exe -m backend.evaluation.run_answer_eval --limit 3

LLM calls are paced to EVAL_REQUESTS_PER_MINUTE (default 5, the Gemini free-tier limit for
gemini-3.8-flash), so a run takes roughly (calls per question / 5) minutes per question.
A question that hits a rate limit or a temporary provider error (503/504) is retried after
a pause, up to RATE_LIMIT_RETRIES times.

For each question the run file records the final response and every generation attempt
(claims with verdicts, and the passages each attempt saw), so claims can be compared
before and after verification and reviewed by a human in the dashboard. The file contains
passage text from your documents: evaluation/results/answers_*.json is gitignored.
"""

import argparse
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

from backend.config import get_settings
from backend.evaluation.dataset import EvalQuestion, fingerprint, load_dataset
from backend.evaluation.run_retrieval_eval import _git_revision

RATE_LIMIT_RETRIES = 2
RATE_LIMIT_PAUSE_SECONDS = 65


def serialize_attempt(attempt) -> dict:
    return {
        "number": attempt.number,
        "answer_status": attempt.answer_status,
        "verified": attempt.verified,
        "error": attempt.error,
        "quote_check_downgrades": attempt.quote_check_downgrades,
        "passages": [
            {"source_label": p.source_label, "chunk_id": p.chunk.chunk_id, "filename": p.chunk.filename,
             "page_number": p.chunk.page_number, "page_label": p.chunk.page_label,
             "context": p.chunk.context, "text": p.chunk.text}
            for p in attempt.passages
        ],
        "claims": [claim.model_dump(mode="json") for claim in attempt.claims],
    }


def evaluate_question(generator, question: EvalQuestion, sleep=time.sleep) -> dict:
    """Answer one question; LLMErrors are recorded (and rate limits retried), never raised."""
    from backend.generation.llm_client import LLMError

    record = {
        "question_id": question.id, "type": question.type, "question": question.question,
        "document": question.document, "key_facts": question.key_facts,
        "should_not_claim": question.should_not_claim, "reviewed_question": question.reviewed,
        "response": None, "attempts": [], "error": None,
    }
    for retry in range(RATE_LIMIT_RETRIES + 1):
        try:
            response = generator.answer(question.question)
        except LLMError as exc:
            # Rate limits (429) and temporary provider failures (503 overloaded, 504
            # deadline) are marked retryable by the client: pause and try again.
            if exc.retryable and retry < RATE_LIMIT_RETRIES:
                print(f"  {exc.code}; pausing {RATE_LIMIT_PAUSE_SECONDS}s before retrying {question.id}")
                sleep(RATE_LIMIT_PAUSE_SECONDS)
                continue
            # detail: the provider's own message (e.g. which quota was exceeded), for diagnosis.
            record["error"] = {"code": exc.code, "message": exc.message, "detail": exc.detail}
            return record
        record["response"] = response.model_dump(mode="json")
        record["attempts"] = [serialize_attempt(a) for a in generator.last_attempts]
        return record
    return record


def execute_answer_eval(
    services, repository, base_llm, settings, dataset: Path, limit: int | None = None,
    requests_per_minute: int | None = None, progress=None, sleep=time.sleep,
) -> dict:
    """Answer each question with the configured pipeline; returns the run record (not saved).
    `progress(done, total, record)` is called after each question, if given."""
    from backend.generation.factory import build_answer_generator
    from backend.generation.llm_client import RateLimitedLLMClient, SlidingWindowRateLimiter

    questions = load_dataset(dataset)[:limit]
    limiter = SlidingWindowRateLimiter(requests_per_minute or settings.eval_requests_per_minute, sleep=sleep)
    llm = RateLimitedLLMClient(base_llm, limiter)
    generator = build_answer_generator(settings, services, repository, llm)

    started = time.perf_counter()
    records = []
    for number, question in enumerate(questions, start=1):
        record = evaluate_question(generator, question, sleep=sleep)
        records.append(record)
        if progress:
            progress(number, len(questions), record)

    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return {
        "run_id": f"answers_{stamp}", "kind": "answers", "timestamp": stamp, "git": _git_revision(),
        "duration_seconds": round(time.perf_counter() - started, 1),
        "rate_limit_wait_seconds": round(limiter.waited_seconds, 1),
        "dataset": {"path": str(dataset), "fingerprint": fingerprint(dataset), "questions": len(questions)},
        "settings": {
            "llm_model": settings.llm_model, "thinking_level": settings.gemini_thinking_level,
            "retrieval_mode": settings.retrieval_mode, "retrieval_top_k": settings.retrieval_top_k,
            "embedding_model": settings.embedding_model_name, "reranker_model": settings.reranker_model_name,
            "verification_enabled": settings.verification_enabled,
            "max_recovery_attempts": settings.max_recovery_attempts,
            "recovery_top_k": settings.recovery_top_k, "unsupported_claim_policy": settings.unsupported_claim_policy,
            "chunk_size_tokens": settings.chunk_size_tokens,
            "requests_per_minute": limiter.max_calls,
        },
        "questions": records,
    }


def main(argv: list[str] | None = None) -> int:
    settings = get_settings()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", type=Path, default=settings.evaluation_dir / "datasets" / "questions.jsonl")
    parser.add_argument("--limit", type=int, default=None, help="only the first N questions (for a quick check)")
    parser.add_argument("--requests-per-minute", type=int, default=settings.eval_requests_per_minute)
    args = parser.parse_args(argv)

    from fastapi.testclient import TestClient

    from backend.evaluation.run_retrieval_eval import save_run
    from backend.main import create_app
    from backend.storage.repositories import DocumentRepository

    def report(done: int, total: int, record: dict) -> None:
        if record["response"]:
            outcome = f"{record['response']['overall_status']} ({record['response']['metadata']['llm_calls']} LLM calls)"
        else:
            outcome = f"ERROR {record['error']['code']}: {record['error'].get('detail') or record['error']['message']}"
        print(f"  [{done}/{total}] {record['question_id']}: {outcome}")

    app = create_app()
    with TestClient(app):
        if app.state.retrieval is None or app.state.llm is None:
            print("Retrieval or the LLM is unavailable (check GEMINI_API_KEY and the backend logs).", file=sys.stderr)
            return 1
        print(f"Evaluating at <= {args.requests_per_minute} LLM requests/minute "
              f"(verification={'on' if settings.verification_enabled else 'off'}, "
              f"max recovery attempts={settings.max_recovery_attempts}).")
        run = execute_answer_eval(app.state.retrieval, DocumentRepository(app.state.db), app.state.llm, settings,
                                  args.dataset, args.limit, args.requests_per_minute, progress=report)

    path = save_run(run, settings.eval_results_dir)
    print(f"\nSaved {path}. Review its claims in the dashboard (Evaluation tab) to compute human-judged metrics.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
