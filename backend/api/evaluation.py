"""Evaluation endpoints used by the dashboard: runs, metrics, human reviews, dataset stats.

Runs are produced by the CLI runners (backend.evaluation.run_retrieval_eval and
run_answer_eval) and stored as JSON in EVALUATION_DIR/results. Human reviews are stored
separately in EVALUATION_DIR/reviews/<run_id>.json, so re-running never overwrites them.
"""

import json
import re
from typing import Literal

from fastapi import APIRouter, Request
from pydantic import BaseModel, Field

from backend.api.dependencies import api_error
from backend.config import get_settings
from backend.evaluation.citation_metrics import (
    VERDICTS,
    compute_answer_metrics,
    iter_claims,
    load_reviews,
    save_review,
)
from backend.evaluation.dataset import DatasetError, load_dataset
from backend.evaluation.jobs import EvaluationJob, JobConflictError

router = APIRouter(prefix="/api/evaluation", tags=["evaluation"])

# Only names the runners produce; anything else (e.g. "../x") is rejected.
_RUN_ID = re.compile(r"^(retrieval|answers)_\d{8}T\d{6}Z$")


class RunSummary(BaseModel):
    run_id: str
    kind: str
    timestamp: str
    git: str | None = None
    questions: int | None = None
    reviewed_claims: int | None = None


class ReviewRequest(BaseModel):
    key: str = Field(min_length=1, description="claim key: <question_id>/a<attempt>/<claim_id>")
    human_verdict: str | None = Field(default=None, description=f"one of {VERDICTS}, or null to delete")
    notes: str = Field(default="", max_length=2000)


def _load_run(run_id: str) -> dict:
    if not _RUN_ID.match(run_id):
        raise api_error(400, "invalid_run_id", "Not a valid evaluation run id.")
    path = get_settings().eval_results_dir / f"{run_id}.json"
    if not path.is_file():
        raise api_error(404, "run_not_found", f"No evaluation run '{run_id}'.")
    run = json.loads(path.read_text(encoding="utf-8"))
    run.setdefault("run_id", run_id)
    run.setdefault("kind", run_id.split("_", 1)[0])
    return run


@router.get("/runs", response_model=list[RunSummary])
def list_runs() -> list[RunSummary]:
    settings = get_settings()
    summaries = []
    for path in sorted(settings.eval_results_dir.glob("*.json"), reverse=True):
        if not _RUN_ID.match(path.stem):
            continue
        try:
            run = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue  # a partially written or foreign file is skipped, not fatal
        kind = path.stem.split("_", 1)[0]
        reviewed = len(load_reviews(settings.eval_reviews_dir, path.stem)) if kind == "answers" else None
        questions = run.get("dataset", {}).get("evaluated") or run.get("dataset", {}).get("questions")
        summaries.append(RunSummary(run_id=path.stem, kind=kind, timestamp=run.get("timestamp", ""),
                                    git=run.get("git"), questions=questions, reviewed_claims=reviewed))
    return summaries


@router.get("/runs/{run_id}")
def get_run(run_id: str) -> dict:
    """The run, plus (answer runs) its human reviews, per-claim review rows and metrics."""
    run = _load_run(run_id)
    if run["kind"] == "answers":
        reviews = load_reviews(get_settings().eval_reviews_dir, run_id)
        run["reviews"] = reviews
        run["metrics"] = compute_answer_metrics(run, reviews)
        run["claim_rows"] = [
            {key: row[key] for key in ("key", "question_id", "question_type", "attempt", "state")}
            | {"claim": row["claim"], "passages": [row["passages"][c["source_label"]] for c in row["claim"]["citations"]
                                                   if c["source_label"] in row["passages"]]}
            for row in iter_claims(run)
        ]
    return run


@router.get("/runs/{run_id}/metrics")
def get_metrics(run_id: str) -> dict:
    run = _load_run(run_id)
    if run["kind"] != "answers":
        return {"summary": run.get("summary", [])}
    return compute_answer_metrics(run, load_reviews(get_settings().eval_reviews_dir, run_id))


@router.put("/runs/{run_id}/reviews")
def put_review(run_id: str, request: ReviewRequest) -> dict:
    run = _load_run(run_id)
    if run["kind"] != "answers":
        raise api_error(400, "not_reviewable", "Only answer-evaluation runs have claims to review.")
    if request.key not in {row["key"] for row in iter_claims(run)}:
        raise api_error(404, "claim_not_found", f"No claim '{request.key}' in run '{run_id}'.")
    try:
        reviews = save_review(get_settings().eval_reviews_dir, run_id, request.key, request.human_verdict, request.notes)
    except ValueError as exc:
        raise api_error(422, "invalid_verdict", str(exc)) from exc
    return {"reviewed_claims": len(reviews), "metrics": compute_answer_metrics(run, reviews)}


class RunRequest(BaseModel):
    kind: Literal["retrieval", "answers"]
    # answers: only the first N questions (a quick check before a full, slow run)
    limit: int | None = Field(default=None, ge=1)
    # retrieval: which retrievers to compare, and cross-encoder candidate pool sizes
    modes: list[Literal["dense", "bm25", "hybrid", "hybrid_rerank"]] = ["dense", "bm25", "hybrid", "hybrid_rerank"]
    rerank_candidates: list[int] | None = None


def _dataset_path():
    return get_settings().evaluation_dir / "datasets" / "questions.jsonl"


@router.post("/run", response_model=EvaluationJob, status_code=202)
def start_run(request: RunRequest, http_request: Request) -> EvaluationJob:
    """Start an evaluation in the background; poll GET /api/evaluation/jobs/{job_id}."""
    from backend.evaluation.run_answer_eval import execute_answer_eval
    from backend.evaluation.run_retrieval_eval import build_configs, execute_retrieval_eval, save_run
    from backend.storage.repositories import DocumentRepository

    state = http_request.app.state
    settings = get_settings()
    if state.retrieval is None:
        raise api_error(503, "retrieval_unavailable", "Retrieval is unavailable. Check the backend logs.")
    if request.kind == "answers" and state.llm is None:
        raise api_error(503, "llm_not_configured", "Answer evaluation needs GEMINI_API_KEY.")
    dataset = _dataset_path()
    try:
        questions = load_dataset(dataset)
    except (OSError, DatasetError) as exc:
        raise api_error(400, "invalid_dataset", f"The evaluation dataset cannot be loaded: {exc}") from exc
    if not questions:
        raise api_error(400, "empty_dataset", "The evaluation dataset has no questions.")

    repository = DocumentRepository(state.db)
    if request.kind == "retrieval":
        configs = build_configs(request.modes, [settings.hybrid_candidates_per_retriever], [settings.rrf_k],
                                request.rerank_candidates or [settings.rerank_candidates])

        def work(progress):
            return execute_retrieval_eval(state.retrieval, repository, settings, dataset, configs,
                                          progress=lambda done, total: progress(done, total, f"configuration {done}/{total}"))
    else:
        def work(progress):
            def report(done, total, record):
                outcome = record["response"]["overall_status"] if record["response"] else f"error {record['error']['code']}"
                progress(done, total, f"{record['question_id']}: {outcome}")

            return execute_answer_eval(state.retrieval, repository, state.llm, settings, dataset,
                                       request.limit, progress=report)

    try:
        return state.jobs.start(request.kind, request.model_dump(), work,
                                lambda record: save_run(record, settings.eval_results_dir))
    except JobConflictError as exc:
        raise api_error(409, "job_running", str(exc)) from exc


@router.get("/jobs", response_model=list[EvaluationJob])
def list_jobs(http_request: Request) -> list[EvaluationJob]:
    return http_request.app.state.jobs.recent()


@router.get("/jobs/{job_id}", response_model=EvaluationJob)
def get_job(job_id: str, http_request: Request) -> EvaluationJob:
    job = http_request.app.state.jobs.get(job_id)
    if job is None:
        raise api_error(404, "job_not_found", f"No evaluation job '{job_id}' (jobs are forgotten on restart).")
    return job


@router.get("/dataset")
def dataset_summary() -> dict:
    """Question counts by type and review state, so dataset gaps are visible."""
    path = _dataset_path()
    if not path.is_file():
        return {"path": str(path), "questions": 0, "by_type": {}, "reviewed": 0, "error": None}
    try:
        questions = load_dataset(path)
    except DatasetError as exc:
        return {"path": str(path), "questions": 0, "by_type": {}, "reviewed": 0, "error": str(exc)}
    by_type: dict[str, int] = {}
    for q in questions:
        by_type[q.type] = by_type.get(q.type, 0) + 1
    return {"path": str(path), "questions": len(questions), "by_type": by_type,
            "reviewed": sum(q.reviewed for q in questions), "error": None}
