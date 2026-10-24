"""Compare retrieval configurations on the evaluation dataset.

Run from the project root (uses the indexes in DATA_DIR; makes no LLM calls):
    .\\.venv\\Scripts\\python.exe -m backend.evaluation.run_retrieval_eval
    .\\.venv\\Scripts\\python.exe -m backend.evaluation.run_retrieval_eval --pool 20,30,50 --rrf-k 60,20

Dense and BM25 are evaluated once; hybrid once per (pool size, RRF k) combination;
hybrid_rerank once per --rerank-candidates value. Mean and max retrieval latency are
measured per configuration after one warm-up query (so one-time model loading is not
counted). Results are printed and saved to evaluation/results/ with the configuration,
dataset fingerprint and git commit, so a run can be reproduced and compared later.
"""

import argparse
import json
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

from backend.config import PROJECT_ROOT, get_settings
from backend.evaluation.dataset import EvalQuestion, contains_quote, fingerprint, load_dataset
from backend.evaluation.retrieval_metrics import (
    QuestionResult,
    hit_at_k,
    mean_reciprocal_rank,
    recall_at_k,
    score_question,
)

DEFAULT_DATASET = PROJECT_ROOT / "evaluation" / "datasets" / "questions.jsonl"
DEFAULT_OUTPUT = PROJECT_ROOT / "evaluation" / "results"
MIN_QUESTIONS_FOR_CONCLUSIONS = 20


def _ints(value: str) -> list[int]:
    return [int(part) for part in value.split(",") if part.strip()]


def _git_revision() -> str:
    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                                cwd=PROJECT_ROOT, check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True,
                               cwd=PROJECT_ROOT, check=True).stdout.strip()
        return commit + ("-dirty" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def validate_against_index(questions: list[EvalQuestion], repository) -> tuple[list[tuple[EvalQuestion, str]], list[str]]:
    """Pair answerable questions with their indexed document; report evidence that is missing.

    A quote found in no chunk (typo, or split across a chunk boundary) would silently count
    as a retrieval miss, so such questions are excluded and listed instead.
    """
    documents = {d.filename: d for d in repository.list_all() if d.status == "indexed"}
    usable, problems = [], []
    for question in questions:
        if question.type == "unanswerable":
            continue
        document = documents.get(question.document)
        if document is None:
            problems.append(f"{question.id}: document '{question.document}' is not indexed")
            continue
        chunks = repository.list_chunks(document.document_id)
        missing = [e.quote for e in question.evidence if not any(contains_quote(c.text, e.quote) for c in chunks)]
        if missing:
            problems.append(f"{question.id}: evidence not found in any chunk: {missing}")
            continue
        usable.append((question, document.document_id))
    return usable, problems


def summarize(name: str, results: list[QuestionResult], ks: list[int], depth: int, latencies_ms: list[float]) -> dict:
    row = {"config": name, "n": len(results)}
    for k in ks:
        row[f"hit@{k}"] = round(hit_at_k(results, k), 3)
    row[f"recall@{max(ks)}"] = round(recall_at_k(results, max(ks)), 3)
    row[f"MRR@{depth}"] = round(mean_reciprocal_rank(results), 3)
    row["mean_ms"] = round(sum(latencies_ms) / len(latencies_ms), 1) if latencies_ms else None
    row["max_ms"] = round(max(latencies_ms), 1) if latencies_ms else None
    return row


def build_configs(modes: list[str], pools: list[int], rrf_ks: list[int], rerank_sizes: list[int]) -> list[tuple[str, str, dict]]:
    configs: list[tuple[str, str, dict]] = []
    for mode in modes:
        if mode == "hybrid":
            for pool in pools:
                for k in rrf_ks:
                    configs.append((f"hybrid(pool={pool},k={k})", mode,
                                    {"hybrid_candidates_per_retriever": pool, "rrf_k": k}))
        elif mode == "hybrid_rerank":
            for size in rerank_sizes:
                configs.append((f"hybrid_rerank(candidates={size})", mode, {"rerank_candidates": size}))
        else:
            configs.append((mode, mode, {}))
    return configs


def execute_retrieval_eval(
    services, repository, settings, dataset: Path, configs: list[tuple[str, str, dict]],
    depth: int = 20, ks: list[int] | None = None, progress=None,
) -> dict:
    """Evaluate each configuration on the dataset; returns the run record (not saved).
    `progress(done, total)` is called after each configuration, if given."""
    from backend.retrieval.hybrid_search import build_retriever

    ks = ks or [1, 3, 5, 8, 10]
    questions = load_dataset(dataset)
    usable, problems = validate_against_index(questions, repository)
    rows, per_question, fallbacks = [], {}, []
    for done, (name, mode, overrides) in enumerate(configs, start=1):
        retriever = build_retriever(mode, services, repository, settings.model_copy(update=overrides))
        if usable:
            retriever.search(usable[0][0].question, depth)  # warm-up (loads lazy models)
        results, latencies = [], []
        for q, doc_id in usable:
            started = time.perf_counter()
            retrieved = retriever.search(q.question, depth)
            latencies.append((time.perf_counter() - started) * 1000)
            # A reranker failure falls back to first-stage order under the first-stage name.
            if retrieved and retrieved[0].retriever != retriever.name:
                fallbacks.append(f"{name}/{q.id}: served by '{retrieved[0].retriever}' (fallback)")
            results.append(score_question(q, doc_id, retrieved))
        rows.append(summarize(name, results, ks, depth, latencies))
        per_question[name] = {r.question_id: r.first_relevant_rank for r in results}
        if progress:
            progress(done, len(configs))

    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return {
        "run_id": f"retrieval_{stamp}", "kind": "retrieval", "timestamp": stamp, "git": _git_revision(),
        "dataset": {"path": str(dataset.relative_to(PROJECT_ROOT)) if dataset.is_relative_to(PROJECT_ROOT) else str(dataset),
                    "fingerprint": fingerprint(dataset), "questions": len(questions),
                    "evaluated": len(usable), "reviewed": sum(1 for q, _ in usable if q.reviewed), "excluded": problems},
        "settings": {
            "embedding_model": settings.embedding_model_name,
            "chunk_size_tokens": settings.chunk_size_tokens,
            "chunk_overlap_tokens": settings.chunk_overlap_tokens,
            "bm25_k1": settings.bm25_k1, "bm25_b": settings.bm25_b, "depth": depth, "ks": ks,
            "reranker_model": settings.reranker_model_name,
        },
        "fallbacks": fallbacks,
        "configs": [{"name": name, "mode": mode, "overrides": overrides} for name, mode, overrides in configs],
        "summary": rows,
        "first_relevant_rank": per_question,
    }


def save_run(record: dict, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{record['run_id']}.json"
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(record, indent=2), encoding="utf-8")
    tmp.replace(path)
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--modes", default="dense,bm25,hybrid,hybrid_rerank")
    parser.add_argument("--pool", default=None, help="hybrid candidates per retriever, e.g. 20,30,50 (default: settings)")
    parser.add_argument("--rrf-k", default=None, help="RRF k values, e.g. 60,20 (default: settings)")
    parser.add_argument("--rerank-candidates", default=None, help="candidates rescored by the cross-encoder, e.g. 20,30,50 (default: settings)")
    parser.add_argument("--depth", type=int, default=20, help="results retrieved per question (MRR cutoff)")
    parser.add_argument("--ks", default="1,3,5,8,10")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args(argv)

    import pandas as pd
    from fastapi.testclient import TestClient

    from backend.main import create_app
    from backend.storage.repositories import DocumentRepository

    settings = get_settings()
    configs = build_configs(
        args.modes.split(","),
        _ints(args.pool) if args.pool else [settings.hybrid_candidates_per_retriever],
        _ints(args.rrf_k) if args.rrf_k else [settings.rrf_k],
        _ints(args.rerank_candidates) if args.rerank_candidates else [settings.rerank_candidates],
    )
    app = create_app()
    with TestClient(app):  # runs startup: loads models, reconciles and builds the indexes
        if app.state.retrieval is None:
            print("Retrieval is unavailable; check the backend logs.", file=sys.stderr)
            return 1
        record = execute_retrieval_eval(app.state.retrieval, DocumentRepository(app.state.db), settings,
                                        args.dataset, configs, args.depth, _ints(args.ks))

    data = record["dataset"]
    print(f"\nDataset: {args.dataset.name} ({data['fingerprint']}) | {data['questions']} questions, "
          f"{data['evaluated']} evaluated ({data['reviewed']} human-reviewed)")
    for problem in data["excluded"]:
        print(f"  EXCLUDED {problem}")
    for fallback in record["fallbacks"]:
        print(f"  FALLBACK {fallback}")
    print("\n" + pd.DataFrame(record["summary"]).set_index("config").to_string())
    print("\nRank of first relevant chunk per question (blank = not in top "
          f"{args.depth}):\n" + pd.DataFrame(record["first_relevant_rank"]).T.to_string(na_rep=""))
    if data["evaluated"] < MIN_QUESTIONS_FOR_CONCLUSIONS:
        print(f"\nWARNING: only {data['evaluated']} evaluated question(s). Differences between configurations "
              f"are anecdotal below about {MIN_QUESTIONS_FOR_CONCLUSIONS}; do not change defaults based on this run.")
    if not args.no_save:
        path = save_run(record, args.out)
        print(f"\nSaved {path.relative_to(PROJECT_ROOT) if path.is_relative_to(PROJECT_ROOT) else path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
