"""Evaluation dataset loading, quote-based relevance, metrics, and a retrieval regression
check for the Table 7 GOOGL query."""

import json

import pytest
from fastapi.testclient import TestClient

from backend.config import PROJECT_ROOT, get_settings
from backend.evaluation.dataset import DatasetError, EvalQuestion, contains_quote, load_dataset
from backend.evaluation.retrieval_metrics import (
    QuestionResult,
    hit_at_k,
    mean_reciprocal_rank,
    recall_at_k,
    score_question,
)
from backend.evaluation.run_retrieval_eval import validate_against_index
from backend.retrieval.hybrid_search import build_retriever
from backend.storage.repositories import DocumentRepository
from tests.test_retrieval import TABLE_PAPER

GOOGL_QUERY = "GOOGL linear kernel RMSE 0.06248 MAPE 0.03752"


def question(**overrides) -> dict:
    base = {"id": "q1", "type": "keyword", "question": "Q?", "document": "paper.pdf",
            "evidence": [{"page": "194", "quote": "GOOGL 0.06248 0.03752"}]}
    return {**base, **overrides}


def write_jsonl(tmp_path, rows) -> object:
    path = tmp_path / "questions.jsonl"
    path.write_text("\n".join(json.dumps(r) if isinstance(r, dict) else r for r in rows), encoding="utf-8")
    return path


# ---- dataset ------------------------------------------------------------------------------

def test_shipped_dataset_is_valid():
    questions = load_dataset(PROJECT_ROOT / "evaluation" / "datasets" / "questions.jsonl")

    assert questions and all(q.question for q in questions)


def test_load_dataset_reports_line_numbers(tmp_path):
    path = write_jsonl(tmp_path, [question(), "{not json"])

    with pytest.raises(DatasetError, match="line 2"):
        load_dataset(path)


def test_duplicate_ids_are_rejected(tmp_path):
    with pytest.raises(DatasetError, match="duplicate id"):
        load_dataset(write_jsonl(tmp_path, [question(), question()]))


def test_unanswerable_questions_must_have_no_evidence():
    with pytest.raises(ValueError):
        EvalQuestion.model_validate(question(type="unanswerable"))
    with pytest.raises(ValueError):
        EvalQuestion.model_validate(question(evidence=[]))
    assert EvalQuestion.model_validate(question(type="unanswerable", evidence=[])).evidence == []


def test_quotes_match_across_pdf_line_breaks():
    chunk_text = "BAC\n0.04395\nGOOGL\n0.06248\n0.03752\n1.97066"

    assert contains_quote(chunk_text, "GOOGL 0.06248 0.03752")
    assert contains_quote(chunk_text, "googl  0.06248\n0.03752")
    assert not contains_quote(chunk_text, "GOOGL 0.06248 0.99999")


# ---- metrics ------------------------------------------------------------------------------

def test_metrics_match_hand_computation():
    results = [
        QuestionResult("a", "keyword", [1]),          # found at rank 1
        QuestionResult("b", "paraphrase", [4, None]),  # one of two quotes, at rank 4
        QuestionResult("c", "factual", [None]),        # not found
    ]

    assert hit_at_k(results, 1) == pytest.approx(1 / 3)
    assert hit_at_k(results, 5) == pytest.approx(2 / 3)
    assert recall_at_k(results, 5) == pytest.approx((1 + 0.5 + 0) / 3)
    assert mean_reciprocal_rank(results) == pytest.approx((1 + 1 / 4 + 0) / 3)


def test_metrics_of_empty_results_are_zero():
    assert hit_at_k([], 5) == recall_at_k([], 5) == mean_reciprocal_rank([]) == 0.0


# ---- regression: the Table 7 GOOGL query ------------------------------------------------------

@pytest.fixture
def table_paper_app(app_factory, monkeypatch):
    """The synthetic Table 7 / Table 13 paper, chunked small enough that Table 7 is split."""
    monkeypatch.setenv("CHUNK_SIZE_TOKENS", "60")
    monkeypatch.setenv("CHUNK_OVERLAP_TOKENS", "5")
    get_settings.cache_clear()
    app = app_factory()
    with TestClient(app) as client:
        response = client.post("/api/documents/upload",
                               files={"file": ("paper.pdf", TABLE_PAPER, "application/pdf")})
        assert response.status_code == 201
        yield app


def test_googl_query_evidence_is_retrieved(table_paper_app):
    repository = DocumentRepository(table_paper_app.state.db)
    q = EvalQuestion.model_validate(question(question=GOOGL_QUERY))
    [(validated, document_id)], problems = validate_against_index([q], repository)
    assert problems == []

    settings = get_settings()
    top_k = settings.retrieval_top_k
    for mode in ("bm25", "hybrid"):
        retriever = build_retriever(mode, table_paper_app.state.retrieval, repository, settings)
        result = score_question(validated, document_id, retriever.search(GOOGL_QUERY, top_k))
        # The passages sent to the LLM must contain the GOOGL row of Table 7.
        assert result.first_relevant_rank is not None and result.first_relevant_rank <= top_k, mode


def test_missing_evidence_is_reported_not_scored_as_a_miss(table_paper_app):
    repository = DocumentRepository(table_paper_app.state.db)
    typo = EvalQuestion.model_validate(question(evidence=[{"quote": "GOOGL 0.06248 0.99999"}]))
    unknown_doc = EvalQuestion.model_validate(question(id="q2", document="other.pdf"))

    usable, problems = validate_against_index([typo, unknown_doc], repository)

    assert usable == []
    assert any("evidence not found" in p for p in problems)
    assert any("not indexed" in p for p in problems)
