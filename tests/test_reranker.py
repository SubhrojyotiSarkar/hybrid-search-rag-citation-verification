"""Cross-encoder reranking: ordering, fallback, lazy loading, and API wiring."""

import os

import pytest
from fastapi.testclient import TestClient

from backend.config import get_settings
from backend.retrieval.reranker import CrossEncoderReranker, RerankingRetriever
from backend.schemas import Chunk, RetrievedChunk
from tests.conftest import FakeReranker, make_pdf
from tests.test_hybrid_search import StaticRetriever


class ScriptedReranker:
    model_name = "scripted"

    def __init__(self, scores: dict[str, float]) -> None:
        self.scores = scores
        self.passages: list[str] = []

    def score(self, query, passages):
        self.passages = passages
        return [self.scores.get(p.split()[-1], 0.0) for p in passages]  # keyed by "text of <id>"


def test_candidates_are_reordered_by_reranker_score():
    base = StaticRetriever("hybrid", ["a", "b", "c", "d"])
    reranker = ScriptedReranker({"c": 5.0, "a": 1.0, "d": 3.0, "b": -2.0})
    retriever = RerankingRetriever(base, reranker, candidates=4)

    results = retriever.search("q", top_k=3, document_ids=["doc_x"])

    assert [r.chunk.chunk_id for r in results] == ["c", "d", "a"]
    assert [r.rank for r in results] == [1, 2, 3]
    assert [r.score for r in results] == [5.0, 3.0, 1.0]
    assert all(r.retriever == "hybrid_rerank" for r in results)
    # The pre-rerank rank is kept, so you can see how far reranking moved each chunk.
    assert results[0].component_ranks["hybrid"] == 3
    # The first stage is asked for the full candidate pool, with the same document filter.
    assert base.calls == [("q", 4, ["doc_x"])]


def test_ties_keep_first_stage_order():
    retriever = RerankingRetriever(StaticRetriever("hybrid", ["a", "b", "c"]), ScriptedReranker({}), candidates=3)

    assert [r.chunk.chunk_id for r in retriever.search("q", top_k=3)] == ["a", "b", "c"]


def test_candidate_pool_is_never_smaller_than_top_k():
    base = StaticRetriever("hybrid", [f"c{i}" for i in range(20)])
    RerankingRetriever(base, ScriptedReranker({}), candidates=5).search("q", top_k=8)

    assert base.calls[0][1] == 8


def test_reranker_sees_table_context_and_text():
    chunk = Chunk(chunk_id="t", document_id="d", filename="f.pdf", page_number=1, chunk_index=0,
                  start_char=0, end_char=5, token_count=3, text="GOOGL 0.06248 0.03752",
                  context="Table 7 daily price test data. Stock Linear kernel RMSE MAPE")

    class OneChunk:
        name = "hybrid"

        def search(self, query, top_k, document_ids=None):
            return [RetrievedChunk(chunk=chunk, score=0.03, rank=1, retriever="hybrid")]

    reranker = ScriptedReranker({})
    RerankingRetriever(OneChunk(), reranker, candidates=5).search("q", top_k=1)

    assert reranker.passages == ["Table 7 daily price test data. Stock Linear kernel RMSE MAPE\nGOOGL 0.06248 0.03752"]


@pytest.mark.parametrize("failure", [RuntimeError("model download failed"), "wrong_length"])
def test_failure_falls_back_to_first_stage_order_visibly(failure):
    class Failing:
        model_name = "failing"

        def score(self, query, passages):
            if failure == "wrong_length":
                return [1.0]
            raise failure

    retriever = RerankingRetriever(StaticRetriever("hybrid", ["a", "b", "c"]), Failing(), candidates=3)
    results = retriever.search("q", top_k=2)

    assert [r.chunk.chunk_id for r in results] == ["a", "b"]
    # Labelled with the first-stage name, so the fallback is visible in responses.
    assert all(r.retriever == "hybrid" for r in results)


def test_no_candidates_means_no_reranker_call():
    reranker = FakeReranker()

    assert RerankingRetriever(StaticRetriever("hybrid", []), reranker, candidates=5).search("q", 3) == []
    assert reranker.calls == []


def test_cross_encoder_loads_lazily():
    reranker = CrossEncoderReranker("cross-encoder/ms-marco-MiniLM-L-6-v2")

    assert reranker.loaded is False
    assert reranker.score("q", []) == []
    assert reranker.loaded is False  # nothing to score, nothing loaded


# ---- through the API ---------------------------------------------------------------------

ML_PDF = make_pdf([
    "L2 regularization adds a penalty on squared weights to reduce overfitting.",
    "Gradient descent updates parameters in the direction of the negative gradient.",
])


def test_hybrid_rerank_mode_through_the_api(client):
    client.post("/api/documents/upload", files={"file": ("ml.pdf", ML_PDF, "application/pdf")})

    body = client.post("/api/retrieval/search",
                       json={"query": "gradient descent parameters", "retriever": "hybrid_rerank"}).json()

    assert body["retriever"] == "hybrid_rerank"
    best = body["results"][0]
    assert best["chunk"]["page_number"] == 2
    assert {"dense", "bm25", "hybrid"} <= set(best["component_ranks"])
    health = client.get("/health").json()
    assert health["reranker_model"] == "cross-encoder/ms-marco-MiniLM-L-6-v2"


def test_failing_reranker_is_visible_in_api_response(app_factory):
    with TestClient(app_factory(reranker=FakeReranker(error=RuntimeError("boom")))) as client:
        client.post("/api/documents/upload", files={"file": ("ml.pdf", ML_PDF, "application/pdf")})
        body = client.post("/api/retrieval/search",
                           json={"query": "gradient descent", "retriever": "hybrid_rerank"}).json()

    assert body["results"]
    assert all(r["retriever"] == "hybrid" for r in body["results"])


def test_chat_uses_reranking_when_configured(app_factory, monkeypatch):
    from backend.schemas import GeneratedAnswer
    from tests.conftest import FakeLLM

    monkeypatch.setenv("RETRIEVAL_MODE", "hybrid_rerank")
    get_settings.cache_clear()
    llm = FakeLLM(output=GeneratedAnswer(answer_status="insufficient_evidence", claims=[], missing_information="x"))

    with TestClient(app_factory(llm_client=llm)) as client:
        client.post("/api/documents/upload", files={"file": ("ml.pdf", ML_PDF, "application/pdf")})
        body = client.post("/api/chat/ask", json={"question": "What does gradient descent do?"}).json()

    assert body["metadata"]["retriever"] == "hybrid_rerank"


# ---- real model (opt-in: RUN_MODEL_TESTS=1, needs the downloaded cross-encoder) -------------

@pytest.mark.skipif(os.environ.get("RUN_MODEL_TESTS") != "1", reason="set RUN_MODEL_TESTS=1 to run real-model tests")
def test_real_cross_encoder_prefers_exact_table_row_over_near_miss():
    # Structure of the real failure: the Table 7 GOOGL row (exact values) vs Table 11's GOOGL
    # row (same stock and metric names, different values), which hybrid ranked first.
    query = "GOOGL linear kernel RMSE 0.06248 MAPE 0.03752"
    exact = ("Table 7 SVR results, using the optimal parameter in Table 4 on the daily price test data. "
             "Stock Linear kernel Radial kernel Polynomial kernel RMSE MAPE RMSE MAPE RMSE MAPE\n"
             "BAC 0.04395 0.12505 0.06929 0.26302 0.08167 0.22308 GOOGL 0.06248 0.03752 1.97066 1.30933 1.18256 0.72193")
    near_miss = ("Table 11 Results of SVR models updated periodically for daily prices. "
                 "Stock Linear kernel Radial kernel Polynomial kernel RMSE MAPE RMSE MAPE RMSE MAPE\n"
                 "BAC 0.03912 0.10233 0.05521 0.20117 0.07433 0.19870 GOOGL 0.05817 0.03390 1.70221 1.11809 1.02447 0.66031")

    scores = CrossEncoderReranker(get_settings().reranker_model_name).score(query, [near_miss, exact])

    assert scores[1] > scores[0]
