"""Reciprocal Rank Fusion and the hybrid retriever, tested without any models."""

import pytest

from backend.retrieval.hybrid_search import HybridRetriever, reciprocal_rank_fusion
from backend.schemas import Chunk, RetrievedChunk


def keys(fused) -> list[str]:
    return [item.key for item in fused]


def test_rrf_matches_hand_computed_scores():
    fused = reciprocal_rank_fusion({"dense": ["a", "b", "c"], "bm25": ["c", "a", "d"]}, k=60)

    scores = {item.key: item.score for item in fused}
    assert scores["a"] == pytest.approx(1 / 61 + 1 / 62)
    assert scores["c"] == pytest.approx(1 / 63 + 1 / 61)
    assert scores["b"] == pytest.approx(1 / 62)
    assert scores["d"] == pytest.approx(1 / 63)
    assert keys(fused) == ["a", "c", "b", "d"]


def test_agreement_between_retrievers_beats_a_single_top_rank():
    # "shared" is only 5th in both lists; "solo" is 1st in one list only.
    dense = ["solo", "x1", "x2", "x3", "shared"]
    bm25 = ["y1", "y2", "y3", "y4", "shared"]

    fused = reciprocal_rank_fusion({"dense": dense, "bm25": bm25}, k=60)

    assert keys(fused)[0] == "shared"


def test_component_ranks_are_recorded():
    fused = reciprocal_rank_fusion({"dense": ["a", "b"], "bm25": ["b"]})
    by_key = {item.key: item for item in fused}

    assert by_key["a"].ranks == {"dense": 1}
    assert by_key["b"].ranks == {"dense": 2, "bm25": 1}


def test_each_key_appears_once():
    fused = reciprocal_rank_fusion({"dense": ["a", "b", "c"], "bm25": ["b", "c", "a"]})

    assert sorted(keys(fused)) == ["a", "b", "c"]


def test_duplicates_within_one_list_count_once_and_do_not_shift_ranks():
    fused = reciprocal_rank_fusion({"dense": ["a", "a", "b"]})
    by_key = {item.key: item for item in fused}

    assert by_key["a"].score == pytest.approx(1 / 61)
    assert by_key["b"].ranks == {"dense": 2}


def test_ties_are_broken_deterministically():
    # Both are rank 1 in one list each: equal scores and equal best rank -> ordered by key.
    assert keys(reciprocal_rank_fusion({"dense": ["b"], "bm25": ["a"]})) == ["a", "b"]


def test_single_list_keeps_its_order():
    assert keys(reciprocal_rank_fusion({"dense": ["c", "a", "b"], "bm25": []})) == ["c", "a", "b"]


def test_empty_rankings_give_empty_result():
    assert reciprocal_rank_fusion({"dense": [], "bm25": []}) == []


def test_smaller_k_rewards_top_ranks_more():
    rankings = {"dense": ["top", "mid"], "bm25": ["x", "mid"]}
    # With a large k ranks barely matter, so appearing twice wins; with k=0 a lone #1
    # (1/1) beats two #2s (1/2 + 1/2) only on the tie-break, so check the scores directly.
    large = {i.key: i.score for i in reciprocal_rank_fusion(rankings, k=1000)}
    small = {i.key: i.score for i in reciprocal_rank_fusion(rankings, k=0)}

    assert large["mid"] > large["top"]
    assert small["top"] == pytest.approx(1.0) and small["mid"] == pytest.approx(1.0)


def test_negative_k_is_rejected():
    with pytest.raises(ValueError):
        reciprocal_rank_fusion({"dense": ["a"]}, k=-1)


# ---- HybridRetriever ----------------------------------------------------------------------

def chunk(chunk_id: str) -> Chunk:
    return Chunk(chunk_id=chunk_id, document_id="doc_x", filename="x.pdf", page_number=1,
                 chunk_index=0, start_char=0, end_char=4, token_count=1, text=f"text of {chunk_id}")


class StaticRetriever:
    def __init__(self, name: str, chunk_ids: list[str]) -> None:
        self.name = name
        self.chunk_ids = chunk_ids
        self.calls: list[tuple] = []

    def search(self, query, top_k, document_ids=None):
        self.calls.append((query, top_k, document_ids))
        return [
            RetrievedChunk(chunk=chunk(cid), score=1.0 / rank, rank=rank, retriever=self.name)
            for rank, cid in enumerate(self.chunk_ids[:top_k], start=1)
        ]


def test_hybrid_fuses_dedupes_and_truncates():
    dense = StaticRetriever("dense", ["a", "b", "c"])
    bm25 = StaticRetriever("bm25", ["c", "a", "d"])
    hybrid = HybridRetriever([dense, bm25], candidates_per_retriever=20, rrf_k=60)

    results = hybrid.search("q", top_k=3, document_ids=["doc_x"])

    assert [r.chunk.chunk_id for r in results] == ["a", "c", "b"]
    assert [r.rank for r in results] == [1, 2, 3]
    assert all(r.retriever == "hybrid" for r in results)
    assert results[0].component_ranks == {"dense": 1, "bm25": 2}
    assert results[0].component_scores == {"dense": 1.0, "bm25": 0.5}
    # Each component is asked for the candidate pool size with the same document filter.
    assert dense.calls == [("q", 20, ["doc_x"])] and bm25.calls == [("q", 20, ["doc_x"])]


# Reproduces the observed rankings for "GOOGL linear kernel RMSE 0.06248 MAPE 0.03752" on
# the real paper (MiniLM + BM25): dense ranks the Table 7 GOOGL row 25th, BM25 ranks it 1st,
# and a Table 11 GOOGL "near miss" is high in both lists.
OBSERVED_DENSE = ["near_miss"] + [f"table_chunk_{i}" for i in range(2, 25)] + ["target"]
OBSERVED_BM25 = ["target", "near_miss"] + [f"table_chunk_{i}" for i in range(2, 20)]


def test_single_retriever_top_hit_stays_in_hybrid_candidates():
    hybrid = HybridRetriever(
        [StaticRetriever("dense", OBSERVED_DENSE), StaticRetriever("bm25", OBSERVED_BM25)],
        candidates_per_retriever=20,
    )
    fused = hybrid.search("q", top_k=50)
    by_id = {r.chunk.chunk_id: r for r in fused}

    # Retrievable: it is in the fused candidates, credited by BM25 only (dense #25 is
    # outside the 20-candidate pool), and is not duplicated.
    assert by_id["target"].component_ranks == {"bm25": 1}
    assert len(by_id) == len(fused)
    # Consensus wins under RRF: the near miss (dense 1, bm25 2) outranks it.
    assert by_id["near_miss"].rank < by_id["target"].rank


@pytest.mark.xfail(strict=True, raises=AssertionError, reason=(
    "Known limitation of rank-only fusion: a decisive single-retriever #1 can fall below "
    "retrieval_top_k when the other retriever cannot match literal numbers. To be addressed "
    "by cross-encoder reranking (Phase 3) or by evaluation-backed fusion changes, not by "
    "tuning to this query."
))
def test_decisive_single_retriever_hit_reaches_the_llm_passages():
    hybrid = HybridRetriever(
        [StaticRetriever("dense", OBSERVED_DENSE), StaticRetriever("bm25", OBSERVED_BM25)],
        candidates_per_retriever=20,
    )

    assert "target" in [r.chunk.chunk_id for r in hybrid.search("q", top_k=8)]


def test_hybrid_works_when_one_retriever_finds_nothing():
    hybrid = HybridRetriever([StaticRetriever("dense", ["a", "b"]), StaticRetriever("bm25", [])], 20)

    assert [r.chunk.chunk_id for r in hybrid.search("q", top_k=5)] == ["a", "b"]
