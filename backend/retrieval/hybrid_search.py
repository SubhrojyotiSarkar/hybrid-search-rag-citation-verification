"""Hybrid retrieval: run dense and BM25 independently, merge with Reciprocal Rank Fusion.

RRF uses ranks, not scores, because BM25 scores (unbounded) and cosine similarities
(-1..1) are on incomparable scales:

    rrf_score(chunk) = sum over retrievers r of  1 / (k + rank_r(chunk))

A chunk missing from a retriever's list contributes nothing for that retriever. k (60 in
the original RRF paper) damps the advantage of the very top ranks.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Protocol

from backend.config import Settings
from backend.retrieval.bm25_search import BM25Index, BM25Retriever
from backend.retrieval.dense_retriever import DenseRetriever
from backend.retrieval.embeddings import Embedder
from backend.retrieval.vector_store import FaissVectorStore
from backend.schemas import RetrievedChunk
from backend.storage.repositories import DocumentRepository


class Retriever(Protocol):
    name: str

    def search(self, query: str, top_k: int, document_ids: list[str] | None = None) -> list[RetrievedChunk]: ...


@dataclass
class FusedItem:
    key: str
    score: float
    ranks: dict[str, int] = field(default_factory=dict)  # retriever name -> 1-based rank


def reciprocal_rank_fusion(rankings: Mapping[str, Sequence[str]], k: int = 60) -> list[FusedItem]:
    """Fuse ranked lists of keys (best first) into one list, best first.

    Duplicate keys within one list count once, at their best rank. Ties are broken by the
    best individual rank, then by key, so the output is deterministic.
    """
    if k < 0:
        raise ValueError("k must be >= 0")
    fused: dict[str, FusedItem] = {}
    for retriever, keys in rankings.items():
        unique_keys = list(dict.fromkeys(keys))  # drop repeats, keep first (best) position
        for rank, key in enumerate(unique_keys, start=1):
            item = fused.setdefault(key, FusedItem(key=key, score=0.0))
            item.score += 1.0 / (k + rank)
            item.ranks[retriever] = rank
    return sorted(fused.values(), key=lambda item: (-item.score, min(item.ranks.values()), item.key))


class HybridRetriever:
    name = "hybrid"

    def __init__(self, retrievers: Sequence[Retriever], candidates_per_retriever: int, rrf_k: int = 60) -> None:
        self.retrievers = list(retrievers)
        self.candidates_per_retriever = candidates_per_retriever
        self.rrf_k = rrf_k

    def search(self, query: str, top_k: int, document_ids: list[str] | None = None) -> list[RetrievedChunk]:
        rankings: dict[str, list[str]] = {}
        by_key: dict[str, RetrievedChunk] = {}
        component_scores: dict[str, dict[str, float]] = {}
        for retriever in self.retrievers:
            results = retriever.search(query, self.candidates_per_retriever, document_ids)
            rankings[retriever.name] = [r.chunk.chunk_id for r in results]
            for result in results:
                by_key.setdefault(result.chunk.chunk_id, result)
                component_scores.setdefault(result.chunk.chunk_id, {})[retriever.name] = result.score

        fused = reciprocal_rank_fusion(rankings, self.rrf_k)[:top_k]
        return [
            RetrievedChunk(
                chunk=by_key[item.key].chunk, score=item.score, rank=position, retriever=self.name,
                component_ranks=item.ranks, component_scores=component_scores[item.key],
            )
            for position, item in enumerate(fused, start=1)
        ]


@dataclass
class RetrievalServices:
    """Long-lived retrieval state created at startup."""

    embedder: Embedder
    vector_store: FaissVectorStore
    bm25_index: BM25Index


def build_retriever(
    mode: str, services: RetrievalServices, repository: DocumentRepository, settings: Settings
) -> Retriever:
    """Construct the retriever for a mode: "dense", "bm25" or "hybrid"."""
    dense = DenseRetriever(services.embedder, services.vector_store, repository)
    bm25 = BM25Retriever(services.bm25_index, repository)
    if mode == "dense":
        return dense
    if mode == "bm25":
        return bm25
    if mode == "hybrid":
        return HybridRetriever([dense, bm25], settings.hybrid_candidates_per_retriever, settings.rrf_k)
    raise ValueError(f"Unknown retrieval mode '{mode}'")
