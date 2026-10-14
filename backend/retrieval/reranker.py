"""Cross-encoder reranking of a retriever's candidates.

A cross-encoder reads the query and a passage together and outputs one relevance score,
so it can use exact matches (numbers, entities) that separate embeddings blur. It is far
too slow to score the whole corpus, so it only reorders the first-stage candidates.

If the model cannot be loaded or scoring fails, the first-stage order is returned
unchanged and labelled with the first-stage retriever's name, so a fallback is visible
in every response and evaluation instead of silently passing as reranked output.
"""

import logging
import threading
import time
from typing import Protocol

from backend.schemas import RetrievedChunk, index_text

logger = logging.getLogger(__name__)


class Reranker(Protocol):
    model_name: str

    def score(self, query: str, passages: list[str]) -> list[float]:
        """One relevance score per passage (higher = more relevant)."""
        ...


class CrossEncoderReranker:
    """sentence-transformers CrossEncoder on CPU, loaded lazily on first use.

    Lazy loading keeps startup fast and means a failed model download only disables
    reranking, not the backend.
    """

    def __init__(self, model_name: str, batch_size: int = 32, device: str = "cpu") -> None:
        self.model_name = model_name
        self.batch_size = batch_size
        self.device = device
        self._model = None
        self._lock = threading.Lock()

    @property
    def loaded(self) -> bool:
        return self._model is not None

    def _load(self):
        if self._model is None:
            with self._lock:
                if self._model is None:
                    from sentence_transformers import CrossEncoder

                    started = time.perf_counter()
                    self._model = CrossEncoder(self.model_name, device=self.device)
                    logger.info("Loaded reranker %s in %.1fs", self.model_name, time.perf_counter() - started)
        return self._model

    def score(self, query: str, passages: list[str]) -> list[float]:
        if not passages:
            return []
        scores = self._load().predict(
            [(query, passage) for passage in passages],
            batch_size=self.batch_size, show_progress_bar=False, convert_to_numpy=True,
        )
        return [float(s) for s in scores]


class RerankingRetriever:
    """Retrieve `candidates` with a first-stage retriever, reorder them by cross-encoder score."""

    def __init__(self, base, reranker: Reranker, candidates: int) -> None:
        self.base = base
        self.reranker = reranker
        self.candidates = candidates
        self.name = f"{base.name}_rerank"

    def search(self, query: str, top_k: int, document_ids: list[str] | None = None) -> list[RetrievedChunk]:
        first_stage = self.base.search(query, max(self.candidates, top_k), document_ids)
        if not first_stage:
            return []
        try:
            # The reranker sees the same text that was indexed: table context + chunk text.
            scores = self.reranker.score(query, [index_text(r.chunk.context, r.chunk.text) for r in first_stage])
            if len(scores) != len(first_stage):
                raise ValueError(f"reranker returned {len(scores)} scores for {len(first_stage)} passages")
        except Exception:
            logger.exception("Reranking failed; returning %s order unchanged", self.base.name)
            return first_stage[:top_k]

        # Sort by cross-encoder score; ties keep the first-stage order.
        order = sorted(range(len(first_stage)), key=lambda i: (-scores[i], first_stage[i].rank))
        return [
            RetrievedChunk(
                chunk=first_stage[i].chunk,
                score=scores[i],
                rank=position,
                retriever=self.name,
                # Keep the first-stage evidence: component ranks plus the pre-rerank rank/score.
                component_ranks={**first_stage[i].component_ranks, self.base.name: first_stage[i].rank},
                component_scores={**first_stage[i].component_scores, self.base.name: first_stage[i].score},
            )
            for position, i in enumerate(order[:top_k], start=1)
        ]
