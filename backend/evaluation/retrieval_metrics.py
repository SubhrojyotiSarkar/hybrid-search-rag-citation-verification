"""Retrieval metrics over ranked chunks, with quote-based relevance.

For each answerable question we record, per evidence quote, the rank of the first retrieved
chunk (from the question's document) that contains it. From those ranks:

- hit@k:    1 if any evidence quote is found in the top k, else 0.
- recall@k: fraction of the question's evidence quotes found in the top k.
- MRR:      1 / rank of the first relevant chunk (0 if none within the retrieved depth).

Each metric is averaged over questions.
"""

from collections.abc import Sequence
from dataclasses import dataclass

from backend.evaluation.dataset import EvalQuestion, contains_quote
from backend.schemas import RetrievedChunk


@dataclass
class QuestionResult:
    question_id: str
    question_type: str
    quote_ranks: list[int | None]  # per evidence quote: rank of first chunk containing it

    @property
    def first_relevant_rank(self) -> int | None:
        found = [rank for rank in self.quote_ranks if rank is not None]
        return min(found) if found else None


def score_question(question: EvalQuestion, document_id: str, results: Sequence[RetrievedChunk]) -> QuestionResult:
    ranks: list[int | None] = []
    for evidence in question.evidence:
        rank = next(
            (r.rank for r in results
             if r.chunk.document_id == document_id and contains_quote(r.chunk.text, evidence.quote)),
            None,
        )
        ranks.append(rank)
    return QuestionResult(question.id, question.type, ranks)


def hit_at_k(results: Sequence[QuestionResult], k: int) -> float:
    if not results:
        return 0.0
    return sum(1 for r in results if r.first_relevant_rank is not None and r.first_relevant_rank <= k) / len(results)


def recall_at_k(results: Sequence[QuestionResult], k: int) -> float:
    if not results:
        return 0.0
    per_question = [
        sum(1 for rank in r.quote_ranks if rank is not None and rank <= k) / len(r.quote_ranks)
        for r in results
    ]
    return sum(per_question) / len(per_question)


def mean_reciprocal_rank(results: Sequence[QuestionResult]) -> float:
    if not results:
        return 0.0
    return sum(1 / r.first_relevant_rank if r.first_relevant_rank else 0.0 for r in results) / len(results)
