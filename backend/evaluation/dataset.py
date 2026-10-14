"""Evaluation dataset: human-reviewed questions with evidence quotes, one JSON object per line.

Relevance is defined by evidence *quotes* copied from the PDF, not by chunk IDs: chunk IDs
change whenever a document is re-chunked or re-uploaded, a quote does not. A retrieved
chunk is relevant to a question if its cited text contains one of the question's quotes
(compared case-insensitively with whitespace collapsed, since PDF extraction turns table
cells and line wraps into newlines).
"""

import hashlib
import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, ValidationError, model_validator

QuestionType = Literal["factual", "paraphrase", "keyword", "unanswerable"]


class Evidence(BaseModel):
    page: str | None = None  # printed page label, for human cross-checking
    quote: str = Field(min_length=3)


class EvalQuestion(BaseModel):
    id: str = Field(min_length=1)
    type: QuestionType
    question: str = Field(min_length=1)
    document: str = Field(min_length=1)  # filename as uploaded
    evidence: list[Evidence] = Field(default_factory=list)
    key_facts: list[str] = Field(default_factory=list)
    should_not_claim: list[str] = Field(default_factory=list)
    reviewed: bool = False  # True once a human has checked question, quotes and page
    notes: str = ""

    @model_validator(mode="after")
    def _evidence_matches_type(self) -> "EvalQuestion":
        if self.type == "unanswerable" and self.evidence:
            raise ValueError("unanswerable questions must have no evidence")
        if self.type != "unanswerable" and not self.evidence:
            raise ValueError("answerable questions need at least one evidence quote")
        return self


class DatasetError(ValueError):
    pass


def load_dataset(path: Path) -> list[EvalQuestion]:
    """Parse and validate a JSONL dataset; errors name the offending line."""
    questions: list[EvalQuestion] = []
    seen: set[str] = set()
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            question = EvalQuestion.model_validate(json.loads(line))
        except (json.JSONDecodeError, ValidationError) as exc:
            raise DatasetError(f"{path.name} line {number}: {exc}") from exc
        if question.id in seen:
            raise DatasetError(f"{path.name} line {number}: duplicate id '{question.id}'")
        seen.add(question.id)
        questions.append(question)
    return questions


def fingerprint(path: Path) -> str:
    """Hash of the dataset file, recorded with results so runs are comparable."""
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def normalize(text: str) -> str:
    return " ".join(text.split()).casefold()


def contains_quote(chunk_text: str, quote: str) -> bool:
    """Evidence must be in the chunk's own (cited) text, not in its table context."""
    return normalize(quote) in normalize(chunk_text)
