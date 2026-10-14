"""BM25 keyword retrieval over chunk text.

The index lives in memory and is rebuilt from SQLite at startup, so there is no second
persisted index that could drift from the database. Uploads and deletes update it.

Tokenization is shared by indexing and querying (the two must match exactly). It only
affects scoring: stored chunk text, citations and snippets keep the original text.
"""

import math
import re
import threading

import numpy as np
from rank_bm25 import BM25Okapi

from backend.schemas import RetrievedChunk
from backend.storage.repositories import DocumentRepository

# A word, optionally joined to more words by - . _ / (keeps "f1-score", "1e-3", "x_t", "tf-idf").
_TOKEN_PATTERN = re.compile(r"\w+(?:[-._/]\w+)*")
_PART_PATTERN = re.compile(r"\w+")
# Decimals such as "0.06248" or versions like "2.5.1" are never split: their parts ("0",
# "06248") are meaningless, and "0" would match every decimal in every table.
_PLAIN_NUMBER = re.compile(r"^\d+(?:\.\d+)+$")

# Small English stopword list: function words that carry no topical signal.
STOPWORDS = frozenset(
    """a an and are as at be been being but by can could did do does doing for from had has
    have having he her here hers him his how i if in into is it its itself just me more most
    my no nor not of off on once only or other our ours out over own same she should so some
    such than that the their theirs them then there these they this those through to too
    under until up very was we were what when where which while who whom why will with would
    you your yours""".split()
)


class LuceneIdfBM25(BM25Okapi):
    """BM25Okapi with the IDF used by Lucene/Elasticsearch: log(1 + (N - n + 0.5) / (n + 0.5)).

    rank-bm25's default IDF, log((N - n + 0.5) / (n + 0.5)), is exactly 0 for a term in
    half of the chunks and negative above that (then floored at a fraction of the average
    IDF, which can itself be <= 0). In small collections that silently gives real query
    terms zero weight. This IDF is always positive and still favours rare terms.
    """

    def _calc_idf(self, nd: dict[str, int]) -> None:
        for word, doc_count in nd.items():
            self.idf[word] = math.log(1 + (self.corpus_size - doc_count + 0.5) / (doc_count + 0.5))
        self.average_idf = sum(self.idf.values()) / len(self.idf) if self.idf else 0.0


def tokenize(text: str) -> list[str]:
    """Lowercase word tokens without stopwords; compounds are kept and also split into parts
    (except plain numbers, which stay whole)."""
    tokens: list[str] = []
    for match in _TOKEN_PATTERN.finditer(text.lower()):
        token = match.group()
        parts = _PART_PATTERN.findall(token)
        if len(parts) > 1 and not _PLAIN_NUMBER.match(token):
            tokens.append(token)
            tokens.extend(parts)
        else:
            tokens.append(token)
    return [token for token in tokens if token not in STOPWORDS]


class BM25Index:
    """Thread-safe BM25 index keyed by chunk vector_id.

    Token lists are cached per chunk, so an update re-tokenizes only the changed document;
    BM25 statistics (document frequencies, average length) are then recomputed over the
    whole corpus, which is fast at this project's scale.
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self._tokens: dict[int, list[str]] = {}
        self._ids: list[int] = []
        self._bm25: LuceneIdfBM25 | None = None
        self._lock = threading.Lock()

    @property
    def count(self) -> int:
        return len(self._ids)

    def add(self, pairs: list[tuple[int, str]]) -> None:
        """Add (vector_id, text) chunks."""
        tokenized = {vector_id: tokenize(text) for vector_id, text in pairs}
        with self._lock:
            self._tokens.update(tokenized)
            self._rebuild()

    def remove(self, vector_ids: list[int]) -> None:
        with self._lock:
            for vector_id in vector_ids:
                self._tokens.pop(vector_id, None)
            self._rebuild()

    def _rebuild(self) -> None:
        # Chunks with no scoring tokens (e.g. only stopwords) can never match; skip them.
        self._ids = [vector_id for vector_id, tokens in self._tokens.items() if tokens]
        corpus = [self._tokens[vector_id] for vector_id in self._ids]
        # BM25Okapi cannot be built from an empty corpus.
        self._bm25 = LuceneIdfBM25(corpus, k1=self.k1, b=self.b) if corpus else None

    def search(self, query: str, top_k: int, allowed_ids: set[int] | None = None) -> list[tuple[int, float]]:
        """Return up to top_k (vector_id, score) pairs with a positive score, best first."""
        query_tokens = tokenize(query)
        with self._lock:
            bm25, ids = self._bm25, self._ids
        if bm25 is None or not query_tokens:
            return []
        scores = bm25.get_scores(query_tokens)
        id_array = np.asarray(ids)
        mask = scores > 0
        if allowed_ids is not None:
            mask &= np.isin(id_array, list(allowed_ids))
        candidates = np.flatnonzero(mask)
        # Sort by score descending, then vector_id ascending, so ties are deterministic.
        order = candidates[np.lexsort((id_array[candidates], -scores[candidates]))][:top_k]
        return [(int(id_array[i]), float(scores[i])) for i in order]


class BM25Retriever:
    name = "bm25"

    def __init__(self, index: BM25Index, repository: DocumentRepository) -> None:
        self.index = index
        self.repository = repository

    def search(self, query: str, top_k: int, document_ids: list[str] | None = None) -> list[RetrievedChunk]:
        allowed_ids = None
        if document_ids is not None:
            allowed_ids = self.repository.indexed_vector_ids(document_ids)
            if not allowed_ids:
                return []
        hits = self.index.search(query, top_k, allowed_ids)
        chunks = self.repository.get_indexed_chunks([vector_id for vector_id, _ in hits])
        results: list[RetrievedChunk] = []
        for vector_id, score in hits:
            chunk = chunks.get(vector_id)
            if chunk is not None:  # skip chunks of documents that are no longer INDEXED
                results.append(
                    RetrievedChunk(chunk=chunk, score=score, rank=len(results) + 1, retriever=self.name)
                )
        return results
