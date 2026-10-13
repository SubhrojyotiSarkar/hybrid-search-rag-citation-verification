"""Dense (semantic) retrieval: embed the query, search FAISS, load chunks from SQLite.

This is the Phase 1 baseline that hybrid retrieval and reranking are measured against.
"""

from backend.retrieval.embeddings import Embedder
from backend.retrieval.vector_store import FaissVectorStore
from backend.schemas import RetrievedChunk
from backend.storage.repositories import DocumentRepository


class DenseRetriever:
    name = "dense"

    def __init__(self, embedder: Embedder, store: FaissVectorStore, repository: DocumentRepository) -> None:
        self.embedder = embedder
        self.store = store
        self.repository = repository

    def search(
        self, query: str, top_k: int, document_ids: list[str] | None = None
    ) -> list[RetrievedChunk]:
        allowed_ids = None
        if document_ids is not None:
            allowed_ids = self.repository.indexed_vector_ids(document_ids)
            if not allowed_ids:
                return []

        query_vector = self.embedder.encode([query])[0]
        hits = self.store.search(query_vector, top_k, allowed_ids)

        # Only chunks of INDEXED documents are returned. A vector without a matching chunk
        # (e.g. left over from an interrupted delete) is skipped, never shown.
        chunks = self.repository.get_indexed_chunks([vector_id for vector_id, _ in hits])
        results: list[RetrievedChunk] = []
        for vector_id, score in hits:
            chunk = chunks.get(vector_id)
            if chunk is not None:
                results.append(
                    RetrievedChunk(chunk=chunk, score=score, rank=len(results) + 1, retriever=self.name)
                )
        return results
