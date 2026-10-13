"""Persistent FAISS index keyed by SQLite chunk vector_ids.

FAISS stores only vectors and integer IDs; everything else about a chunk lives in SQLite.
The index can therefore always be rebuilt from the database (see reconcile_index).
"""

import json
import logging
import os
import threading
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path

import faiss
import numpy as np

logger = logging.getLogger(__name__)

INDEX_FILENAME = "faiss.index"
META_FILENAME = "faiss_meta.json"


def _atomic_write(path: Path, data: bytes) -> None:
    """Write to a temp file and swap it in, so a crash never leaves a half-written file."""
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_bytes(data)
    os.replace(tmp_path, path)


class FaissVectorStore:
    """Exact inner-product search (cosine on normalized vectors) with ID-based deletion."""

    def __init__(self, index_dir: Path, model_name: str, dimension: int) -> None:
        self.index_dir = index_dir
        self.model_name = model_name
        self.dimension = dimension
        self.index = faiss.IndexIDMap2(faiss.IndexFlatIP(dimension))
        # How the index was obtained on open(): "loaded", "created", "reset_mismatch" or
        # "reset_corrupt". Anything but "loaded" means reconcile_index must re-embed.
        self.load_status = "created"
        self._lock = threading.Lock()

    @property
    def index_path(self) -> Path:
        return self.index_dir / INDEX_FILENAME

    @property
    def meta_path(self) -> Path:
        return self.index_dir / META_FILENAME

    @classmethod
    def open(cls, index_dir: Path, model_name: str, dimension: int) -> "FaissVectorStore":
        """Load the persisted index, or start empty if it is missing, corrupt, or was built
        with a different embedding model. An empty store is refilled by reconcile_index."""
        store = cls(index_dir, model_name, dimension)
        if not store.index_path.exists():
            return store

        try:
            meta = json.loads(store.meta_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            meta = {}
        if meta.get("embedding_model") != model_name or meta.get("dimension") != dimension:
            logger.warning(
                "FAISS index was built with %s (dim %s); current model is %s (dim %d). Rebuilding.",
                meta.get("embedding_model"), meta.get("dimension"), model_name, dimension,
            )
            store.load_status = "reset_mismatch"
            return store

        try:
            data = np.frombuffer(store.index_path.read_bytes(), dtype=np.uint8)
            index = faiss.deserialize_index(data)
            if not isinstance(index, faiss.IndexIDMap2) or index.d != dimension:
                raise ValueError("unexpected index type or dimension")
        except Exception:
            backup = store.index_path.with_suffix(".corrupt")
            os.replace(store.index_path, backup)
            logger.exception("FAISS index is unreadable; moved it to %s and rebuilding.", backup)
            store.load_status = "reset_corrupt"
            return store

        store.index = index
        store.load_status = "loaded"
        return store

    def save(self) -> None:
        with self._lock:
            data = faiss.serialize_index(self.index).tobytes()
            count = self.index.ntotal
        self.index_dir.mkdir(parents=True, exist_ok=True)
        _atomic_write(self.index_path, data)
        meta = {
            "embedding_model": self.model_name,
            "dimension": self.dimension,
            "vector_count": count,
            "saved_at": datetime.now(UTC).isoformat(timespec="seconds"),
        }
        _atomic_write(self.meta_path, json.dumps(meta, indent=2).encode("utf-8"))

    @property
    def count(self) -> int:
        return self.index.ntotal

    def ids(self) -> set[int]:
        with self._lock:
            return set(faiss.vector_to_array(self.index.id_map).tolist())

    def add(self, vectors: np.ndarray, ids: list[int]) -> None:
        if len(ids) == 0:
            return
        vectors = np.ascontiguousarray(vectors, dtype=np.float32)
        if vectors.shape != (len(ids), self.dimension):
            raise ValueError(f"expected vectors of shape ({len(ids)}, {self.dimension}), got {vectors.shape}")
        with self._lock:
            self.index.add_with_ids(vectors, np.asarray(ids, dtype=np.int64))

    def remove(self, ids: Iterable[int]) -> int:
        id_array = np.asarray(list(ids), dtype=np.int64)
        if id_array.size == 0:
            return 0
        with self._lock:
            return int(self.index.remove_ids(id_array))

    def search(
        self, query_vector: np.ndarray, top_k: int, allowed_ids: set[int] | None = None
    ) -> list[tuple[int, float]]:
        """Return up to top_k (vector_id, score) pairs, best first.

        allowed_ids restricts the search to those vectors (e.g. selected documents) inside
        FAISS itself, so a filter never reduces the number of results below top_k.
        """
        if allowed_ids is not None and not allowed_ids:
            return []
        query = np.ascontiguousarray(query_vector, dtype=np.float32).reshape(1, -1)
        params = None
        if allowed_ids is not None:
            selector = faiss.IDSelectorBatch(np.asarray(sorted(allowed_ids), dtype=np.int64))
            params = faiss.SearchParameters(sel=selector)
        with self._lock:
            if self.index.ntotal == 0:
                return []
            scores, ids = self.index.search(query, top_k, params=params)
        # FAISS pads with id -1 when fewer than top_k vectors match.
        return [(int(i), float(s)) for i, s in zip(ids[0], scores[0]) if i != -1]
