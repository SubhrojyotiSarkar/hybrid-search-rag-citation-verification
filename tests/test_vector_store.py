import json

import numpy as np

from backend.retrieval.vector_store import FaissVectorStore

DIM = 4
MODEL = "test-model"


def unit(*values: float) -> np.ndarray:
    vector = np.array(values, dtype=np.float32)
    return vector / np.linalg.norm(vector)


def make_store(tmp_path) -> FaissVectorStore:
    store = FaissVectorStore(tmp_path, MODEL, DIM)
    store.add(np.stack([unit(1, 0, 0, 0), unit(0, 1, 0, 0), unit(1, 1, 0, 0)]), [10, 20, 30])
    return store


def test_search_returns_nearest_first(tmp_path):
    store = make_store(tmp_path)
    hits = store.search(unit(1, 0, 0, 0), top_k=3)

    assert [vector_id for vector_id, _ in hits] == [10, 30, 20]
    assert abs(hits[0][1] - 1.0) < 1e-5  # identical vector: cosine similarity 1


def test_top_k_larger_than_index_returns_only_real_hits(tmp_path):
    hits = make_store(tmp_path).search(unit(1, 0, 0, 0), top_k=10)

    assert len(hits) == 3


def test_allowed_ids_filter_inside_faiss(tmp_path):
    hits = make_store(tmp_path).search(unit(1, 0, 0, 0), top_k=2, allowed_ids={20, 30})

    assert [vector_id for vector_id, _ in hits] == [30, 20]


def test_empty_allowed_ids_returns_nothing(tmp_path):
    assert make_store(tmp_path).search(unit(1, 0, 0, 0), top_k=2, allowed_ids=set()) == []


def test_remove_deletes_vectors(tmp_path):
    store = make_store(tmp_path)

    assert store.remove([10]) == 1
    assert store.ids() == {20, 30}
    assert 10 not in [vector_id for vector_id, _ in store.search(unit(1, 0, 0, 0), top_k=3)]


def test_save_and_reopen_gives_identical_results(tmp_path):
    store = make_store(tmp_path)
    store.save()
    reopened = FaissVectorStore.open(tmp_path, MODEL, DIM)

    assert reopened.load_status == "loaded"
    assert reopened.ids() == {10, 20, 30}
    assert reopened.search(unit(1, 1, 0, 0), 3) == store.search(unit(1, 1, 0, 0), 3)


def test_missing_index_starts_empty(tmp_path):
    store = FaissVectorStore.open(tmp_path, MODEL, DIM)

    assert store.load_status == "created"
    assert store.count == 0


def test_different_embedding_model_is_not_loaded(tmp_path):
    make_store(tmp_path).save()
    reopened = FaissVectorStore.open(tmp_path, "another-model", DIM)

    assert reopened.load_status == "reset_mismatch"
    assert reopened.count == 0


def test_corrupt_index_is_set_aside(tmp_path):
    store = make_store(tmp_path)
    store.save()
    store.index_path.write_bytes(b"not a faiss index")

    reopened = FaissVectorStore.open(tmp_path, MODEL, DIM)

    assert reopened.load_status == "reset_corrupt"
    assert reopened.count == 0
    assert store.index_path.with_suffix(".corrupt").exists()


def test_metadata_records_model_and_count(tmp_path):
    store = make_store(tmp_path)
    store.save()
    meta = json.loads(store.meta_path.read_text())

    assert meta["embedding_model"] == MODEL
    assert meta["dimension"] == DIM
    assert meta["vector_count"] == 3
