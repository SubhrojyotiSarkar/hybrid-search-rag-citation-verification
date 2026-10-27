"""Re-processing documents in place, index rebuilding and the consistency status."""

from fastapi.testclient import TestClient

from backend.config import get_settings
from backend.maintenance import index_status, rebuild_index, reprocess_documents
from tests.test_retrieval import TABLE_PAPER


def start(app_factory):
    app = app_factory()
    client = TestClient(app)
    client.__enter__()
    doc = client.post("/api/documents/upload", files={"file": ("paper.pdf", TABLE_PAPER, "application/pdf")}).json()
    return app, client, doc


def test_reprocess_keeps_document_id_and_applies_new_chunking(app_factory, monkeypatch):
    app, client, doc = start(app_factory)
    try:
        before = client.get(f"/api/documents/{doc['document_id']}/chunks").json()
        monkeypatch.setenv("CHUNK_SIZE_TOKENS", "40")  # e.g. after a chunking change
        monkeypatch.setenv("CHUNK_OVERLAP_TOKENS", "5")
        get_settings.cache_clear()

        response = client.post(f"/api/documents/{doc['document_id']}/reprocess")

        assert response.status_code == 200
        after = client.get(f"/api/documents/{doc['document_id']}/chunks").json()
        assert response.json()["document_id"] == doc["document_id"]
        assert response.json()["status"] == "indexed"
        assert len(after) > len(before)
        assert index_status(app)["consistent"] is True
        # Search finds the new chunks and nothing stale.
        results = client.post("/api/retrieval/search", json={"query": "GOOGL 0.06248", "retriever": "bm25"}).json()["results"]
        assert {r["chunk"]["chunk_id"] for r in results} <= {c["chunk_id"] for c in after}
    finally:
        client.__exit__(None, None, None)


def test_reprocess_failure_leaves_old_chunks_searchable(app_factory):
    app, client, doc = start(app_factory)
    try:
        chunks = client.get(f"/api/documents/{doc['document_id']}/chunks").json()
        (get_settings().upload_dir / f"{doc['document_id']}.pdf").write_bytes(b"%PDF-1.7 corrupted")

        response = client.post(f"/api/documents/{doc['document_id']}/reprocess")

        assert response.status_code == 422
        assert client.get(f"/api/documents/{doc['document_id']}/chunks").json() == chunks
        assert client.get(f"/api/documents/{doc['document_id']}").json()["status"] == "indexed"
        assert index_status(app)["consistent"] is True
    finally:
        client.__exit__(None, None, None)


def test_reprocess_reports_missing_file_and_unknown_document(app_factory):
    app, client, doc = start(app_factory)
    try:
        (get_settings().upload_dir / f"{doc['document_id']}.pdf").unlink()
        assert client.post(f"/api/documents/{doc['document_id']}/reprocess").json()["detail"]["code"] == "file_missing"
        assert client.post("/api/documents/doc_missing/reprocess").status_code == 404
    finally:
        client.__exit__(None, None, None)


def test_reprocess_all_continues_after_a_failure(app_factory):
    app, client, doc = start(app_factory)
    try:
        results = reprocess_documents(app, [doc["document_id"], "doc_missing"])

        assert [r["ok"] for r in results] == [True, False]
        assert "document_not_found" in results[1]["error"]
    finally:
        client.__exit__(None, None, None)


def test_rebuild_index_restores_every_vector(app_factory):
    app, client, doc = start(app_factory)
    try:
        expected = index_status(app)["chunks_expected_in_index"]
        app.state.retrieval.vector_store.remove(list(app.state.retrieval.vector_store.ids())[:2])
        assert index_status(app)["missing_from_faiss"] == 2

        result = rebuild_index(app)

        assert result["vectors"] == expected
        status = index_status(app)
        assert status["consistent"] is True and status["faiss_vectors"] == expected
    finally:
        client.__exit__(None, None, None)
