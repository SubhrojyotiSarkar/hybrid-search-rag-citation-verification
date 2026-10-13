from backend.config import get_settings
from tests.conftest import FAKE_API_KEY, make_pdf

SAMPLE_PDF = make_pdf(
    [
        "L2 regularization adds a penalty proportional to the squared magnitude of the weights. "
        "It discourages large weights and can reduce overfitting.",
        "",
        "Dropout randomly disables units during training.",
    ]
)


def upload(client, content: bytes, filename: str = "notes.pdf"):
    return client.post(
        "/api/documents/upload", files={"file": (filename, content, "application/pdf")}
    )


# ---- Health -----------------------------------------------------------------------------

def test_health_reports_configuration(client):
    response = client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["embedding_model"] == "sentence-transformers/all-MiniLM-L6-v2"
    assert body["retrieval_ready"] is True
    assert body["llm_configured"] is True


def test_health_never_exposes_api_key(client):
    assert FAKE_API_KEY not in client.get("/health").text


def test_startup_creates_data_directories(client):
    settings = get_settings()

    assert settings.upload_dir.is_dir()
    assert settings.index_dir.is_dir()


# ---- Upload -----------------------------------------------------------------------------

def test_upload_extracts_chunks_and_indexes(client):
    response = upload(client, SAMPLE_PDF)

    assert response.status_code == 201
    doc = response.json()
    assert doc["status"] == "indexed"
    assert client.get("/health").json()["indexed_vectors"] == doc["chunk_count"]
    assert doc["page_count"] == 3
    assert doc["empty_pages"] == [2]
    assert doc["chunk_count"] >= 2
    assert (get_settings().upload_dir / f"{doc['document_id']}.pdf").is_file()

    chunks = client.get(f"/api/documents/{doc['document_id']}/chunks").json()
    assert len(chunks) == doc["chunk_count"]
    assert {c["page_number"] for c in chunks} == {1, 3}
    assert all(c["filename"] == "notes.pdf" for c in chunks)


def test_duplicate_upload_is_rejected(client):
    first = upload(client, SAMPLE_PDF)
    second = upload(client, SAMPLE_PDF, filename="copy.pdf")

    assert second.status_code == 409
    assert second.json()["detail"]["code"] == "duplicate_document"
    assert second.json()["detail"]["document_id"] == first.json()["document_id"]
    assert len(client.get("/api/documents").json()) == 1


def test_non_pdf_extension_is_rejected(client):
    response = upload(client, b"hello", filename="notes.txt")

    assert response.status_code == 400
    assert response.json()["detail"]["code"] == "not_a_pdf"


def test_pdf_extension_without_pdf_content_is_rejected(client):
    response = upload(client, b"<html>not a pdf</html>")

    assert response.status_code == 400
    assert response.json()["detail"]["code"] == "not_a_pdf"


def test_oversized_upload_is_rejected(client, monkeypatch):
    monkeypatch.setenv("MAX_UPLOAD_MB", "1")
    get_settings.cache_clear()

    response = upload(client, b"%PDF-" + b"0" * (1024 * 1024))

    assert response.status_code == 413
    assert response.json()["detail"]["code"] == "file_too_large"


def test_pdf_without_text_is_recorded_as_failed(client):
    response = upload(client, make_pdf(["", ""]))

    assert response.status_code == 422
    detail = response.json()["detail"]
    assert detail["code"] == "no_extractable_text"

    doc = client.get(f"/api/documents/{detail['document_id']}").json()
    assert doc["status"] == "failed"
    assert doc["empty_pages"] == [1, 2]


def test_corrupt_pdf_is_recorded_as_failed(client):
    response = upload(client, b"%PDF-1.7 garbage content")

    assert response.status_code == 422
    assert response.json()["detail"]["code"] == "unreadable_pdf"


def test_path_components_are_stripped_from_filename(client):
    response = upload(client, SAMPLE_PDF, filename="..\\..\\secret\\notes.pdf")

    assert response.status_code == 201
    assert response.json()["filename"] == "notes.pdf"


# ---- Read and delete ----------------------------------------------------------------------

def test_unknown_document_returns_404(client):
    response = client.get("/api/documents/doc_missing")

    assert response.status_code == 404
    assert response.json()["detail"]["code"] == "document_not_found"


def test_delete_removes_document_chunks_and_file(client):
    doc = upload(client, SAMPLE_PDF).json()
    file_path = get_settings().upload_dir / f"{doc['document_id']}.pdf"

    response = client.delete(f"/api/documents/{doc['document_id']}")

    assert response.status_code == 204
    assert client.get(f"/api/documents/{doc['document_id']}").status_code == 404
    assert not file_path.exists()
    # The same file can be uploaded again after deletion.
    assert upload(client, SAMPLE_PDF).status_code == 201


def test_delete_unknown_document_returns_404(client):
    assert client.delete("/api/documents/doc_missing").status_code == 404
