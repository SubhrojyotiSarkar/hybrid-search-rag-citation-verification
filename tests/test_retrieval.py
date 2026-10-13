"""Dense retrieval through the API, including persistence and index recovery across restarts."""

from fastapi.testclient import TestClient

from backend.config import get_settings
from backend.retrieval.vector_store import INDEX_FILENAME
from tests.conftest import make_pdf

ML_PDF = make_pdf(
    [
        "L2 regularization adds a penalty on squared weights to reduce overfitting.",
        "Gradient descent updates parameters in the direction of the negative gradient.",
    ]
)
COOKING_PDF = make_pdf(["Knead the bread dough and let it rise for two hours before baking."])


def upload(client, content: bytes, filename: str) -> dict:
    response = client.post(
        "/api/documents/upload", files={"file": (filename, content, "application/pdf")}
    )
    assert response.status_code == 201, response.text
    return response.json()


def search(client, query: str, **extra):
    return client.post("/api/retrieval/search", json={"query": query, **extra})


def top_result(client, query: str) -> dict:
    return search(client, query).json()["results"][0]["chunk"]


def test_search_finds_the_relevant_page(client):
    upload(client, ML_PDF, "ml.pdf")
    upload(client, COOKING_PDF, "cooking.pdf")

    response = search(client, "how does L2 regularization reduce overfitting", top_k=3)

    assert response.status_code == 200
    body = response.json()
    assert body["retriever"] == "dense"
    best = body["results"][0]
    assert best["rank"] == 1
    assert best["chunk"]["filename"] == "ml.pdf"
    assert best["chunk"]["page_number"] == 1
    scores = [r["score"] for r in body["results"]]
    assert scores == sorted(scores, reverse=True)


def test_results_carry_printed_page_labels(client):
    article = make_pdf(
        ["Introduction to support vector regression.", "Kernel functions map inputs to features."],
        first_page_label=183,
    )
    upload(client, article, "article.pdf")

    best = top_result(client, "kernel functions")

    assert (best["page_number"], best["page_label"]) == (2, "184")


def test_document_filter_restricts_results(client):
    upload(client, ML_PDF, "ml.pdf")
    cooking = upload(client, COOKING_PDF, "cooking.pdf")

    body = search(client, "regularization", document_ids=[cooking["document_id"]]).json()

    assert body["results"]
    assert all(r["chunk"]["document_id"] == cooking["document_id"] for r in body["results"])


def test_unknown_document_filter_is_rejected(client):
    response = search(client, "anything", document_ids=["doc_missing"])

    assert response.status_code == 400
    assert response.json()["detail"]["code"] == "documents_not_searchable"


def test_blank_query_is_rejected(client):
    assert search(client, "   ").status_code == 422


def test_search_with_no_documents_returns_empty(client):
    body = search(client, "regularization").json()

    assert body["results"] == []


def test_deleted_document_is_no_longer_retrieved(client):
    ml = upload(client, ML_PDF, "ml.pdf")
    upload(client, COOKING_PDF, "cooking.pdf")

    client.delete(f"/api/documents/{ml['document_id']}")
    body = search(client, "L2 regularization overfitting", top_k=5).json()

    assert all(r["chunk"]["document_id"] != ml["document_id"] for r in body["results"])
    assert client.get("/health").json()["indexed_vectors"] == 1


def test_results_are_identical_after_restart(app_factory):
    with TestClient(app_factory()) as client:
        upload(client, ML_PDF, "ml.pdf")
        upload(client, COOKING_PDF, "cooking.pdf")
        before = search(client, "gradient descent parameters", top_k=3).json()["results"]

    with TestClient(app_factory()) as client:
        after = search(client, "gradient descent parameters", top_k=3).json()["results"]

    assert [r["chunk"]["chunk_id"] for r in after] == [r["chunk"]["chunk_id"] for r in before]


def test_missing_index_is_rebuilt_from_sqlite(app_factory):
    with TestClient(app_factory()) as client:
        upload(client, ML_PDF, "ml.pdf")
        expected = top_result(client, "gradient descent")

    (get_settings().index_dir / INDEX_FILENAME).unlink()

    with TestClient(app_factory()) as client:
        assert client.get("/health").json()["indexed_vectors"] == 2
        assert top_result(client, "gradient descent")["chunk_id"] == expected["chunk_id"]


def test_corrupt_index_is_rebuilt_from_sqlite(app_factory):
    with TestClient(app_factory()) as client:
        upload(client, ML_PDF, "ml.pdf")

    (get_settings().index_dir / INDEX_FILENAME).write_bytes(b"garbage")

    with TestClient(app_factory()) as client:
        health = client.get("/health").json()
        assert health["retrieval_ready"] is True
        assert health["indexed_vectors"] == 2
        assert top_result(client, "regularization")["filename"] == "ml.pdf"
