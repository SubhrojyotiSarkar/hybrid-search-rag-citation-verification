"""Dense retrieval through the API, including persistence and index recovery across restarts."""

import pytest
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

    response = search(client, "how does L2 regularization reduce overfitting", top_k=3, retriever="dense")

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


@pytest.mark.parametrize("mode", ["dense", "bm25", "hybrid"])
def test_every_retrieval_mode_finds_the_relevant_page(client, mode):
    upload(client, ML_PDF, "ml.pdf")
    upload(client, COOKING_PDF, "cooking.pdf")

    body = search(client, "gradient descent", top_k=3, retriever=mode).json()

    assert body["retriever"] == mode
    best = body["results"][0]["chunk"]
    assert (best["filename"], best["page_number"]) == ("ml.pdf", 2)


def test_default_mode_is_hybrid_with_component_ranks(client):
    upload(client, ML_PDF, "ml.pdf")

    body = search(client, "gradient descent parameters").json()

    assert body["retriever"] == "hybrid"
    best = body["results"][0]
    assert set(best["component_ranks"]) == {"dense", "bm25"}
    assert set(best["component_scores"]) == {"dense", "bm25"}


def test_bm25_returns_nothing_without_shared_words(client):
    upload(client, ML_PDF, "ml.pdf")

    assert search(client, "bread baking", retriever="bm25").json()["results"] == []


def test_bm25_follows_uploads_and_deletes(client):
    ml = upload(client, ML_PDF, "ml.pdf")
    assert client.get("/health").json()["bm25_chunks"] == 2

    client.delete(f"/api/documents/{ml['document_id']}")

    assert search(client, "gradient descent", retriever="bm25").json()["results"] == []
    assert client.get("/health").json()["bm25_chunks"] == 0


def test_bm25_is_rebuilt_after_restart(app_factory):
    with TestClient(app_factory()) as client:
        upload(client, ML_PDF, "ml.pdf")
        before = search(client, "gradient descent", retriever="bm25").json()["results"]

    with TestClient(app_factory()) as client:
        assert client.get("/health").json()["bm25_chunks"] == 2
        after = search(client, "gradient descent", retriever="bm25").json()["results"]

    assert [r["chunk"]["chunk_id"] for r in after] == [r["chunk"]["chunk_id"] for r in before]


def test_unknown_retrieval_mode_is_rejected(client):
    assert search(client, "anything", retriever="magic").status_code == 422


def _table_page(caption: str, description: str, rows: list[list[str]]) -> str:
    header = ["Stock", "Linear kernel", "Radial kernel", "Polynomial kernel", "RMSE", "MAPE", "RMSE", "MAPE", "RMSE", "MAPE"]
    return "\n".join([caption, description, *header, *[cell for row in rows for cell in row]])


TABLE_PAPER = make_pdf([
    _table_page(
        "Table 7", "SVR results, using the optimal parameter in Table 4 on the daily price test data.",
        [["BBAS3", "0.16427", "0.20141", "1.13716", "1.06158", "19.1481", "32.3253"],
         ["PETR4", "0.06727", "0.40963", "0.57590", "1.00977", "2.18556", "7.60178"],
         ["VALE5", "0.05786", "0.57993", "0.86932", "2.96455", "3.46965", "29.2500"],
         ["BAC", "0.04395", "0.12505", "0.06929", "0.26302", "0.08167", "0.22308"],
         ["GOOGL", "0.06248", "0.03752", "1.97066", "1.30933", "1.18256", "0.72193"],
         ["XOM", "0.07517", "0.06040", "1.08478", "0.87865", "6.87381", "4.13189"]],
    ),
    _table_page(
        "Table 13", "Results of SVR models updated periodically for up-to-the-minute prices (2-years cases).",
        [["BAC", "0.00475", "0.03128", "0.00753", "0.05383", "0.01444", "0.07176"],
         ["GOOGL", "0.01127", "0.07320", "0.01893", "0.12204", "0.02951", "0.18810"],
         ["XOM", "0.00933", "0.05021", "0.01205", "0.06480", "0.02012", "0.10111"]],
    ),
], first_page_label=194)


def test_table_row_is_retrieved_with_its_table_context(app_factory, monkeypatch):
    # Regression for the query "GOOGL linear kernel RMSE 0.06248 MAPE 0.03752": the GOOGL row
    # of Table 7 was split away from the caption and headers, so the chunk was bare numbers.
    monkeypatch.setenv("CHUNK_SIZE_TOKENS", "60")
    monkeypatch.setenv("CHUNK_OVERLAP_TOKENS", "5")
    get_settings.cache_clear()

    with TestClient(app_factory()) as client:
        doc = upload(client, TABLE_PAPER, "paper.pdf")
        chunks = client.get(f"/api/documents/{doc['document_id']}/chunks").json()
        [row_chunk] = [c for c in chunks if "GOOGL" in c["text"] and "0.06248" in c["text"]]
        assert "Table 7" not in row_chunk["text"]  # the split really happened
        assert row_chunk["context"].startswith("Table 7 SVR results")
        assert row_chunk["page_label"] == "194"

        exact = search(client, "GOOGL linear kernel RMSE 0.06248 MAPE 0.03752", top_k=5, retriever="bm25").json()
        assert exact["results"][0]["chunk"]["chunk_id"] == row_chunk["chunk_id"]

        question = "What were the linear kernel RMSE and MAPE for GOOGL on the daily price test data?"
        for mode in ("bm25", "hybrid"):
            results = search(client, question, top_k=3, retriever=mode).json()["results"]
            assert row_chunk["chunk_id"] in [r["chunk"]["chunk_id"] for r in results], mode


def test_corrupt_index_is_rebuilt_from_sqlite(app_factory):
    with TestClient(app_factory()) as client:
        upload(client, ML_PDF, "ml.pdf")

    (get_settings().index_dir / INDEX_FILENAME).write_bytes(b"garbage")

    with TestClient(app_factory()) as client:
        health = client.get("/health").json()
        assert health["retrieval_ready"] is True
        assert health["indexed_vectors"] == 2
        assert top_result(client, "regularization")["filename"] == "ml.pdf"
