"""Streamlit frontend. It only talks to the FastAPI backend over HTTP.

Run from the project root (with the backend already running):
    .\\.venv\\Scripts\\python.exe -m streamlit run frontend/app.py
"""

import os

import httpx
import streamlit as st

BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8000")
UPLOAD_TIMEOUT_SECONDS = 300.0
STATUS_ICONS = {"processing": "⏳", "chunked": "🧩", "indexed": "✅", "failed": "❌"}

st.set_page_config(page_title="Hybrid-Search RAG", layout="wide")


def error_message(response: httpx.Response) -> str:
    """Extract the backend's readable error message, falling back to the status code."""
    try:
        detail = response.json().get("detail")
    except ValueError:
        detail = None
    if isinstance(detail, dict) and "message" in detail:
        return detail["message"]
    return f"Request failed (HTTP {response.status_code})."


def page_display(chunk: dict) -> str:
    """Printed page number when the PDF defines one, plus the physical page if they differ."""
    label, physical = chunk.get("page_label"), chunk["page_number"]
    if label and label != str(physical):
        return f"p. {label} (PDF page {physical})"
    return f"page {physical}"


def api_request(method: str, path: str, **kwargs) -> httpx.Response | None:
    """Send a request to the backend; show an error and return None if it is unreachable."""
    try:
        return httpx.request(method, f"{BACKEND_URL}{path}", **kwargs)
    except httpx.HTTPError as exc:
        st.error(f"Cannot reach backend at {BACKEND_URL} ({exc.__class__.__name__})")
        return None


# ---- Sidebar: system status ---------------------------------------------------------
with st.sidebar:
    st.header("System status")
    health_response = api_request("GET", "/health", timeout=5.0)
    if health_response is None or not health_response.is_success:
        st.stop()
    health = health_response.json()
    if health["retrieval_ready"]:
        st.success("Backend: OK")
    else:
        st.warning("Backend running, but retrieval is unavailable. Check the backend logs.")
    st.caption(f"Embedding model: {health['embedding_model']}")
    st.caption(f"Indexed vectors: {health['indexed_vectors']}")
    st.caption(f"LLM: {health['llm_provider']} / {health['llm_model']}")
    if not health["llm_configured"]:
        st.warning("No LLM API key configured. Question answering will be unavailable.")

st.title("Hybrid-Search RAG with Citation Verification")

list_response = api_request("GET", "/api/documents", timeout=10.0)
documents = list_response.json() if list_response is not None and list_response.is_success else []

documents_tab, retrieval_tab = st.tabs(["Documents", "Retrieval inspector"])

# ---- Documents tab ----------------------------------------------------------------------
with documents_tab:
    with st.form("upload_form", clear_on_submit=True):
        files = st.file_uploader("Upload PDF documents", type=["pdf"], accept_multiple_files=True)
        submitted = st.form_submit_button("Upload and process")

    if submitted and files:
        for uploaded in files:
            with st.spinner(f"Processing {uploaded.name}..."):
                response = api_request(
                    "POST",
                    "/api/documents/upload",
                    files={"file": (uploaded.name, uploaded.getvalue(), "application/pdf")},
                    timeout=UPLOAD_TIMEOUT_SECONDS,
                )
            if response is None:
                break
            if response.status_code == 201:
                doc = response.json()
                empty = f", {len(doc['empty_pages'])} empty page(s)" if doc["empty_pages"] else ""
                st.success(f"{uploaded.name}: {doc['page_count']} pages, {doc['chunk_count']} chunks{empty}")
            else:
                st.error(f"{uploaded.name}: {error_message(response)}")
        if st.button("Refresh list"):
            st.rerun()

    if not documents:
        st.info("No documents uploaded yet.")

    for doc in documents:
        icon = STATUS_ICONS.get(doc["status"], "")
        with st.expander(f"{icon} {doc['filename']} — {doc['status']}"):
            st.write(
                f"**Pages:** {doc['page_count'] or '–'} · **Chunks:** {doc['chunk_count']} · "
                f"**Uploaded:** {doc['created_at']}"
            )
            if doc["empty_pages"]:
                st.warning(f"No extractable text on page(s): {', '.join(map(str, doc['empty_pages']))}")
            if doc["status"] == "failed":
                st.error(f"{doc['error_code']}: {doc['error_message']}")
            st.caption(f"Document ID: {doc['document_id']}")

            col_chunks, col_delete = st.columns([1, 1])
            if col_chunks.button("Show chunks", key=f"chunks_{doc['document_id']}"):
                chunk_response = api_request("GET", f"/api/documents/{doc['document_id']}/chunks", timeout=10.0)
                if chunk_response is not None and chunk_response.is_success:
                    for chunk in chunk_response.json():
                        st.markdown(
                            f"**{chunk['chunk_id']}** · {page_display(chunk)} · "
                            f"{chunk['token_count']} tokens"
                        )
                        st.text(chunk["text"])
            if col_delete.button("Delete", key=f"delete_{doc['document_id']}"):
                delete_response = api_request("DELETE", f"/api/documents/{doc['document_id']}", timeout=30.0)
                if delete_response is not None and delete_response.status_code == 204:
                    st.rerun()
                elif delete_response is not None:
                    st.error(error_message(delete_response))

# ---- Retrieval inspector tab -------------------------------------------------------------
with retrieval_tab:
    st.caption("Shows exactly which passages the retriever returns for a query, before any LLM is involved.")
    indexed = {doc["document_id"]: doc["filename"] for doc in documents if doc["status"] == "indexed"}

    if not indexed:
        st.info("Upload and index a document first.")
    else:
        with st.form("search_form"):
            query = st.text_input("Query")
            col_k, col_docs = st.columns([1, 3])
            top_k = col_k.number_input("Top-k", min_value=1, max_value=50, value=5)
            selected = col_docs.multiselect(
                "Limit to documents (empty = all)", options=list(indexed), format_func=indexed.get
            )
            run_search = st.form_submit_button("Search")

        if run_search and query.strip():
            payload = {"query": query, "top_k": int(top_k), "document_ids": selected or None}
            response = api_request("POST", "/api/retrieval/search", json=payload, timeout=30.0)
            if response is not None and response.is_success:
                body = response.json()
                st.caption(f"{len(body['results'])} result(s) from '{body['retriever']}' in {body['latency_ms']} ms")
                if not body["results"]:
                    st.warning("No passages found.")
                for result in body["results"]:
                    chunk = result["chunk"]
                    st.markdown(
                        f"**#{result['rank']}** · score {result['score']:.3f} · "
                        f"**{chunk['filename']}**, {page_display(chunk)} · `{chunk['chunk_id']}`"
                    )
                    st.text(chunk["text"])
                    st.divider()
            elif response is not None:
                st.error(error_message(response))
