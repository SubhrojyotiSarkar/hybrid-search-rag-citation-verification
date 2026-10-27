"""Streamlit frontend. It only talks to the FastAPI backend over HTTP.

Run from the project root (with the backend already running):
    .\\.venv\\Scripts\\python.exe -m streamlit run frontend/app.py
"""

import os

import httpx
import streamlit as st

from evaluation_view import render_evaluation_tab  # sibling module (Streamlit adds this folder to sys.path)

BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8000")
# The address the *browser* uses for "open PDF" links. Differs from BACKEND_URL in Docker,
# where the frontend reaches the backend as http://backend:8000 but the browser cannot.
PUBLIC_BACKEND_URL = os.getenv("PUBLIC_BACKEND_URL", BACKEND_URL)
UPLOAD_TIMEOUT_SECONDS = 300.0
ASK_TIMEOUT_SECONDS = 180.0
STATUS_ICONS = {"processing": "⏳", "chunked": "🧩", "indexed": "✅", "failed": "❌"}

# One distinct presentation per overall status: never a single green "verified" for all.
OVERALL_STATUS_DISPLAY = {
    "unverified": ("warning", "Unverified: citations are attached but were not checked "
                              "(verification is disabled)."),
    "insufficient_evidence": ("info", "Not enough verifiable evidence in the selected documents to answer."),
    "verified_within_provided_evidence": ("success", "Every claim shown was checked against its cited "
                                                     "passages. This means the documents support it, not "
                                                     "that it is true in general."),
    "contains_unresolved_claims": ("warning", "Some claims are not supported by their citations (marked below)."),
    "verification_failed": ("error", "Verification could not be completed due to a technical error. "
                                     "The claims below have NOT been checked."),
}
CLAIM_STATUS_ICONS = {
    "unverified": "◻️", "supported": "✅", "contradicted": "❌",
    "insufficient_evidence": "⚠️", "verification_failed": "⛔",
}

st.set_page_config(page_title="Hybrid-Search RAG", layout="wide")


def error_message(response: httpx.Response) -> str:
    """Extract the backend's readable error message, falling back to the status code."""
    try:
        detail = response.json().get("detail")
    except ValueError:
        detail = None
    request_id = response.headers.get("X-Request-ID")
    reference = f" (request ID {request_id} in the backend log)" if request_id and response.status_code >= 500 else ""
    if isinstance(detail, dict) and "message" in detail:
        return detail["message"] + reference
    return f"Request failed (HTTP {response.status_code}).{reference}"


def page_display(chunk: dict) -> str:
    """Printed page number when the PDF defines one, plus the physical page if they differ."""
    label, physical = chunk.get("page_label"), chunk["page_number"]
    if label and label != str(physical):
        return f"p. {label} (PDF page {physical})"
    return f"page {physical}"


def pdf_link(document_id: str, page_number: int) -> str:
    """Opens the original PDF at the physical page (browsers support #page=N). With login on,
    the link carries the file-only token: it can download PDFs but call nothing else."""
    session = st.session_state.get("auth")
    query = f"?token={session['file_token']}" if session else ""
    return f"{PUBLIC_BACKEND_URL}/api/documents/{document_id}/file{query}#page={page_number}"


SUPPORT_MARKS = {True: "✅ supports", False: "✖ does not support", None: ""}


def render_claim(claim: dict) -> None:
    icon = CLAIM_STATUS_ICONS.get(claim["verification_status"], "")
    attempt = f" · attempt {claim['attempt']}" if claim.get("attempt", 1) > 1 else ""
    st.markdown(f"{icon} **{claim['text']}**  \n*Status: {claim['verification_status']}{attempt}*")
    if claim.get("verification_reason"):
        st.caption(f"Verifier: {claim['verification_reason']}")
    if claim.get("evidence_quote"):
        st.caption(f'Evidence quote (checked to occur in the passage): "{claim["evidence_quote"]}"')
    for citation in claim["citations"]:
        mark = SUPPORT_MARKS.get(citation.get("supports_claim"), "")
        st.markdown(
            f"&nbsp;&nbsp;**[{citation['source_label']}]** {citation['filename']}, "
            f"{page_display(citation)} · [open PDF]({pdf_link(citation['document_id'], citation['page_number'])})"
            + (f" · {mark}" if mark else "")
        )
        st.caption(citation["snippet"])
    if claim["invalid_source_labels"]:
        st.error(f"Cited sources that do not exist: {', '.join(claim['invalid_source_labels'])}")
    if not claim["citations"]:
        st.warning("This claim has no valid citation.")
    st.divider()


def render_answer(response: dict) -> None:
    level, message = OVERALL_STATUS_DISPLAY.get(response["overall_status"], ("info", response["overall_status"]))
    getattr(st, level)(message)
    st.markdown(response["answer"])

    if response["claims"]:
        with st.expander(f"Claims and citations ({len(response['claims'])})", expanded=True):
            for claim in response["claims"]:
                render_claim(claim)

    removed = response.get("removed_claims") or []
    if removed:
        with st.expander(f"Removed claims ({len(removed)}): not supported by their citations"):
            for claim in removed:
                render_claim(claim)

    verification = response.get("verification") or {}
    if verification.get("enabled"):
        parts = [f"verification: {verification['supported']} supported"]
        if verification.get("removed"):
            parts.append(f"{verification['removed']} removed")
        if verification.get("unresolved"):
            parts.append(f"{verification['unresolved']} unresolved")
        if verification.get("failed"):
            parts.append(f"{verification['failed']} failed")
        if verification.get("quote_check_downgrades"):
            parts.append(f"{verification['quote_check_downgrades']} verdict(s) rejected by the quote check")
        parts.append(f"{verification['attempts']} generation attempt(s)")
        st.caption(" · ".join(parts))
        if verification.get("error"):
            st.warning(f"Verification issue: {verification['error']}")

    if response["sources"]:
        with st.expander(f"All passages given to the model ({len(response['sources'])})"):
            for source in response["sources"]:
                chunk = source["chunk"]
                st.markdown(
                    f"**[{source['source_label']}]** {chunk['filename']}, {page_display(chunk)} · "
                    f"retrieval rank {source['retrieval_rank']}, score {source['retrieval_score']:.3f} · "
                    f"[open PDF]({pdf_link(chunk['document_id'], chunk['page_number'])})"
                )
                st.text(chunk["text"])

    meta = response["metadata"]
    parts = [f"retriever: {meta['retriever']} (top {meta['top_k']})", f"retrieval {meta['retrieval_ms']:.0f} ms"]
    if meta.get("llm_model"):
        parts += [f"model: {meta['llm_model']}", f"generation {meta['generation_ms']:.0f} ms"]
        if meta.get("verification_ms") is not None:
            parts.append(f"verification {meta['verification_ms']:.0f} ms")
        parts += [f"LLM calls: {meta.get('llm_calls', 0)}",
                  f"tokens in/out: {meta['input_tokens']}/{meta['output_tokens']}"]
    parts.append(f"total {meta['total_ms']:.0f} ms")
    st.caption(" · ".join(parts))


def api_request(method: str, path: str, **kwargs) -> httpx.Response | None:
    """Send a request to the backend (with the login token, if any); show an error and
    return None if it is unreachable. A 401 means the session expired: log out."""
    session = st.session_state.get("auth")
    if session:
        kwargs["headers"] = {**kwargs.get("headers", {}), "Authorization": f"Bearer {session['access_token']}"}
    try:
        response = httpx.request(method, f"{BACKEND_URL}{path}", **kwargs)
    except httpx.HTTPError as exc:
        st.error(f"Cannot reach backend at {BACKEND_URL} ({exc.__class__.__name__})")
        return None
    if response.status_code == 401 and session:
        st.session_state.pop("auth", None)
        st.warning("Your session has expired. Please log in again.")
        st.rerun()
    return response


def render_login() -> None:
    """Login form; stops the page until the user is logged in."""
    st.title("Hybrid-Search RAG with Citation Verification")
    st.caption("Log in to continue. Ask the administrator for an account.")
    with st.form("login"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        if st.form_submit_button("Log in"):
            try:
                response = httpx.post(f"{BACKEND_URL}/api/auth/login",
                                      json={"username": username, "password": password}, timeout=15.0)
            except httpx.HTTPError as exc:
                st.error(f"Cannot reach backend at {BACKEND_URL} ({exc.__class__.__name__})")
                st.stop()
            if response.is_success:
                st.session_state.auth = response.json()
                st.rerun()
            else:
                st.error(error_message(response))
    st.stop()


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
    st.caption(f"Retrieval: {health['retrieval_mode']}")
    st.caption(f"Reranker: {health['reranker_model']} ({'loaded' if health['reranker_loaded'] else 'loads on first use'})")
    st.caption(f"Indexed vectors: {health['indexed_vectors']} · BM25 chunks: {health['bm25_chunks']}")
    st.caption(f"LLM: {health['llm_provider']} / {health['llm_model']}")
    if not health["llm_configured"]:
        st.warning("No LLM API key configured. Question answering will be unavailable.")

AUTH_ENABLED = health.get("auth_enabled", False)
if AUTH_ENABLED and "auth" not in st.session_state:
    render_login()

session = st.session_state.get("auth")
# Without login (local development) the user has full access. Hiding admin controls is a
# convenience only: the backend refuses admin actions from viewers regardless.
IS_ADMIN = not AUTH_ENABLED or (session and session["role"] == "admin")
with st.sidebar:
    if AUTH_ENABLED:
        st.divider()
        st.caption(f"Logged in as **{session['username']}** ({session['role']})")
        if not IS_ADMIN:
            st.caption("Read-only account: uploading, deleting and evaluation runs are disabled.")
        if st.button("Log out"):
            st.session_state.pop("auth", None)
            st.session_state.pop("history", None)
            st.rerun()
    else:
        st.caption("Login is disabled (local mode).")

st.title("Hybrid-Search RAG with Citation Verification")

list_response = api_request("GET", "/api/documents", timeout=10.0)
documents = list_response.json() if list_response is not None and list_response.is_success else []

ask_tab, documents_tab, retrieval_tab, evaluation_tab = st.tabs(
    ["Ask", "Documents", "Retrieval inspector", "Evaluation"]
)
indexed = {doc["document_id"]: doc["filename"] for doc in documents if doc["status"] == "indexed"}

# ---- Ask tab ------------------------------------------------------------------------------
with ask_tab:
    if "history" not in st.session_state:
        st.session_state.history = []  # list of {"question", "response" | "error"}

    if not indexed:
        st.info("Upload and index a document in the Documents tab first.")
    elif not health["llm_configured"]:
        st.warning("Set GEMINI_API_KEY in .env and restart the backend to ask questions.")
    else:
        col_docs, col_clear = st.columns([4, 1])
        ask_docs = col_docs.multiselect(
            "Answer from documents (empty = all indexed documents)",
            options=list(indexed), format_func=indexed.get, key="ask_docs",
        )
        if col_clear.button("Clear chat"):
            st.session_state.history = []

        for entry in st.session_state.history:
            with st.chat_message("user"):
                st.markdown(entry["question"])
            with st.chat_message("assistant"):
                if "error" in entry:
                    st.error(entry["error"])
                else:
                    render_answer(entry["response"])

        question = st.chat_input("Ask a question about your documents")
        if question and question.strip():
            with st.chat_message("user"):
                st.markdown(question)
            with st.chat_message("assistant"):
                with st.spinner("Retrieving passages and generating a cited answer..."):
                    response = api_request(
                        "POST", "/api/chat/ask",
                        json={"question": question, "document_ids": ask_docs or None},
                        timeout=ASK_TIMEOUT_SECONDS,
                    )
                if response is not None and response.is_success:
                    entry = {"question": question, "response": response.json()}
                    render_answer(entry["response"])
                else:
                    message = error_message(response) if response is not None else "Backend unreachable."
                    entry = {"question": question, "error": message}
                    st.error(message)
            st.session_state.history.append(entry)

# ---- Documents tab ----------------------------------------------------------------------
with documents_tab:
    if not IS_ADMIN:
        st.info("Read-only account: you can browse documents and their chunks, but not upload or delete.")
    with st.form("upload_form", clear_on_submit=True):
        files = st.file_uploader("Upload PDF documents", type=["pdf"], accept_multiple_files=True,
                                 disabled=not IS_ADMIN)
        submitted = st.form_submit_button("Upload and process", disabled=not IS_ADMIN)

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

            col_chunks, col_reprocess, col_delete = st.columns([1, 1, 1])
            if col_reprocess.button("Re-process", key=f"reprocess_{doc['document_id']}", disabled=not IS_ADMIN,
                                    help="Re-extract and re-chunk the stored PDF with the current settings, "
                                         "keeping the document ID. Use after changing chunking settings."):
                with st.spinner("Re-processing..."):
                    reprocess_response = api_request(
                        "POST", f"/api/documents/{doc['document_id']}/reprocess", timeout=UPLOAD_TIMEOUT_SECONDS)
                if reprocess_response is not None and reprocess_response.is_success:
                    st.success(f"Re-processed: {reprocess_response.json()['chunk_count']} chunks.")
                elif reprocess_response is not None:
                    st.error(error_message(reprocess_response))
            if col_chunks.button("Show chunks", key=f"chunks_{doc['document_id']}"):
                chunk_response = api_request("GET", f"/api/documents/{doc['document_id']}/chunks", timeout=10.0)
                if chunk_response is not None and chunk_response.is_success:
                    for chunk in chunk_response.json():
                        st.markdown(
                            f"**{chunk['chunk_id']}** · {page_display(chunk)} · "
                            f"{chunk['token_count']} tokens"
                        )
                        st.text(chunk["text"])
            if col_delete.button("Delete", key=f"delete_{doc['document_id']}", disabled=not IS_ADMIN):
                delete_response = api_request("DELETE", f"/api/documents/{doc['document_id']}", timeout=30.0)
                if delete_response is not None and delete_response.status_code == 204:
                    st.rerun()
                elif delete_response is not None:
                    st.error(error_message(delete_response))

# ---- Retrieval inspector tab -------------------------------------------------------------
with retrieval_tab:
    st.caption("Shows exactly which passages the retriever returns for a query, before any LLM is involved.")

    if not indexed:
        st.info("Upload and index a document first.")
    else:
        with st.form("search_form"):
            query = st.text_input("Query")
            col_mode, col_k, col_docs = st.columns([2, 1, 3])
            mode = col_mode.radio(
                "Retriever", ["hybrid", "hybrid_rerank", "dense", "bm25", "compare all"], horizontal=True,
                help="Hybrid fuses dense and BM25 rankings with Reciprocal Rank Fusion; "
                     "hybrid_rerank reorders hybrid's candidates with a cross-encoder (slower).",
            )
            top_k = col_k.number_input("Top-k", min_value=1, max_value=50, value=5)
            selected = col_docs.multiselect(
                "Limit to documents (empty = all)", options=list(indexed), format_func=indexed.get
            )
            run_search = st.form_submit_button("Search")

        def run(retriever: str) -> dict | None:
            payload = {"query": query, "top_k": int(top_k), "document_ids": selected or None, "retriever": retriever}
            response = api_request("POST", "/api/retrieval/search", json=payload, timeout=30.0)
            if response is not None and response.is_success:
                return response.json()
            if response is not None:
                st.error(error_message(response))
            return None

        if run_search and query.strip() and mode == "compare all":
            columns = st.columns(4)
            for column, retriever in zip(columns, ["dense", "bm25", "hybrid", "hybrid_rerank"]):
                body = run(retriever)
                if body is None:
                    continue
                served_by = body["results"][0]["retriever"] if body["results"] else retriever
                fallback = " ⚠️ reranker failed, showing hybrid order" if served_by != retriever else ""
                column.markdown(f"**{retriever}** · {body['latency_ms']} ms{fallback}")
                for result in body["results"]:
                    chunk = result["chunk"]
                    column.markdown(f"#{result['rank']} · {page_display(chunk)} · `…{chunk['chunk_id'][-10:]}`")
                    if chunk.get("context"):
                        column.caption(f"Table context: {chunk['context'][:120]}…")
                    # The full text: a matching row can be anywhere in the chunk, not just the start.
                    with column.expander(chunk["text"][:80].replace("\n", " ") + "…"):
                        st.text(chunk["text"])
                if not body["results"]:
                    column.caption("No results.")
        elif run_search and query.strip():
            body = run(mode)
            if body is not None:
                st.caption(f"{len(body['results'])} result(s) from '{body['retriever']}' in {body['latency_ms']} ms")
                if not body["results"]:
                    st.warning("No passages found.")
                for result in body["results"]:
                    chunk = result["chunk"]
                    components = ""
                    if result["component_ranks"]:
                        components = " · " + ", ".join(
                            f"{name} #{rank}" for name, rank in sorted(result["component_ranks"].items())
                        )
                    st.markdown(
                        f"**#{result['rank']}** · score {result['score']:.4f}{components} · "
                        f"**{chunk['filename']}**, {page_display(chunk)} · `{chunk['chunk_id']}`"
                    )
                    if chunk.get("context"):
                        st.caption(f"Table context (used for search, not part of the cited text): {chunk['context']}")
                    st.text(chunk["text"])
                    st.divider()

# ---- Evaluation tab ---------------------------------------------------------------------------
with evaluation_tab:
    render_evaluation_tab(api_request, error_message, page_display, is_admin=IS_ADMIN)
