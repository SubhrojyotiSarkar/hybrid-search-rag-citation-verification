"""Streamlit frontend. It only talks to the FastAPI backend over HTTP.

Run from the project root (with the backend already running):
    .\\.venv\\Scripts\\python.exe -m streamlit run frontend/app.py
"""

import os

import httpx
import streamlit as st

BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8000")

st.set_page_config(page_title="Hybrid-Search RAG", layout="wide")


def fetch_health() -> tuple[dict | None, str | None]:
    """Return (health payload, None) on success or (None, error message) on failure."""
    try:
        response = httpx.get(f"{BACKEND_URL}/health", timeout=5.0)
        response.raise_for_status()
        return response.json(), None
    except httpx.HTTPError as exc:
        return None, f"Cannot reach backend at {BACKEND_URL} ({exc.__class__.__name__})"


with st.sidebar:
    st.header("System status")
    health, error = fetch_health()
    if health:
        st.success("Backend: OK")
        st.caption(f"Embedding model: {health['embedding_model']}")
        st.caption(f"LLM: {health['llm_provider']} / {health['llm_model']}")
        if not health["llm_configured"]:
            st.warning("No LLM API key configured. Question answering will be unavailable.")
    else:
        st.error(error)

st.title("Hybrid-Search RAG with Citation Verification")
st.info("Document upload and question answering are added in the next steps.")
