"""FastAPI entry point.

Run from the project root:
    .\\.venv\\Scripts\\python.exe -m uvicorn backend.main:app --reload
"""

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request

from backend.api import documents, search
from backend.config import get_settings
from backend.ingestion.chunker import HuggingFaceTokenizer, Tokenizer
from backend.ingestion.pipeline import reconcile_index
from backend.retrieval.embeddings import Embedder, SentenceTransformerEmbedder
from backend.retrieval.vector_store import FaissVectorStore
from backend.schemas import HealthResponse
from backend.storage.database import Database
from backend.storage.repositories import DocumentRepository

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def create_app(embedder: Embedder | None = None, tokenizer: Tokenizer | None = None) -> FastAPI:
    """Build the app. Tests pass lightweight fakes; production loads the real models."""

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        # Everything expensive is loaded once here, not on every request.
        settings = get_settings()
        settings.ensure_directories()
        logger.info("Data directory: %s", settings.data_dir)

        app.state.db = Database(settings.database_path)
        app.state.db.initialize()
        # Lazy: tokenizer files are downloaded on the first upload, not at startup.
        app.state.tokenizer = tokenizer or HuggingFaceTokenizer(settings.embedding_model_name)

        # If the embedding model or index cannot be loaded, the app still starts so that
        # document listing works; retrieval endpoints then return 503 instead of crashing.
        app.state.embedder = None
        app.state.vector_store = None
        try:
            active_embedder = embedder or SentenceTransformerEmbedder(
                settings.embedding_model_name, settings.embedding_batch_size
            )
            store = FaissVectorStore.open(settings.index_dir, active_embedder.model_name, active_embedder.dimension)
            reconcile_index(DocumentRepository(app.state.db), store, active_embedder)
            app.state.embedder = active_embedder
            app.state.vector_store = store
        except Exception:
            logger.exception("Retrieval is unavailable: embedding model or index failed to load")

        if not settings.llm_configured:
            logger.warning("No LLM API key configured; question answering will be unavailable.")
        yield

    app = FastAPI(
        title="Hybrid-Search RAG with Citation Verification", version="0.1.0", lifespan=lifespan
    )
    app.include_router(documents.router)
    app.include_router(search.router)

    @app.get("/health", response_model=HealthResponse)
    def health(request: Request) -> HealthResponse:
        """Liveness check that also reports the active configuration (never secrets)."""
        settings = get_settings()
        store = request.app.state.vector_store
        retrieval_ready = request.app.state.embedder is not None and store is not None
        return HealthResponse(
            status="ok" if retrieval_ready else "degraded",
            app_version=settings.app_version,
            embedding_model=settings.embedding_model_name,
            retrieval_ready=retrieval_ready,
            indexed_vectors=store.count if store is not None else 0,
            llm_provider=settings.llm_provider,
            llm_model=settings.llm_model,
            llm_configured=settings.llm_configured,
        )

    return app


app = create_app()
