"""FastAPI entry point.

Run from the project root:
    .\\.venv\\Scripts\\python.exe -m uvicorn backend.main:app --reload
"""

import logging
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, Request
from fastapi.responses import JSONResponse

from backend.api import auth as auth_routes
from backend.api import chat, documents, evaluation, search
from backend.api.dependencies import get_principal
from backend.auth import AuthContext, build_auth
from backend.config import get_settings
from backend.logging_config import configure_logging, new_request_id, request_id_var
from backend.evaluation.jobs import JobManager
from backend.generation.llm_client import LLMClient, create_llm_client
from backend.ingestion.chunker import HuggingFaceTokenizer, Tokenizer
from backend.ingestion.pipeline import reconcile_index
from backend.retrieval.bm25_search import BM25Index
from backend.retrieval.embeddings import Embedder, SentenceTransformerEmbedder
from backend.retrieval.hybrid_search import RetrievalServices
from backend.retrieval.reranker import CrossEncoderReranker, Reranker
from backend.retrieval.vector_store import FaissVectorStore
from backend.schemas import HealthResponse
from backend.storage.database import Database
from backend.storage.repositories import DocumentRepository

configure_logging(get_settings().log_level)
logger = logging.getLogger(__name__)
access_logger = logging.getLogger("backend.access")


def create_app(
    embedder: Embedder | None = None,
    tokenizer: Tokenizer | None = None,
    llm_client: LLMClient | None = None,
    reranker: Reranker | None = None,
) -> FastAPI:
    """Build the app. Tests pass lightweight fakes; production loads the real models."""

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        # Everything expensive is loaded once here, not on every request.
        settings = get_settings()
        settings.ensure_directories()
        logger.info("Data directory: %s", settings.data_dir)

        # Login first: if it is enabled but misconfigured, refuse to start (fail closed)
        # rather than serve an unprotected app.
        app.state.auth = build_auth(settings)
        if app.state.auth.enabled:
            logger.info("Login enabled: %d account(s) in %s", len(app.state.auth.users.list_users()),
                        settings.auth_users_path)
        else:
            logger.warning("Login is DISABLED (AUTH_ENABLED=false): anyone who can reach this server has "
                           "full access. Enable it before exposing the app beyond your own machine.")

        app.state.db = Database(settings.database_path)
        app.state.db.initialize()
        # Lazy: tokenizer files are downloaded on the first upload, not at startup.
        app.state.tokenizer = tokenizer or HuggingFaceTokenizer(settings.embedding_model_name)

        # If the embedding model or index cannot be loaded, the app still starts so that
        # document listing works; retrieval endpoints then return 503 instead of crashing.
        app.state.retrieval = None
        try:
            repository = DocumentRepository(app.state.db)
            active_embedder = embedder or SentenceTransformerEmbedder(
                settings.embedding_model_name, settings.embedding_batch_size
            )
            store = FaissVectorStore.open(settings.index_dir, active_embedder.model_name, active_embedder.dimension)
            reconcile_index(repository, store, active_embedder)
            # BM25 is rebuilt from SQLite on every start (after reconciliation, so it sees
            # exactly the INDEXED documents that FAISS holds).
            bm25_index = BM25Index(settings.bm25_k1, settings.bm25_b)
            bm25_index.add(repository.indexed_chunk_texts())
            logger.info("BM25 index built over %d chunks", bm25_index.count)
            # Lazy: the cross-encoder loads on the first reranked query, not at startup.
            active_reranker = reranker or CrossEncoderReranker(
                settings.reranker_model_name, settings.reranker_batch_size
            )
            app.state.retrieval = RetrievalServices(active_embedder, store, bm25_index, active_reranker)
        except Exception:
            logger.exception("Retrieval is unavailable: embedding model or index failed to load")

        # Constructing the client makes no network call; a bad key surfaces on first use.
        # A misconfigured provider disables question answering instead of stopping the app.
        try:
            app.state.llm = llm_client or create_llm_client(settings)
        except ValueError as exc:
            logger.error("LLM is unavailable: %s. Set LLM_PROVIDER=gemini in .env.", exc)
            app.state.llm = None
        if app.state.llm is None:
            logger.warning("No LLM configured (GEMINI_API_KEY); question answering will be unavailable.")
        else:
            logger.info(
                "LLM: %s / %s (thinking_level=%s)",
                settings.llm_provider, settings.llm_model, settings.gemini_thinking_level,
            )
        yield

    app = FastAPI(
        title="Hybrid-Search RAG with Citation Verification", version=get_settings().app_version, lifespan=lifespan
    )
    # Every /api route needs a logged-in user (no-op while login is disabled); routes that
    # change data additionally require an admin (declared on the routes themselves).
    logged_in = [Depends(get_principal)]
    app.include_router(auth_routes.router)
    app.include_router(documents.router)  # per-route: reads / admin changes / PDF file tokens
    app.include_router(search.router, dependencies=logged_in)
    app.include_router(chat.router, dependencies=logged_in)
    app.include_router(evaluation.router, dependencies=logged_in)
    app.state.jobs = JobManager()  # background evaluation runs (one at a time)
    app.state.auth = AuthContext(enabled=False)  # replaced at startup by build_auth()

    @app.middleware("http")
    async def request_context(request: Request, call_next):
        """Give every request an ID (logged on every line, returned in X-Request-ID), log one
        access line with latency, and turn unhandled exceptions into a clean JSON 500."""
        request_id = new_request_id(request.headers.get("X-Request-ID"))
        token = request_id_var.set(request_id)
        started = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception:
            # Full details go to the log only; the client gets a stable code and the ID.
            logger.exception("Unhandled error in %s %s", request.method, request.url.path)
            response = JSONResponse(status_code=500, content={"detail": {
                "code": "internal_error", "message": "An unexpected error occurred.", "request_id": request_id}})
        duration_ms = (time.perf_counter() - started) * 1000
        log = access_logger.debug if request.url.path == "/health" else access_logger.info
        principal = getattr(request.state, "principal", None)
        user = f" user={principal.username}" if principal else ""
        log("%s %s -> %d (%.0f ms)%s", request.method, request.url.path, response.status_code, duration_ms, user)
        request_id_var.reset(token)
        response.headers["X-Request-ID"] = request_id
        return response

    @app.get("/health", response_model=HealthResponse)
    def health(request: Request) -> HealthResponse:
        """Liveness check that also reports the active configuration (never secrets)."""
        settings = get_settings()
        retrieval = request.app.state.retrieval
        return HealthResponse(
            status="ok" if retrieval is not None else "degraded",
            app_version=settings.app_version,
            embedding_model=settings.embedding_model_name,
            retrieval_ready=retrieval is not None,
            retrieval_mode=settings.retrieval_mode,
            auth_enabled=request.app.state.auth.enabled,
            indexed_vectors=retrieval.vector_store.count if retrieval else 0,
            bm25_chunks=retrieval.bm25_index.count if retrieval else 0,
            reranker_model=settings.reranker_model_name,
            reranker_loaded=bool(retrieval and getattr(retrieval.reranker, "loaded", True)),
            llm_provider=settings.llm_provider,
            llm_model=settings.llm_model,
            llm_configured=request.app.state.llm is not None,
        )

    return app


app = create_app()
