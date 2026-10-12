"""FastAPI entry point.

Run from the project root:
    .\\.venv\\Scripts\\python.exe -m uvicorn backend.main:app --reload
"""

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from backend.api import documents
from backend.config import get_settings
from backend.ingestion.chunker import HuggingFaceTokenizer
from backend.schemas import HealthResponse
from backend.storage.database import Database

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    # Startup: later steps also load the embedding model and FAISS index here, once,
    # instead of on every request.
    settings = get_settings()
    settings.ensure_directories()
    logger.info("Data directory: %s", settings.data_dir)

    app.state.db = Database(settings.database_path)
    app.state.db.initialize()
    # Lazy: the tokenizer files are downloaded on the first upload, not at startup.
    app.state.tokenizer = HuggingFaceTokenizer(settings.embedding_model_name)
    if not settings.llm_configured:
        logger.warning("No LLM API key configured; question answering will be unavailable.")
    yield


app = FastAPI(title="Hybrid-Search RAG with Citation Verification", version="0.1.0", lifespan=lifespan)
app.include_router(documents.router)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Liveness check that also reports the active configuration (never secrets)."""
    settings = get_settings()
    return HealthResponse(
        status="ok",
        app_version=settings.app_version,
        embedding_model=settings.embedding_model_name,
        llm_provider=settings.llm_provider,
        llm_model=settings.llm_model,
        llm_configured=settings.llm_configured,
    )
