"""Application settings, loaded from environment variables and the local .env file.

Every tunable value (paths, model names, chunk sizes, LLM provider) lives here so that
experiments are reproducible: the active configuration can be logged and compared.
Environment variables take precedence over values in .env.
"""

import os
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import AliasChoices, Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# RAG_ENV_FILE overrides which settings file is read; an empty value reads none. The test
# suite sets it to "" so a developer's personal .env (e.g. AUTH_ENABLED=true) cannot change
# test behaviour.
_env_file_override = os.environ.get("RAG_ENV_FILE")
if _env_file_override is None:
    ENV_FILE: Path | None = PROJECT_ROOT / ".env"
else:
    ENV_FILE = Path(_env_file_override) if _env_file_override else None


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=ENV_FILE,
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_version: str = "1.0.0"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"

    # Storage. Relative paths are resolved against the project root, not the CWD,
    # so the backend finds the same data no matter where it is launched from.
    data_dir: Path = Path("data")

    # Upload limits.
    max_upload_mb: int = Field(default=25, gt=0)

    # Pages with fewer visible (non-whitespace) characters are reported as empty,
    # e.g. scanned images or blank pages.
    min_page_chars: int = Field(default=20, ge=1)

    # Embeddings and chunking. all-MiniLM-L6-v2 truncates input at 256 word-pieces,
    # so chunks are kept below that limit (measured with the model's own tokenizer).
    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    chunk_size_tokens: int = Field(default=220, gt=0)
    chunk_overlap_tokens: int = Field(default=40, ge=0)
    embedding_batch_size: int = Field(default=32, gt=0)

    # Retrieval. retrieval_top_k passages are sent to the LLM (DENSE_TOP_K, its Phase 1
    # name, is still accepted).
    retrieval_mode: Literal["dense", "bm25", "hybrid", "hybrid_rerank"] = "hybrid"
    retrieval_top_k: int = Field(
        default=8, gt=0, validation_alias=AliasChoices("retrieval_top_k", "dense_top_k")
    )
    # Hybrid: candidates taken from each retriever before fusion, and the RRF constant.
    hybrid_candidates_per_retriever: int = Field(default=20, gt=0)
    rrf_k: int = Field(default=60, ge=0)
    # BM25: term-frequency saturation (k1) and document-length normalization (b).
    bm25_k1: float = Field(default=1.5, ge=0)
    bm25_b: float = Field(default=0.75, ge=0, le=1)
    # Cross-encoder reranking (retrieval_mode "hybrid_rerank"): how many first-stage
    # candidates are rescored. More candidates = better recall, linearly more latency.
    reranker_model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    rerank_candidates: int = Field(default=30, gt=0, le=100)
    reranker_batch_size: int = Field(default=32, gt=0)

    # Citation verification (one extra LLM call per answer; recovery adds up to 2 per attempt).
    verification_enabled: bool = True
    # Regenerate (with more evidence and the verifier's feedback) at most this many times.
    max_recovery_attempts: int = Field(default=1, ge=0, le=2)
    # Passages retrieved for a recovery attempt (more evidence than the first attempt).
    recovery_top_k: int = Field(default=12, gt=0, le=50)
    # "remove": unsupported claims are dropped from the answer (and listed separately);
    # "flag": they stay in the answer, marked, and the answer is "contains_unresolved_claims".
    unsupported_claim_policy: Literal["remove", "flag"] = "remove"
    verification_max_claims_per_call: int = Field(default=10, gt=0)

    # Login. Off by default for local development; turn it on before exposing the app
    # anywhere. With it on, the backend refuses to start without a secret key and an admin.
    auth_enabled: bool = False
    auth_secret_key: SecretStr | None = None
    auth_token_ttl_minutes: int = Field(default=480, gt=0, le=10080)
    # Defaults to <DATA_DIR>/users.json (gitignored, kept with the rest of the data).
    auth_users_file: Path | None = None

    # Evaluation: datasets, run results and human reviews live under this folder.
    evaluation_dir: Path = Path("evaluation")
    # Pace answer-evaluation runs to the provider's per-minute request quota.
    eval_requests_per_minute: int = Field(default=5, gt=0)

    # LLM provider (Google Gemini). The key is a SecretStr so it never appears in logs.
    llm_provider: str = "gemini"
    gemini_api_key: SecretStr | None = None
    # A free-tier model by default. Changing it may select a model without a free tier.
    gemini_model: str = "gemini-3.8-flash"
    # Reasoning depth: higher is more thorough but slower and uses more of the quota.
    gemini_thinking_level: Literal["low", "medium", "high"] = "medium"
    # Includes Gemini's thinking tokens.
    llm_max_tokens: int = Field(default=16000, gt=0)
    llm_timeout_seconds: float = Field(default=120.0, gt=0)

    @field_validator("data_dir", "evaluation_dir")
    @classmethod
    def _resolve_data_dir(cls, value: Path) -> Path:
        return value if value.is_absolute() else (PROJECT_ROOT / value).resolve()

    @model_validator(mode="after")
    def _check_chunking(self) -> "Settings":
        if self.chunk_overlap_tokens >= self.chunk_size_tokens:
            raise ValueError("chunk_overlap_tokens must be smaller than chunk_size_tokens")
        return self

    @property
    def upload_dir(self) -> Path:
        return self.data_dir / "uploads"

    @property
    def index_dir(self) -> Path:
        return self.data_dir / "indexes"

    @property
    def auth_users_path(self) -> Path:
        if self.auth_users_file is None:
            return self.data_dir / "users.json"
        return self.auth_users_file if self.auth_users_file.is_absolute() else (PROJECT_ROOT / self.auth_users_file).resolve()

    @property
    def eval_results_dir(self) -> Path:
        return self.evaluation_dir / "results"

    @property
    def eval_reviews_dir(self) -> Path:
        return self.evaluation_dir / "reviews"

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024

    @property
    def database_path(self) -> Path:
        return self.data_dir / "app.db"

    @property
    def llm_model(self) -> str:
        """Provider-neutral name used in health checks and logs."""
        return self.gemini_model

    @property
    def llm_configured(self) -> bool:
        if self.gemini_api_key is None:
            return False
        key = self.gemini_api_key.get_secret_value().strip()
        # The .env.example placeholder does not count as a configured key.
        return bool(key) and key != "your-api-key-here"

    def ensure_directories(self) -> None:
        for directory in (self.data_dir, self.upload_dir, self.index_dir):
            directory.mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    """Return the process-wide settings (cached; call get_settings.cache_clear() in tests)."""
    return Settings()
