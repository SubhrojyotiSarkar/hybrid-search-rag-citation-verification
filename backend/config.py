"""Application settings, loaded from environment variables and the local .env file.

Every tunable value (paths, model names, chunk sizes, LLM provider) lives here so that
experiments are reproducible: the active configuration can be logged and compared.
Environment variables take precedence over values in .env.
"""

from functools import lru_cache
from pathlib import Path

from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_version: str = "0.1.0"

    # Storage. Relative paths are resolved against the project root, not the CWD,
    # so the backend finds the same data no matter where it is launched from.
    data_dir: Path = Path("data")

    # Upload limits.
    max_upload_mb: int = Field(default=25, gt=0)

    # Embeddings and chunking. all-MiniLM-L6-v2 truncates input at 256 word-pieces,
    # so chunks are kept below that limit (measured with the model's own tokenizer).
    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    chunk_size_tokens: int = Field(default=220, gt=0)
    chunk_overlap_tokens: int = Field(default=40, ge=0)

    # Retrieval.
    dense_top_k: int = Field(default=8, gt=0)

    # LLM provider. The key is a SecretStr so it never appears in logs or reprs.
    llm_provider: str = "anthropic"
    llm_model: str = "claude-opus-5-5"
    anthropic_api_key: SecretStr | None = None

    @field_validator("data_dir")
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
    def database_path(self) -> Path:
        return self.data_dir / "app.db"

    @property
    def llm_configured(self) -> bool:
        return self.anthropic_api_key is not None and bool(self.anthropic_api_key.get_secret_value())

    def ensure_directories(self) -> None:
        for directory in (self.data_dir, self.upload_dir, self.index_dir):
            directory.mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    """Return the process-wide settings (cached; call get_settings.cache_clear() in tests)."""
    return Settings()
