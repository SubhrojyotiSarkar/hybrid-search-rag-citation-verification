"""Provider-agnostic LLM interface plus the Google Gemini implementation.

The rest of the application calls generate_structured() and receives a validated Pydantic
object, or an LLMError with a stable code. Switching providers means adding one class
here; nothing in retrieval, generation or the API changes.
"""

import logging
import time
from dataclasses import dataclass
from typing import Generic, Protocol, TypeVar

from pydantic import BaseModel, ValidationError

from backend.config import Settings

logger = logging.getLogger(__name__)

SchemaT = TypeVar("SchemaT", bound=BaseModel)


class LLMError(Exception):
    """A failed LLM call. `code` is stable and safe to show; details go to the logs only."""

    def __init__(self, code: str, message: str, retryable: bool = False) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.retryable = retryable


@dataclass
class LLMResult(Generic[SchemaT]):
    output: SchemaT
    model: str  # the model version that actually answered
    input_tokens: int
    output_tokens: int
    latency_ms: float


@dataclass
class LLMUsage:
    """Totals over every LLM call made while answering one question."""

    calls: int = 0
    verification_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    generation_ms: float = 0.0
    verification_ms: float = 0.0
    model: str | None = None  # model that produced the (first) answer

    def add(self, result: "LLMResult", stage: str) -> None:
        self.calls += 1
        self.input_tokens += result.input_tokens
        self.output_tokens += result.output_tokens
        if stage == "verification":
            self.verification_calls += 1
            self.verification_ms += result.latency_ms
        else:
            self.generation_ms += result.latency_ms
            self.model = self.model or result.model


class LLMClient(Protocol):
    model: str

    def generate_structured(
        self, system: str, user: str, schema: type[SchemaT], max_tokens: int | None = None
    ) -> LLMResult[SchemaT]:
        """Return output validated against `schema`, or raise LLMError."""
        ...


# Finish reasons meaning the model or its safety filters declined to produce the answer.
GEMINI_REFUSAL_FINISH_REASONS = {
    "SAFETY", "PROHIBITED_CONTENT", "BLOCKLIST", "SPII", "RECITATION", "LANGUAGE",
}
# HTTP statuses the SDK retries with exponential backoff before giving up.
GEMINI_RETRY_STATUSES = [408, 429, 500, 502, 503, 504]


class GeminiLLMClient:
    """Gemini via Google's official google-genai SDK, using JSON-schema structured output.

    The response is constrained to the JSON schema of the Pydantic model, then validated
    into that model here; any mismatch becomes LLMError("llm_invalid_output"). The finish
    reason is checked before parsing, so refusals and truncation are reported as such.
    """

    def __init__(
        self, api_key: str, model: str, thinking_level: str, max_tokens: int, timeout_seconds: float,
        retry_attempts: int = 3,
    ) -> None:
        from google import genai
        from google.genai import types

        self._types = types
        self.client = genai.Client(
            api_key=api_key,
            http_options=types.HttpOptions(
                timeout=int(timeout_seconds * 1000),  # milliseconds
                retry_options=types.HttpRetryOptions(
                    attempts=retry_attempts, http_status_codes=GEMINI_RETRY_STATUSES
                ),
            ),
        )
        self.model = model
        self.thinking_level = thinking_level
        self.max_tokens = max_tokens

    def generate_structured(
        self, system: str, user: str, schema: type[SchemaT], max_tokens: int | None = None
    ) -> LLMResult[SchemaT]:
        import httpx
        from google.genai import errors

        types = self._types
        config = types.GenerateContentConfig(
            system_instruction=system,
            response_mime_type="application/json",
            response_json_schema=schema.model_json_schema(),
            # Thinking tokens count towards this limit on Gemini.
            max_output_tokens=max_tokens or self.max_tokens,
            thinking_config=types.ThinkingConfig(thinking_level=types.ThinkingLevel[self.thinking_level.upper()]),
        )
        started = time.perf_counter()
        try:
            response = self.client.models.generate_content(model=self.model, contents=user, config=config)
        except errors.APIError as exc:
            raise _map_api_error(exc, self.model) from exc
        except httpx.HTTPError as exc:  # timeouts and connection failures
            raise LLMError("llm_unavailable", "Could not reach the Gemini API.", retryable=True) from exc

        latency_ms = (time.perf_counter() - started) * 1000
        usage = response.usage_metadata
        input_tokens = (usage.prompt_token_count or 0) if usage else 0
        # Thinking tokens are generated (and count against quotas) like answer tokens.
        output_tokens = ((usage.candidates_token_count or 0) + (usage.thoughts_token_count or 0)) if usage else 0
        candidate = response.candidates[0] if response.candidates else None
        finish_reason = candidate.finish_reason.name if candidate and candidate.finish_reason else None
        logger.info(
            "LLM call: model=%s finish=%s in=%d out=%d %.0fms response_id=%s",
            response.model_version, finish_reason, input_tokens, output_tokens, latency_ms, response.response_id,
        )

        # Check why generation stopped before trusting the content.
        block_reason = response.prompt_feedback.block_reason if response.prompt_feedback else None
        if block_reason or finish_reason in GEMINI_REFUSAL_FINISH_REASONS:
            raise LLMError("llm_refused", "The model declined to answer this request.")
        if finish_reason == "MAX_TOKENS":
            raise LLMError("llm_truncated", "The model's answer was cut off before it finished.")
        text = _response_text(response)
        if candidate is None or not text:
            raise LLMError("llm_invalid_output", "The LLM returned no answer.")
        try:
            output = schema.model_validate_json(text)
        except ValidationError as exc:
            logger.warning("LLM output failed schema validation: %s", exc)
            raise LLMError("llm_invalid_output", "The LLM returned output in an unexpected format.") from exc

        return LLMResult(
            output=output,
            model=response.model_version or self.model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=latency_ms,
        )


def _response_text(response) -> str:
    """Concatenate the answer's text parts, skipping thought summaries."""
    if not response.candidates or not response.candidates[0].content:
        return ""
    parts = response.candidates[0].content.parts or []
    return "".join(part.text for part in parts if part.text and not part.thought).strip()


def _map_api_error(exc, model: str) -> LLMError:
    """Translate a google-genai APIError into a stable LLMError code."""
    code = exc.code or 0
    status = (exc.status or "").upper()
    message = (exc.message or "").lower()
    logger.error("Gemini API error %s %s: %s", code, status, exc.message)
    # Order matters: quota messages can mention the API key, so check 429 first.
    if code == 429 or status == "RESOURCE_EXHAUSTED":
        return LLMError(
            "llm_rate_limited",
            "The Gemini rate limit or free-tier quota was reached. Wait a minute and try again.",
            retryable=True,
        )
    if code == 404 or status == "NOT_FOUND":
        return LLMError("llm_model_not_found", f"Model '{model}' was not found. Check GEMINI_MODEL.")
    if code == 403 or status == "PERMISSION_DENIED":
        return LLMError("llm_permission_denied", "The API key cannot use this model.")
    # An invalid key is reported as 400 INVALID_ARGUMENT with "API key not valid".
    if code == 401 or status == "UNAUTHENTICATED" or (code == 400 and "api key" in message):
        return LLMError("llm_auth_failed", "The Gemini API key was rejected. Check GEMINI_API_KEY.")
    if code == 400:
        return LLMError("llm_bad_request", "The Gemini API rejected the request.")
    return LLMError("llm_unavailable", "The Gemini API returned an error.", retryable=code >= 500)


def create_llm_client(settings: Settings) -> LLMClient | None:
    """Build the configured client, or None if no credentials are configured."""
    if not settings.llm_configured:
        return None
    if settings.llm_provider != "gemini":
        raise ValueError(f"Unsupported LLM_PROVIDER '{settings.llm_provider}'; expected 'gemini'")
    return GeminiLLMClient(
        api_key=settings.gemini_api_key.get_secret_value(),
        model=settings.gemini_model,
        thinking_level=settings.gemini_thinking_level,
        max_tokens=settings.llm_max_tokens,
        timeout_seconds=settings.llm_timeout_seconds,
    )
