"""Logging setup: one format for the whole backend, with the current request's ID on every
line, so all log lines of one request (retrieval, LLM calls, verification) can be found
together. The request ID is also returned to the client in the X-Request-ID header and in
error bodies, so a user-visible error can be matched to its log lines."""

import logging
import re
import uuid
from contextvars import ContextVar

request_id_var: ContextVar[str] = ContextVar("request_id", default="-")

# Accept a caller-supplied ID only if it is short and harmless in logs.
_SAFE_REQUEST_ID = re.compile(r"^[A-Za-z0-9._-]{1,64}$")


class RequestIdFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = request_id_var.get()
        return True


def configure_logging(level: str = "INFO") -> None:
    handler = logging.StreamHandler()
    handler.addFilter(RequestIdFilter())
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s [%(request_id)s] %(message)s"))
    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(level.upper())
    # Third-party libraries are chatty at INFO (HTTP requests, model downloads).
    for noisy in ("httpx", "httpcore", "urllib3", "huggingface_hub", "sentence_transformers", "google_genai"):
        logging.getLogger(noisy).setLevel(max(logging.WARNING, root.level))


def new_request_id(supplied: str | None) -> str:
    return supplied if supplied and _SAFE_REQUEST_ID.match(supplied) else uuid.uuid4().hex[:12]
