"""Shared Pydantic contracts between the backend modules, the API, and the frontend.

Keeping every request/response shape in one module prevents the API, the pipeline and
the UI from drifting apart. Document, chunk, citation and claim models are added here
as each pipeline stage is built.
"""

from pydantic import BaseModel


class HealthResponse(BaseModel):
    status: str
    app_version: str
    embedding_model: str
    llm_provider: str
    llm_model: str
    llm_configured: bool
