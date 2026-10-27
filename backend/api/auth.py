"""Login endpoints."""

import logging
from datetime import UTC, datetime

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field

from backend.api.dependencies import api_error, get_principal
from backend.auth import Principal

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/auth", tags=["auth"])


class LoginRequest(BaseModel):
    username: str = Field(min_length=1, max_length=64)
    password: str = Field(min_length=1, max_length=256)


class LoginResponse(BaseModel):
    access_token: str  # send as "Authorization: Bearer <token>"
    file_token: str  # only for PDF links (?token=...); cannot call other endpoints
    token_type: str = "bearer"
    username: str
    role: str
    expires_at: str


class Me(BaseModel):
    username: str
    role: str
    auth_enabled: bool


@router.post("/login", response_model=LoginResponse)
def login(body: LoginRequest, request: Request) -> LoginResponse:
    auth = request.app.state.auth
    if not auth.enabled:
        raise api_error(400, "auth_disabled", "Login is not enabled on this server.")
    username = body.username.strip().lower()
    wait = auth.throttle.retry_after(username)
    if wait:
        raise HTTPException(status_code=429, headers={"Retry-After": str(wait)}, detail={
            "code": "too_many_attempts",
            "message": f"Too many failed attempts. Try again in {max(1, round(wait / 60))} minute(s)."})

    principal = auth.users.authenticate(username, body.password)
    if principal is None:
        auth.throttle.record_failure(username)
        logger.warning("Failed login for %r", username)
        # One message for unknown user and wrong password: do not reveal which usernames exist.
        raise api_error(401, "invalid_credentials", "Incorrect username or password.")

    auth.throttle.reset(username)
    access_token, expires = auth.tokens.issue(principal, "api")
    file_token, _ = auth.tokens.issue(principal, "file")
    request.state.principal = principal
    logger.info("Login: %s (%s)", principal.username, principal.role)
    return LoginResponse(access_token=access_token, file_token=file_token, username=principal.username,
                         role=principal.role, expires_at=datetime.fromtimestamp(expires, UTC).isoformat(timespec="seconds"))


@router.get("/me", response_model=Me)
def me(request: Request, principal: Principal = Depends(get_principal)) -> Me:
    return Me(username=principal.username, role=principal.role, auth_enabled=request.app.state.auth.enabled)
