"""Authentication: password hashing, the user store, signed tokens and login throttling.

Only the Python standard library is used:
- passwords: salted scrypt (hashlib.scrypt), compared in constant time;
- tokens: base64url(JSON payload) + "." + base64url(HMAC-SHA256 signature), with an expiry
  and a scope. "api" tokens authorize API calls; "file" tokens can only download PDFs
  (they travel in "open PDF" links, where a browser cannot send an Authorization header).

Roles: "admin" can change things (upload, delete, re-process, evaluation runs, reviews);
"viewer" is read-only (ask, browse, inspect, view evaluation results).
"""

import base64
import hashlib
import hmac
import json
import os
import re
import secrets
import threading
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

ROLES = ("admin", "viewer")
USERNAME_PATTERN = re.compile(r"^[a-z0-9][a-z0-9._-]{1,31}$")
MIN_PASSWORD_LENGTH = 10
MIN_SECRET_KEY_LENGTH = 32

# scrypt cost parameters (~16 MB memory, tens of ms per hash): slow for attackers, fine for logins.
_SCRYPT_N, _SCRYPT_R, _SCRYPT_P = 2**14, 8, 1


@dataclass(frozen=True)
class Principal:
    username: str
    role: str

    @property
    def is_admin(self) -> bool:
        return self.role == "admin"


def _b64encode(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def _b64decode(text: str) -> bytes:
    return base64.urlsafe_b64decode(text + "=" * (-len(text) % 4))


# ---- passwords ----------------------------------------------------------------------------

def hash_password(password: str) -> str:
    salt = os.urandom(16)
    digest = hashlib.scrypt(password.encode("utf-8"), salt=salt, n=_SCRYPT_N, r=_SCRYPT_R, p=_SCRYPT_P)
    return f"scrypt${_SCRYPT_N}${_SCRYPT_R}${_SCRYPT_P}${_b64encode(salt)}${_b64encode(digest)}"


def verify_password(password: str, stored: str) -> bool:
    try:
        scheme, n, r, p, salt, expected = stored.split("$")
        if scheme != "scrypt":
            return False
        digest = hashlib.scrypt(password.encode("utf-8"), salt=_b64decode(salt), n=int(n), r=int(r), p=int(p))
    except (ValueError, TypeError):
        return False
    return hmac.compare_digest(digest, _b64decode(expected))


def validate_new_credentials(username: str, password: str, role: str) -> None:
    if not USERNAME_PATTERN.match(username):
        raise ValueError("Username: 2-32 characters, lowercase letters, digits, '.', '_' or '-'.")
    if len(password) < MIN_PASSWORD_LENGTH:
        raise ValueError(f"Password must be at least {MIN_PASSWORD_LENGTH} characters.")
    if role not in ROLES:
        raise ValueError(f"Role must be one of {ROLES}.")


# ---- users ------------------------------------------------------------------------------------

class UserStore:
    """Users in a JSON file: {"users": {name: {"role", "password_hash", "created_at"}}}.
    The file holds only password hashes and is kept out of git (it lives under data/)."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._lock = threading.Lock()

    def _read(self) -> dict:
        if not self.path.exists():
            return {}
        return json.loads(self.path.read_text(encoding="utf-8")).get("users", {})

    def _write(self, users: dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps({"users": users}, indent=2), encoding="utf-8")
        tmp.replace(self.path)

    def list_users(self) -> dict[str, str]:
        return {name: info["role"] for name, info in self._read().items()}

    def add_user(self, username: str, password: str, role: str) -> None:
        validate_new_credentials(username, password, role)
        with self._lock:
            users = self._read()
            users[username] = {"role": role, "password_hash": hash_password(password),
                               "created_at": datetime.now(UTC).isoformat(timespec="seconds")}
            self._write(users)

    def remove_user(self, username: str) -> bool:
        with self._lock:
            users = self._read()
            if users.pop(username, None) is None:
                return False
            self._write(users)
            return True

    def authenticate(self, username: str, password: str) -> Principal | None:
        user = self._read().get(username)
        if user is None:
            # Spend the same time as a real check, so response time doesn't reveal which
            # usernames exist.
            verify_password(password, _DUMMY_HASH)
            return None
        if not verify_password(password, user["password_hash"]):
            return None
        return Principal(username, user["role"])


_DUMMY_HASH = hash_password(secrets.token_urlsafe(16))


# ---- tokens -----------------------------------------------------------------------------------

class TokenService:
    def __init__(self, secret_key: str, ttl_seconds: int, clock=time.time) -> None:
        if len(secret_key) < MIN_SECRET_KEY_LENGTH:
            raise ValueError(f"AUTH_SECRET_KEY must be at least {MIN_SECRET_KEY_LENGTH} characters.")
        self._key = secret_key.encode("utf-8")
        self.ttl_seconds = ttl_seconds
        self.clock = clock

    def _sign(self, payload: bytes) -> str:
        return _b64encode(hmac.new(self._key, payload, hashlib.sha256).digest())

    def issue(self, principal: Principal, scope: str = "api") -> tuple[str, int]:
        expires = int(self.clock()) + self.ttl_seconds
        payload = json.dumps({"sub": principal.username, "role": principal.role, "scope": scope, "exp": expires},
                             separators=(",", ":"), sort_keys=True).encode("utf-8")
        encoded = _b64encode(payload)
        return f"{encoded}.{self._sign(encoded.encode('ascii'))}", expires

    def verify(self, token: str, scope: str = "api") -> Principal | None:
        """The token's principal if the signature, scope and expiry are valid; else None."""
        try:
            encoded, signature = token.split(".")
            if not hmac.compare_digest(signature, self._sign(encoded.encode("ascii"))):
                return None
            payload = json.loads(_b64decode(encoded))
        except (ValueError, TypeError, UnicodeError):
            return None
        if payload.get("scope") != scope or payload.get("role") not in ROLES:
            return None
        if not isinstance(payload.get("exp"), int) or payload["exp"] < self.clock():
            return None
        return Principal(payload["sub"], payload["role"])


# ---- brute-force protection -------------------------------------------------------------------

class LoginThrottle:
    """After `max_failures` failed logins for a username within `window` seconds, refuse
    further attempts for that username until the window has passed."""

    def __init__(self, max_failures: int = 5, window_seconds: int = 900, clock=time.monotonic) -> None:
        self.max_failures = max_failures
        self.window = window_seconds
        self.clock = clock
        self._failures: dict[str, list[float]] = {}
        self._lock = threading.Lock()

    def _recent(self, key: str) -> list[float]:
        now = self.clock()
        recent = [t for t in self._failures.get(key, []) if now - t < self.window]
        self._failures[key] = recent
        return recent

    def retry_after(self, username: str) -> int:
        """Seconds until another attempt is allowed (0 = allowed now)."""
        with self._lock:
            recent = self._recent(username.lower())
            if len(recent) < self.max_failures:
                return 0
            return max(1, int(self.window - (self.clock() - recent[0])))

    def record_failure(self, username: str) -> None:
        with self._lock:
            self._recent(username.lower()).append(self.clock())

    def reset(self, username: str) -> None:
        with self._lock:
            self._failures.pop(username.lower(), None)


def generate_secret_key() -> str:
    return secrets.token_urlsafe(48)


# ---- wiring -------------------------------------------------------------------------------------

class AuthConfigurationError(RuntimeError):
    """Login is enabled but cannot work safely; the backend refuses to start (fail closed)."""


@dataclass
class AuthContext:
    enabled: bool
    users: UserStore | None = None
    tokens: TokenService | None = None
    throttle: LoginThrottle | None = None


def build_auth(settings) -> AuthContext:
    if not settings.auth_enabled:
        return AuthContext(enabled=False)
    key = settings.auth_secret_key.get_secret_value() if settings.auth_secret_key else ""
    if len(key) < MIN_SECRET_KEY_LENGTH:
        raise AuthConfigurationError(
            "AUTH_ENABLED=true but AUTH_SECRET_KEY is missing or shorter than "
            f"{MIN_SECRET_KEY_LENGTH} characters. Generate one with: "
            "python -m backend.auth_admin generate-secret"
        )
    users = UserStore(settings.auth_users_path)
    if "admin" not in users.list_users().values():
        raise AuthConfigurationError(
            f"AUTH_ENABLED=true but {settings.auth_users_path} has no admin account. Create one with: "
            "python -m backend.auth_admin add-user <name> --role admin"
        )
    return AuthContext(True, users, TokenService(key, settings.auth_token_ttl_minutes * 60), LoginThrottle())
