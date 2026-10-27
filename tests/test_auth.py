"""Login: hashing, tokens, throttling, startup checks, role enforcement and the admin CLI."""

import pytest
from fastapi.testclient import TestClient

from backend.auth import (
    AuthConfigurationError,
    LoginThrottle,
    Principal,
    TokenService,
    UserStore,
    hash_password,
    verify_password,
)
from backend.auth_admin import main as auth_admin
from backend.config import get_settings
from tests.test_verification_flow import ML_PDF

SECRET = "x" * 48
ADMIN_PW = "admin-password-123"
VIEWER_PW = "viewer-password-456"


# ---- building blocks ------------------------------------------------------------------------

def test_password_hashing():
    stored = hash_password("correct horse battery")

    assert stored.startswith("scrypt$") and "correct horse" not in stored
    assert verify_password("correct horse battery", stored)
    assert not verify_password("wrong password", stored)
    assert not verify_password("anything", "not-a-hash")
    assert hash_password("same") != hash_password("same")  # salted


def test_tokens_are_signed_scoped_and_expire():
    clock = {"t": 1000.0}
    tokens = TokenService(SECRET, ttl_seconds=60, clock=lambda: clock["t"])
    token, _ = tokens.issue(Principal("alice", "admin"), "api")

    assert tokens.verify(token, "api") == Principal("alice", "admin")
    assert tokens.verify(token, "file") is None  # wrong scope
    payload, signature = token.split(".")
    assert tokens.verify(payload + "." + signature[:-2] + "AA", "api") is None  # tampered signature
    forged = TokenService("y" * 48, 60, clock=lambda: clock["t"]).issue(Principal("alice", "admin"))[0]
    assert tokens.verify(forged, "api") is None  # signed with another key
    clock["t"] += 61
    assert tokens.verify(token, "api") is None  # expired
    assert tokens.verify("garbage", "api") is None


def test_short_secret_keys_are_rejected():
    with pytest.raises(ValueError):
        TokenService("short", 60)


def test_login_throttle_locks_after_repeated_failures():
    clock = {"t": 0.0}
    throttle = LoginThrottle(max_failures=3, window_seconds=600, clock=lambda: clock["t"])
    for _ in range(3):
        assert throttle.retry_after("Alice") == 0
        throttle.record_failure("alice")

    assert throttle.retry_after("ALICE") > 0  # case-insensitive
    clock["t"] = 601
    assert throttle.retry_after("alice") == 0


def test_user_store_validates_and_removes(tmp_path):
    store = UserStore(tmp_path / "users.json")
    store.add_user("alice", "long-enough-pw", "admin")

    assert store.authenticate("alice", "long-enough-pw") == Principal("alice", "admin")
    assert store.authenticate("alice", "wrong-password") is None
    assert store.authenticate("nobody", "long-enough-pw") is None
    with pytest.raises(ValueError):
        store.add_user("bob", "short", "viewer")
    with pytest.raises(ValueError):
        store.add_user("Bad Name!", "long-enough-pw", "viewer")
    with pytest.raises(ValueError):
        store.add_user("carol", "long-enough-pw", "superuser")
    assert store.remove_user("alice") and store.list_users() == {}


# ---- the app with login enabled ---------------------------------------------------------------

def enable_auth(monkeypatch, secret=SECRET, with_users=True):
    monkeypatch.setenv("AUTH_ENABLED", "true")
    if secret is not None:
        monkeypatch.setenv("AUTH_SECRET_KEY", secret)
    get_settings.cache_clear()
    if with_users:
        store = UserStore(get_settings().auth_users_path)
        store.add_user("admin", ADMIN_PW, "admin")
        store.add_user("viewer", VIEWER_PW, "viewer")


@pytest.fixture
def auth_client(app_factory, monkeypatch):
    enable_auth(monkeypatch)
    with TestClient(app_factory()) as client:
        yield client


def login(client, username, password):
    return client.post("/api/auth/login", json={"username": username, "password": password})


def headers(client, username, password):
    return {"Authorization": f"Bearer {login(client, username, password).json()['access_token']}"}


def test_startup_refuses_without_secret_key(app_factory, monkeypatch):
    enable_auth(monkeypatch, secret=None, with_users=False)
    with pytest.raises(AuthConfigurationError, match="AUTH_SECRET_KEY"):
        with TestClient(app_factory()):
            pass


def test_startup_refuses_without_an_admin(app_factory, monkeypatch):
    enable_auth(monkeypatch, with_users=False)
    UserStore(get_settings().auth_users_path).add_user("viewer", VIEWER_PW, "viewer")
    with pytest.raises(AuthConfigurationError, match="no admin"):
        with TestClient(app_factory()):
            pass


def test_login_success_and_failure(auth_client):
    ok = login(auth_client, "Admin", ADMIN_PW)  # usernames are case-insensitive
    assert ok.status_code == 200
    body = ok.json()
    assert (body["username"], body["role"], body["token_type"]) == ("admin", "admin", "bearer")
    assert body["access_token"] != body["file_token"]

    wrong = login(auth_client, "admin", "nope-nope-nope")
    unknown = login(auth_client, "ghost", "nope-nope-nope")
    assert wrong.status_code == unknown.status_code == 401
    assert wrong.json()["detail"]["message"] == unknown.json()["detail"]["message"]  # no user enumeration


def test_repeated_failures_lock_the_account(auth_client):
    for _ in range(5):
        assert login(auth_client, "viewer", "wrong-password").status_code == 401
    locked = login(auth_client, "viewer", VIEWER_PW)  # even the right password is refused for now

    assert locked.status_code == 429 and int(locked.headers["Retry-After"]) > 0


def test_health_is_public_and_reports_login(auth_client):
    response = auth_client.get("/health")

    assert response.status_code == 200 and response.json()["auth_enabled"] is True


@pytest.mark.parametrize(("method", "path"), [
    ("GET", "/api/documents"), ("GET", "/api/documents/doc_x/chunks"),
    ("POST", "/api/retrieval/search"), ("POST", "/api/chat/ask"),
    ("GET", "/api/evaluation/runs"), ("GET", "/api/evaluation/dataset"), ("GET", "/api/auth/me"),
])
def test_api_requires_login(auth_client, method, path):
    response = auth_client.request(method, path, json={"query": "x", "question": "x"})

    assert response.status_code == 401
    assert response.json()["detail"]["code"] == "not_authenticated"


def test_viewer_can_read_but_not_change(auth_client):
    admin = headers(auth_client, "admin", ADMIN_PW)
    viewer = headers(auth_client, "viewer", VIEWER_PW)
    doc = auth_client.post("/api/documents/upload", headers=admin,
                           files={"file": ("ml.pdf", ML_PDF, "application/pdf")})
    assert doc.status_code == 201
    doc_id = doc.json()["document_id"]

    # Reads work for the viewer.
    assert auth_client.get("/api/documents", headers=viewer).status_code == 200
    assert auth_client.get(f"/api/documents/{doc_id}/chunks", headers=viewer).status_code == 200
    assert auth_client.post("/api/retrieval/search", headers=viewer, json={"query": "L2"}).status_code == 200
    assert auth_client.get("/api/evaluation/runs", headers=viewer).status_code == 200
    assert auth_client.get("/api/auth/me", headers=viewer).json()["role"] == "viewer"

    # Changes are refused by the backend, not just hidden in the UI.
    forbidden = [
        auth_client.post("/api/documents/upload", headers=viewer, files={"file": ("x.pdf", ML_PDF, "application/pdf")}),
        auth_client.delete(f"/api/documents/{doc_id}", headers=viewer),
        auth_client.post(f"/api/documents/{doc_id}/reprocess", headers=viewer),
        auth_client.post("/api/evaluation/run", headers=viewer, json={"kind": "retrieval"}),
        auth_client.put("/api/evaluation/runs/answers_20261027T000000Z/reviews", headers=viewer,
                        json={"key": "q1/a1/claim_001", "human_verdict": "supported"}),
    ]
    assert [r.status_code for r in forbidden] == [403] * 5
    assert all(r.json()["detail"]["code"] == "admin_required" for r in forbidden)
    assert auth_client.get(f"/api/documents/{doc_id}", headers=viewer).status_code == 200  # still there


def test_pdf_links_use_a_file_only_token(auth_client):
    admin_login = login(auth_client, "admin", ADMIN_PW).json()
    admin = {"Authorization": f"Bearer {admin_login['access_token']}"}
    doc_id = auth_client.post("/api/documents/upload", headers=admin,
                              files={"file": ("ml.pdf", ML_PDF, "application/pdf")}).json()["document_id"]
    viewer = login(auth_client, "viewer", VIEWER_PW).json()
    file_url = f"/api/documents/{doc_id}/file"

    assert auth_client.get(f"{file_url}?token={viewer['file_token']}").status_code == 200
    assert auth_client.get(file_url, headers=admin).status_code == 200
    assert auth_client.get(file_url).status_code == 401
    # An API token in the URL is not accepted (wrong scope)...
    assert auth_client.get(f"{file_url}?token={viewer['access_token']}").status_code == 401
    # ...and a file token cannot call any other endpoint.
    assert auth_client.get("/api/documents", headers={"Authorization": f"Bearer {viewer['file_token']}"}).status_code == 401


def test_removed_user_is_logged_out(auth_client):
    viewer = headers(auth_client, "viewer", VIEWER_PW)
    assert auth_client.get("/api/documents", headers=viewer).status_code == 200

    UserStore(get_settings().auth_users_path).remove_user("viewer")

    assert auth_client.get("/api/documents", headers=viewer).status_code == 401


def test_role_change_invalidates_old_tokens(auth_client):
    viewer = headers(auth_client, "viewer", VIEWER_PW)
    UserStore(get_settings().auth_users_path).add_user("viewer", VIEWER_PW, "admin")

    assert auth_client.get("/api/documents", headers=viewer).status_code == 401  # old role in token


def test_login_endpoint_reports_disabled_login(client):
    assert login(client, "admin", ADMIN_PW).json()["detail"]["code"] == "auth_disabled"


def test_with_login_disabled_everything_acts_as_local_admin(client):
    assert client.get("/api/auth/me").json() == {"username": "local", "role": "admin", "auth_enabled": False}


# ---- admin CLI ----------------------------------------------------------------------------------

def test_cli_manages_users(app_factory, capsys):
    answers = iter(["a-good-password", "a-good-password"])
    assert auth_admin(["add-user", "Alice", "--role", "admin"], prompt=lambda _: next(answers)) == 0
    assert auth_admin(["list-users"]) == 0
    assert "alice" in capsys.readouterr().out
    assert UserStore(get_settings().auth_users_path).authenticate("alice", "a-good-password")

    mismatched = iter(["a-good-password", "another-password"])
    assert auth_admin(["add-user", "bob", "--role", "viewer"], prompt=lambda _: next(mismatched)) == 1
    short = iter(["short", "short"])
    assert auth_admin(["add-user", "bob", "--role", "viewer"], prompt=lambda _: next(short)) == 1
    assert "bob" not in UserStore(get_settings().auth_users_path).list_users()

    assert auth_admin(["remove-user", "alice"]) == 0
    assert auth_admin(["remove-user", "alice"]) == 1


def test_cli_generates_a_long_secret(capsys):
    assert auth_admin(["generate-secret"]) == 0
    line = next(l for l in capsys.readouterr().out.splitlines() if l.startswith("AUTH_SECRET_KEY="))
    assert len(line.split("=", 1)[1]) >= 48
