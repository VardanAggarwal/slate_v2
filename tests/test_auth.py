"""AUTH.md §2: users-table auth, bootstrap admin, and OAuth token→user binding."""
import pytest

from core import config, store
from core.auth import authenticate, check_password, hash_password


def test_hash_and_check_password():
    h = hash_password("hunter2")
    assert h != "hunter2"
    assert check_password("hunter2", h)
    assert not check_password("hunter3", h)
    assert not check_password("hunter2", "not-a-bcrypt-hash")  # fail closed


def test_authenticate_against_users_table(conn):
    with conn:
        store.create_user(conn, "alice", hash_password("pw-a"))
    assert authenticate(conn, "alice", "pw-a")["username"] == "alice"
    assert authenticate(conn, "alice", "wrong") is None
    assert authenticate(conn, "nobody", "pw-a") is None


def test_env_bootstrap_provisions_first_admin(conn, monkeypatch):
    monkeypatch.setattr(config, "AUTH_USER", "boot")
    monkeypatch.setattr(config, "AUTH_PASS", "strap")
    user = authenticate(conn, "boot", "strap")
    assert user is not None and user["is_admin"] == 1
    assert store.count_users(conn) == 1
    # the row now owns auth: env creds no longer matter
    monkeypatch.setattr(config, "AUTH_PASS", "changed")
    assert authenticate(conn, "boot", "strap") is not None


def test_env_bootstrap_dead_once_a_user_exists(conn, monkeypatch):
    with conn:
        store.create_user(conn, "alice", hash_password("pw-a"))
    monkeypatch.setattr(config, "AUTH_USER", "boot")
    monkeypatch.setattr(config, "AUTH_PASS", "strap")
    assert authenticate(conn, "boot", "strap") is None


# ── OAuth provider binding (the Phase-1 spike, as a regression test) ──────────
def _make_client(client_id="c1"):
    from mcp.shared.auth import OAuthClientInformationFull
    from pydantic import AnyUrl
    return OAuthClientInformationFull(
        client_id=client_id, client_secret="s",
        redirect_uris=[AnyUrl("http://localhost/cb")], scope="")


async def _dance(provider, client, user_id="usr_alice"):
    """register → authorize → login approval → code exchange. Returns OAuthToken."""
    from mcp.server.auth.provider import AuthorizationParams
    from pydantic import AnyUrl
    await provider.register_client(client)
    params = AuthorizationParams(
        redirect_uri=AnyUrl("http://localhost/cb"),
        redirect_uri_provided_explicitly=True,
        state="st", code_challenge="c" * 43, scopes=[])
    login_url = await provider.authorize(client, params)
    auth_id = login_url.split("auth_id=")[1]
    redirect = await provider.approve_authorization(auth_id, user_id)
    code = redirect.split("code=")[1].split("&")[0]
    code_obj = await provider.load_authorization_code(client, code)
    return await provider.exchange_authorization_code(client, code_obj)


@pytest.mark.asyncio
async def test_oauth_token_carries_user_id(tmp_path, monkeypatch):
    """authorize → login approval → code exchange must stamp user_id into the
    access token's claims, and the refresh chain must preserve it."""
    monkeypatch.setattr(config, "AUTH_USER", "u")
    monkeypatch.setattr(config, "AUTH_PASS", "p")
    monkeypatch.setattr(config, "DB_PATH", tmp_path / "auth.db")
    import importlib
    import mcp_server
    importlib.reload(mcp_server)
    try:
        provider = mcp_server.slate_auth
        assert provider is not None
        client = _make_client()
        token = await _dance(provider, client)
        at = provider.access_tokens[token.access_token]
        assert at.claims == {"user_id": "usr_alice"}

        rt = await provider.load_refresh_token(client, token.refresh_token)
        token2 = await provider.exchange_refresh_token(client, rt, [])
        at2 = provider.access_tokens[token2.access_token]
        assert at2.claims == {"user_id": "usr_alice"}
    finally:
        monkeypatch.undo()
        importlib.reload(mcp_server)


@pytest.mark.asyncio
async def test_oauth_state_survives_restart(tmp_path, monkeypatch):
    """AUTH.md §2: tokens, bindings, and DCR clients are persisted — a fresh
    provider over the same DB (= a server restart) must honor them."""
    monkeypatch.setattr(config, "AUTH_USER", "u")
    monkeypatch.setattr(config, "AUTH_PASS", "p")
    monkeypatch.setattr(config, "DB_PATH", tmp_path / "auth.db")
    import importlib
    import mcp_server
    importlib.reload(mcp_server)
    try:
        base = "http://127.0.0.1/mcp"
        client = _make_client()
        token = await _dance(mcp_server.slate_auth, client)

        # restart #1: bearer token still verifies, with its binding intact
        p2 = mcp_server.SlateOAuthProvider(base_url=base)
        at = await p2.load_access_token(token.access_token)
        assert at is not None and at.claims == {"user_id": "usr_alice"}
        # DCR client survived (token/refresh exchanges need it)
        assert (await p2.get_client("c1")).client_secret == "s"

        # refresh chain works on the new instance and keeps the binding
        rt = await p2.load_refresh_token(client, token.refresh_token)
        assert rt is not None
        token2 = await p2.exchange_refresh_token(client, rt, [])
        assert p2.access_tokens[token2.access_token].claims == {"user_id": "usr_alice"}

        # restart #2: rotation persisted — old pair dead, new pair alive+bound
        p3 = mcp_server.SlateOAuthProvider(base_url=base)
        assert await p3.load_access_token(token.access_token) is None
        assert await p3.load_refresh_token(client, token.refresh_token) is None
        at3 = await p3.load_access_token(token2.access_token)
        assert at3 is not None and at3.claims == {"user_id": "usr_alice"}
        rt3 = await p3.load_refresh_token(client, token2.refresh_token)
        token3 = await p3.exchange_refresh_token(client, rt3, [])
        assert p3.access_tokens[token3.access_token].claims == {"user_id": "usr_alice"}

        # explicit revocation persists too
        await p3.revoke_token(p3.access_tokens[token3.access_token])
        p4 = mcp_server.SlateOAuthProvider(base_url=base)
        assert await p4.load_access_token(token3.access_token) is None
        assert await p4.load_refresh_token(client, token3.refresh_token) is None
    finally:
        monkeypatch.undo()
        importlib.reload(mcp_server)


@pytest.mark.asyncio
async def test_expired_access_token_purged_on_boot_refresh_survives(tmp_path, monkeypatch):
    """An expired access token must not be reloaded, but its non-expiring
    refresh token must still mint a new bound pair — that's what lets a client
    reconnect after downtime longer than the 1h access-token lifetime."""
    monkeypatch.setattr(config, "AUTH_USER", "u")
    monkeypatch.setattr(config, "AUTH_PASS", "p")
    monkeypatch.setattr(config, "DB_PATH", tmp_path / "auth.db")
    import importlib
    import mcp_server
    importlib.reload(mcp_server)
    try:
        client = _make_client()
        token = await _dance(mcp_server.slate_auth, client)

        # age the access token past expiry, directly in the DB
        conn = store.connect()
        with conn:
            conn.execute("UPDATE oauth_access_tokens SET expires_at = 1 WHERE token = ?",
                         (token.access_token,))
        conn.close()

        p2 = mcp_server.SlateOAuthProvider(base_url="http://127.0.0.1/mcp")
        assert await p2.load_access_token(token.access_token) is None
        rt = await p2.load_refresh_token(client, token.refresh_token)
        assert rt is not None, "refresh token must survive access-token expiry"
        token2 = await p2.exchange_refresh_token(client, rt, [])
        assert p2.access_tokens[token2.access_token].claims == {"user_id": "usr_alice"}
    finally:
        monkeypatch.undo()
        importlib.reload(mcp_server)


def test_unbound_token_is_rejected_by_tools(monkeypatch, tmp_path):
    """A token with no user binding must hard-error, never fall back to a
    default corpus (AUTH.md §2)."""
    monkeypatch.setattr(config, "AUTH_USER", "u")
    monkeypatch.setattr(config, "AUTH_PASS", "p")
    monkeypatch.setattr(config, "DB_PATH", tmp_path / "auth.db")
    import importlib
    import mcp_server
    importlib.reload(mcp_server)
    try:
        from fastmcp.exceptions import ToolError
        from mcp.server.auth.provider import AccessToken
        monkeypatch.setattr(
            "fastmcp.server.dependencies.get_access_token",
            lambda: AccessToken(token="t", client_id="c", scopes=[], expires_at=None))
        with pytest.raises(ToolError, match="no user is bound"):
            mcp_server._user_id()
    finally:
        monkeypatch.undo()
        importlib.reload(mcp_server)
