import pytest
from fastapi.testclient import TestClient

from core import config


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DB_PATH", tmp_path / "srv.db")
    import server
    return TestClient(server.app)


def test_health_public(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_status_open_without_configured_creds(client):
    r = client.get("/status")
    assert r.status_code == 200
    assert "slate-engine" in r.text
    assert "episodes" in r.text


def test_wellknown_candidates_strip_mcp_suffix():
    from server import _wellknown_candidates
    assert _wellknown_candidates("oauth-protected-resource/mcp/") == [
        "oauth-protected-resource/mcp/", "oauth-protected-resource"]
    assert _wellknown_candidates("oauth-authorization-server/mcp") == [
        "oauth-authorization-server/mcp", "oauth-authorization-server"]
    assert _wellknown_candidates("oauth-authorization-server") == [
        "oauth-authorization-server"]


def test_wellknown_forward_returns_cleanly_when_oauth_disabled(client):
    # No AUTH creds in tests → mounted app has no oauth routes → clean 404
    r = client.get("/.well-known/oauth-protected-resource/mcp/")
    assert r.status_code == 404


def test_status_requires_auth_when_configured(client, monkeypatch):
    monkeypatch.setattr(config, "AUTH_USER", "u")
    monkeypatch.setattr(config, "AUTH_PASS", "p")
    assert client.get("/status").status_code == 401
    assert client.get("/status", auth=("u", "wrong")).status_code == 401
    ok = client.get("/status", auth=("u", "p"))
    assert ok.status_code == 200
    assert "Last consolidation" in ok.text
