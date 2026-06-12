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


def test_status_has_run_cta(client):
    r = client.get("/status")
    assert 'action="run"' in r.text
    assert "Run nightly jobs now" in r.text


def test_run_requires_auth_when_configured(client, monkeypatch):
    monkeypatch.setattr(config, "AUTH_USER", "u")
    monkeypatch.setattr(config, "AUTH_PASS", "p")
    assert client.post("/run").status_code == 401


def test_run_triggers_and_redirects(client, monkeypatch):
    import server
    started = {}

    def fake_run():  # don't run the real (LLM-calling) job in tests
        started["yes"] = True

    monkeypatch.setattr(server, "_run_nightly", fake_run)
    r = client.post("/run", follow_redirects=False)
    assert r.status_code == 303
    assert r.headers["location"] == "status"
    # background thread should have invoked the runner
    import time as _t
    for _ in range(50):
        if started:
            break
        _t.sleep(0.01)
    assert started.get("yes")
