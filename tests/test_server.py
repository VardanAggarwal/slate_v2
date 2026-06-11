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


def test_status_requires_auth_when_configured(client, monkeypatch):
    monkeypatch.setattr(config, "AUTH_USER", "u")
    monkeypatch.setattr(config, "AUTH_PASS", "p")
    assert client.get("/status").status_code == 401
    assert client.get("/status", auth=("u", "wrong")).status_code == 401
    ok = client.get("/status", auth=("u", "p"))
    assert ok.status_code == 200
    assert "Last consolidation" in ok.text
