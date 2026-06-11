import pytest

from core import config, store


@pytest.fixture(autouse=True)
def _offline(monkeypatch):
    """Keep tests offline + deterministic: local embedder, no NLI/Haiku stance."""
    monkeypatch.setattr(config, "STANCE_PROVIDER", "off")
    monkeypatch.setattr(config, "HF_TOKEN", "")


@pytest.fixture
def conn(tmp_path):
    c = store.connect(tmp_path / "test.db")
    yield c
    c.close()
