import pytest

from core import config, store


@pytest.fixture(autouse=True)
def _no_stance_model(monkeypatch):
    """Keep tests offline: never download the NLI cross-encoder or call Haiku."""
    monkeypatch.setattr(config, "STANCE_PROVIDER", "off")


@pytest.fixture
def conn(tmp_path):
    c = store.connect(tmp_path / "test.db")
    yield c
    c.close()
