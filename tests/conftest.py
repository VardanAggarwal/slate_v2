import pytest

from core import config, store

# The corpus owner used across tests (AUTH.md: every call scopes to a user).
UID = "usr_test"
# A second user for isolation tests — must never see UID's data.
UID_B = "usr_other"


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
