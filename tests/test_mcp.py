import pytest
from fastmcp import Client

from core import config
from core.consolidate import consolidate
from core.encode import encode
from tests.test_consolidate import S1, S2, fake_llm  # noqa: F401

# No OAuth in tests (AUTH_USER unset) → every tool call resolves to the
# local-dev corpus owner.
UID = config.DEFAULT_USER_ID


@pytest.fixture
def mcp_db(tmp_path, monkeypatch):
    """Point the MCP server's per-call connections at a fresh DB."""
    db = tmp_path / "mcp.db"
    monkeypatch.setattr(config, "DB_PATH", db)
    return db


@pytest.fixture
def client():
    import mcp_server
    return Client(mcp_server.mcp)


async def _call(client, tool, **args):
    async with client:
        result = await client.call_tool(tool, args)
    return result.data


@pytest.mark.asyncio
async def test_save_note_returns_narratable_receipt(mcp_db, client):
    r1 = await _call(client, "save_note", text=S1, title="memory note")
    assert r1["episode_id"].startswith("ep_")
    assert "✨" in r1["narrate"]  # all novelty on an empty store

    r2 = await _call(client, "save_note", text=S1, title="memory again")
    assert "Resonates with" in r2["narrate"]  # prior-episode echo
    assert "memory note" in r2["narrate"]     # provenance in the narration


@pytest.mark.asyncio
async def test_save_note_rejects_empty(mcp_db, client):
    from fastmcp.exceptions import ToolError
    with pytest.raises(ToolError):
        await _call(client, "save_note", text="   ", title="x")


@pytest.mark.asyncio
async def test_recall_and_context_tools(mcp_db, client, fake_llm):
    from core import store
    conn = store.connect()
    encode(conn, UID, S1, source="test", title="memory note")
    encode(conn, UID, S2, source="test", title="sleep note")
    consolidate(conn, UID)

    hits = await _call(client, "recall", query="retaining knowledge over time")
    assert hits and hits[0]["type"] in ("claim", "concept")

    md = await _call(client, "assemble_context", topic="memory")
    assert md.startswith("## Slate context")

    notes = await _call(client, "list_recent_notes")
    assert len(notes) == 2
    full = await _call(client, "get_note", episode_id=notes[0]["id"])
    assert full["blueprint"] is not None

    concepts = store.all_concepts(conn, UID)
    cfull = await _call(client, "get_concept", concept_id=concepts[0]["id"])
    assert cfull["members"]

    tl = await _call(client, "timeline", concept_id=concepts[0]["id"])
    assert any(e["type"] == "CONCEPT_CREATED" for e in tl)

    s = await _call(client, "stats")
    assert s["episodes"] == 2 and s["concepts"] >= 1


@pytest.mark.asyncio
async def test_digest_tool(mcp_db, client, fake_llm):
    from core import store
    conn = store.connect()
    encode(conn, UID, S1, source="test", title="memory note")
    consolidate(conn, UID)
    md = await _call(client, "digest")
    assert "🌱" in md  # new concept appears in the digest
