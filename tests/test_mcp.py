import pytest
from fastmcp import Client

from core import config
from core.consolidate import consolidate
from core.encode import encode
from tests.test_consolidate import S1, S2, _seed, fake_llm  # noqa: F401

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
async def test_edit_note_supersedes_old_version(mcp_db, client):
    from fastmcp.exceptions import ToolError
    r1 = await _call(client, "save_note", text=S1, title="memory note")
    r2 = await _call(client, "edit_note", episode_id=r1["episode_id"], new_text=S2)
    assert r2["superseded_episode_id"] == r1["episode_id"]
    assert r2["title"] == "memory note"     # kept when not overridden
    assert "narrate" in r2
    notes = await _call(client, "list_recent_notes")
    ids = [n["id"] for n in notes]
    assert r2["episode_id"] in ids and r1["episode_id"] not in ids
    # the old version stays fetchable, flagged with its replacement
    old = await _call(client, "get_note", episode_id=r1["episode_id"])
    assert old["superseded_by"] == r2["episode_id"]
    # editing a superseded version is refused with a pointer to the head
    with pytest.raises(ToolError, match=r2["episode_id"]):
        await _call(client, "edit_note", episode_id=r1["episode_id"], new_text=S1)


@pytest.mark.asyncio
async def test_recall_and_context_tools(mcp_db, client, fake_llm):
    from core import store
    conn = store.connect()
    _seed(conn, UID, S1, title="memory note")
    _seed(conn, UID, S2, title="sleep note")
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


# ── Evidence lane: E6 (save_evidence) + E7 (receipt variant) ─────────────────
SOURCE = ("Across 29 trials, spaced repetition was the most reliable method for "
          "retaining knowledge over multi-year intervals (Cepeda et al., 2006).")


@pytest.mark.asyncio
async def test_save_evidence_lands_as_a_research_episode(mcp_db, client):
    from core import store
    r = await _call(client, "save_evidence", text=SOURCE,
                    source_url="https://example.org/spacing",
                    source_title="Distributed practice meta-analysis",
                    retrieved_at="2026-07-30")
    conn = store.connect()
    ep = store.get_episode(conn, UID, r["episode_id"])
    assert ep["source"] == "research"
    assert store.episode_citation(conn, UID, r["episode_id"])["url"] == \
        "https://example.org/spacing"
    # E6: no synchronous refine (fragmentation is about YOUR surprise, not a source's)
    n_frags = conn.execute("SELECT COUNT(*) AS n FROM fragments").fetchone()["n"]
    assert n_frags == 0
    # and no engagement vote — a source save is not the user's thinking spawning writing
    n_eng = conn.execute(
        "SELECT COUNT(*) AS n FROM events WHERE type = 'ENGAGEMENT'").fetchone()["n"]
    assert n_eng == 0


@pytest.mark.asyncio
async def test_save_evidence_requires_provenance(mcp_db, client):
    from fastmcp.exceptions import ToolError
    with pytest.raises(ToolError, match="provenance"):
        await _call(client, "save_evidence", text=SOURCE, source_url="",
                    source_title="x", retrieved_at="2026-07-30")
    with pytest.raises(ToolError, match="provenance"):
        await _call(client, "save_evidence", text=SOURCE,
                    source_url="https://example.org", source_title="  ",
                    retrieved_at="2026-07-30")


def test_evidence_receipt_never_quotes_a_source_as_the_user():
    """`— your new line: "…"` is a misattribution on the evidence path: it renders a
    source's sentence as something the user wrote. Fix, not restyle."""
    import mcp_server
    receipt = {"source": "research", "n_novelties": 0,
               "contradictions": [{"claim_text": "sleep is optional",
                                   "sentence": "Sleep loss impaired recall by 40%.",
                                   "similarity": 0.81}],
               "echoes": [], "prior_episode_matches": []}
    md = mcp_server.receipt_markdown(receipt)
    assert "your new line" not in md
    assert "A source refutes your claim" in md
    assert "⚡" in md


def test_evidence_receipt_splits_backs_from_relates():
    import mcp_server
    receipt = {"source": "research", "n_novelties": 1, "contradictions": [],
               "echoes": [{"claim_text": "spacing works", "sentence": "…",
                           "similarity": 0.9, "stance": "entailment"},
                          {"claim_text": "sleep matters", "sentence": "…",
                           "similarity": 0.75, "stance": "neutral"}],
               "prior_episode_matches": []}
    md = mcp_server.receipt_markdown(receipt)
    assert "📎 **Backs your claim**" in md
    assert "**Relates to**" in md and "unverified" in md
    assert "🗄️" in md and "parked for the nightly sweep" in md


def test_empty_evidence_receipt_has_its_own_wording():
    import mcp_server
    md = mcp_server.receipt_markdown({"source": "research", "n_novelties": 0,
                                      "echoes": [], "contradictions": [],
                                      "prior_episode_matches": []})
    assert md == "Filed. Nothing in your corpus touches this yet."


def test_weak_match_is_reported_not_swallowed_note_path():
    """[NOVELTY_THRESHOLD, ECHO_THRESHOLD) used to hit neither receipt branch, so a
    near-miss produced NO line at all and a note that did touch stored thinking
    read as touching nothing. It must hedge, and must not claim agreement."""
    import mcp_server
    receipt = {"source": "mcp", "n_novelties": 0, "echoes": [], "contradictions": [],
               "weak_matches": [{"claim_text": "RAG degrades at scale",
                                 "sentence": "…", "similarity": 0.57}],
               "prior_episode_matches": []}
    md = mcp_server.receipt_markdown(receipt)
    assert "**Relates to**" in md and "unverified" in md
    assert "0.57" in md
    assert "🔁" not in md and "⚡" not in md          # no stance was computed
    assert md != "Saved. No overlaps with stored thinking detected."


def test_weak_match_is_reported_not_swallowed_evidence_path():
    """The case that actually bit: every sentence of an evidence save landed in the
    band, so the receipt said only 'backs nothing you've written yet' — false."""
    import mcp_server
    receipt = {"source": "research", "n_novelties": 0, "echoes": [],
               "contradictions": [],
               "weak_matches": [{"claim_text": "spacing works", "sentence": "…",
                                 "similarity": 0.58}],
               "prior_episode_matches": []}
    md = mcp_server.receipt_markdown(receipt)
    assert "**Relates to**" in md and "unverified" in md
    assert "📎" not in md                              # backing was never verified
    assert "backs nothing you've written yet" not in md
    assert md != "Filed. Nothing in your corpus touches this yet."


def test_build_receipt_buckets_the_dead_zone(monkeypatch):
    """Bucketing itself: sim in the band goes to weak_matches, NOT novelties —
    calling it novel would assert 'nothing like this is stored', which is false."""
    from core import encode as enc

    monkeypatch.setattr(config, "ECHO_THRESHOLD", 0.60)
    monkeypatch.setattr(config, "NOVELTY_THRESHOLD", 0.55)
    monkeypatch.setattr(enc.store, "knn_claims", lambda *a, **k: [
        {"claim_id": "c1", "text": "a stored claim", "similarity": 0.57}])
    monkeypatch.setattr(enc.store, "knn_sentences", lambda *a, **k: [])

    r = enc._build_receipt(None, "u1", ["a sentence long enough to survive."],
                           [[0.0] * 384])
    assert r["weak_matches"] and r["weak_matches"][0]["similarity"] == 0.57
    assert r["novelties"] == [] and r["n_novelties"] == 0
    assert r["echoes"] == [] and r["contradictions"] == []


def test_a_note_receipt_attributes_a_research_match_to_the_source():
    """E7 is NOT confined to the evidence receipt: knn_sentences spans all episodes,
    so a NOTE receipt must not render a paper as 'your note'."""
    import mcp_server
    receipt = {"source": "mcp", "n_novelties": 0, "echoes": [], "contradictions": [],
               "prior_episode_matches": [
                   {"episode_title": "Cepeda 2006", "episode_ts": "2026-07-30",
                    "episode_source": "research", "matched_sentence": "…",
                    "similarity": 0.8},
                   {"episode_title": "my old note", "episode_ts": "2026-01-01",
                    "episode_source": "mcp", "matched_sentence": "…",
                    "similarity": 0.75}]}
    md = mcp_server.receipt_markdown(receipt)
    assert "📚 **Resonates with** another source “Cepeda 2006”" in md
    assert "🕰️ **Resonates with** your note “my old note”" in md


@pytest.mark.asyncio
async def test_digest_tool(mcp_db, client, fake_llm):
    from core import store
    conn = store.connect()
    _seed(conn, UID, S1, title="memory note")
    consolidate(conn, UID)
    md = await _call(client, "digest")
    assert "🌱" in md  # new concept appears in the digest
