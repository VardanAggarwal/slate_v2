"""SQLite + sqlite-vec store. Schema init, event-log append/apply. The ONLY module that touches the DB file. See PLAN.md §4.

Invariants (PLAN.md §10):
- Episodes are immutable: there is no UPDATE path for episodes here, and there must never be one.
- The semantic store (claims/concepts/relations) is only written by consolidate.py, only via events.
- Embeddings are stored normalized; vec distance is L2, so cosine = 1 - d²/2.
"""
import json
import os
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path

import sqlite_vec
from sqlite_vec import serialize_float32

from core import config

# ── IDs ───────────────────────────────────────────────────────────────────────
_B32 = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"  # Crockford


def ulid(ts: float | None = None) -> str:
    """ULID: 48-bit ms timestamp + 80 random bits, Crockford base32.

    Pass ts (unix seconds) for replayed episodes so ids sort chronologically.
    """
    t = int((time.time() if ts is None else ts) * 1000)
    chars = []
    for _ in range(10):
        chars.append(_B32[t & 31])
        t >>= 5
    head = "".join(reversed(chars))
    n = int.from_bytes(os.urandom(10), "big")
    chars = []
    for _ in range(16):
        chars.append(_B32[n & 31])
        n >>= 5
    return head + "".join(reversed(chars))


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# ── Schema ────────────────────────────────────────────────────────────────────
_SCHEMA = """
-- EPISODIC STORE (append-only, immutable)
CREATE TABLE IF NOT EXISTS episodes (
    id           TEXT PRIMARY KEY,          -- ep_<ulid>
    ts           TEXT NOT NULL,             -- ISO; replayed notes keep original created_at
    raw_text     TEXT NOT NULL,
    title        TEXT,
    source       TEXT,                      -- 'mcp' | 'replay' | 'import' | ...
    receipt_json TEXT
);

CREATE TABLE IF NOT EXISTS episode_sentences (
    episode_id   TEXT NOT NULL REFERENCES episodes(id),
    idx          INTEGER NOT NULL,
    text         TEXT NOT NULL,
    PRIMARY KEY (episode_id, idx)
);

-- SEMANTIC STORE (derived; rebuildable from episodes + events)
CREATE TABLE IF NOT EXISTS claims (
    id         TEXT PRIMARY KEY,            -- clm_<md5 of canonical text>
    text       TEXT NOT NULL,
    strength   REAL DEFAULT 1.0,
    created_at TEXT,
    last_seen  TEXT
);

CREATE TABLE IF NOT EXISTS claim_support (
    claim_id          TEXT NOT NULL,
    episode_id        TEXT NOT NULL,
    verbatim_sentence TEXT,
    PRIMARY KEY (claim_id, episode_id)
);

CREATE TABLE IF NOT EXISTS concepts (
    id            TEXT PRIMARY KEY,
    label         TEXT,
    canonical     TEXT,
    state         TEXT DEFAULT 'active',    -- 'grounded'|'active'|'stale'|'dormant'
    strength      REAL DEFAULT 1.0,
    created_at    TEXT,
    last_activity TEXT
);

CREATE TABLE IF NOT EXISTS concept_members (
    concept_id TEXT NOT NULL,
    claim_id   TEXT NOT NULL,
    weight     REAL DEFAULT 1.0,
    PRIMARY KEY (concept_id, claim_id)
);

CREATE TABLE IF NOT EXISTS relations (
    from_id             TEXT NOT NULL,      -- claim or concept id
    to_id               TEXT NOT NULL,
    relation            TEXT NOT NULL,      -- 'leads_to'|'contradicts'|'supports'|'bridges'|...
    weight              REAL DEFAULT 1.0,
    created_at          TEXT,
    evidence_episode_id TEXT,
    PRIMARY KEY (from_id, to_id, relation)
);

-- EVENT LOG (backbone; semantic store is a materialized view of this)
CREATE TABLE IF NOT EXISTS events (
    seq          INTEGER PRIMARY KEY AUTOINCREMENT,
    ts           TEXT NOT NULL,
    run_id       TEXT,
    type         TEXT NOT NULL,             -- ENCODED|CANONICALIZED|CONCEPT_CREATED|MERGED|
                                            -- SPLIT|BRIDGED|RELATED|DECAYED|STRENGTHENED
    payload_json TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS consolidation_runs (
    id            TEXT PRIMARY KEY,
    started_at    TEXT,
    finished_at   TEXT,
    n_episodes    INTEGER,
    batch_id      TEXT,
    status        TEXT,
    cost_estimate REAL
);

-- Bookkeeping (NOT part of the immutable episode row)
CREATE TABLE IF NOT EXISTS episode_consolidations (
    episode_id      TEXT PRIMARY KEY,
    run_id          TEXT NOT NULL,
    consolidated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS replay_map (
    old_source_id TEXT PRIMARY KEY,
    episode_id    TEXT NOT NULL,
    replayed_at   TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_events_type ON events(type);
CREATE INDEX IF NOT EXISTS idx_episodes_ts ON episodes(ts);
"""

_VEC_SCHEMA = [
    # sent_key = "<episode_id>:<idx>"
    f"CREATE VIRTUAL TABLE IF NOT EXISTS vec_sentences USING vec0(sent_key TEXT PRIMARY KEY, embedding FLOAT[{config.EMBED_DIM}])",
    f"CREATE VIRTUAL TABLE IF NOT EXISTS vec_claims USING vec0(claim_id TEXT PRIMARY KEY, embedding FLOAT[{config.EMBED_DIM}])",
    f"CREATE VIRTUAL TABLE IF NOT EXISTS vec_concepts USING vec0(concept_id TEXT PRIMARY KEY, embedding FLOAT[{config.EMBED_DIM}])",
]

_FTS_SCHEMA = "CREATE VIRTUAL TABLE IF NOT EXISTS episodes_fts USING fts5(episode_id UNINDEXED, title, raw_text)"


def connect(db_path: str | Path | None = None) -> sqlite3.Connection:
    """Open (and initialise) the engine database."""
    path = Path(db_path or config.DB_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), timeout=30)
    conn.row_factory = sqlite3.Row
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.executescript(_SCHEMA)
    for stmt in _VEC_SCHEMA:
        conn.execute(stmt)
    conn.execute(_FTS_SCHEMA)
    conn.commit()
    return conn


# ── Event log ─────────────────────────────────────────────────────────────────
def append_event(conn: sqlite3.Connection, type_: str, payload: dict,
                 run_id: str | None = None) -> int:
    cur = conn.execute(
        "INSERT INTO events (ts, run_id, type, payload_json) VALUES (?, ?, ?, ?)",
        (now_iso(), run_id, type_, json.dumps(payload, ensure_ascii=False)),
    )
    return cur.lastrowid


def events_since(conn: sqlite3.Connection, seq: int = 0,
                 types: list[str] | None = None) -> list[sqlite3.Row]:
    if types:
        qmarks = ",".join("?" * len(types))
        return conn.execute(
            f"SELECT * FROM events WHERE seq > ? AND type IN ({qmarks}) ORDER BY seq",
            (seq, *types)).fetchall()
    return conn.execute("SELECT * FROM events WHERE seq > ? ORDER BY seq", (seq,)).fetchall()


# ── Episodic writes (called by encode.py inside one transaction) ──────────────
def new_episode_id(ts_unix: float | None = None) -> str:
    return "ep_" + ulid(ts_unix)


def insert_episode(conn: sqlite3.Connection, episode_id: str, ts: str, raw_text: str,
                   title: str | None, source: str, receipt: dict,
                   sentences: list[str], embeddings) -> None:
    """Write one episode + its sentences + vectors + FTS row. Caller owns the transaction."""
    conn.execute(
        "INSERT INTO episodes (id, ts, raw_text, title, source, receipt_json) VALUES (?, ?, ?, ?, ?, ?)",
        (episode_id, ts, raw_text, title, source, json.dumps(receipt, ensure_ascii=False)),
    )
    for idx, sent in enumerate(sentences):
        conn.execute(
            "INSERT INTO episode_sentences (episode_id, idx, text) VALUES (?, ?, ?)",
            (episode_id, idx, sent),
        )
        conn.execute(
            "INSERT INTO vec_sentences (sent_key, embedding) VALUES (?, ?)",
            (f"{episode_id}:{idx}", serialize_float32([float(x) for x in embeddings[idx]])),
        )
    conn.execute(
        "INSERT INTO episodes_fts (episode_id, title, raw_text) VALUES (?, ?, ?)",
        (episode_id, title or "", raw_text),
    )


# ── kNN lookups ───────────────────────────────────────────────────────────────
def _sim(l2_distance: float) -> float:
    """Normalized vectors: cosine similarity = 1 - (L2 distance)² / 2."""
    return 1.0 - (l2_distance * l2_distance) / 2.0


def knn_claims(conn: sqlite3.Connection, embedding, k: int = 5) -> list[dict]:
    """Nearest canonical claims to one sentence embedding."""
    rows = conn.execute(
        "SELECT claim_id, distance FROM vec_claims WHERE embedding MATCH ? AND k = ?",
        (serialize_float32([float(x) for x in embedding]), k),
    ).fetchall()
    out = []
    for r in rows:
        claim = conn.execute("SELECT text, strength FROM claims WHERE id = ?",
                             (r["claim_id"],)).fetchone()
        out.append({
            "claim_id": r["claim_id"],
            "text": claim["text"] if claim else None,
            "strength": claim["strength"] if claim else None,
            "similarity": _sim(r["distance"]),
        })
    return out


def knn_sentences(conn: sqlite3.Connection, embedding, k: int = 5) -> list[dict]:
    """Nearest prior episode sentences to one sentence embedding."""
    rows = conn.execute(
        "SELECT sent_key, distance FROM vec_sentences WHERE embedding MATCH ? AND k = ?",
        (serialize_float32([float(x) for x in embedding]), k),
    ).fetchall()
    out = []
    for r in rows:
        episode_id, idx = r["sent_key"].rsplit(":", 1)
        meta = conn.execute(
            """SELECT s.text, e.title, e.ts FROM episode_sentences s
               JOIN episodes e ON e.id = s.episode_id
               WHERE s.episode_id = ? AND s.idx = ?""",
            (episode_id, int(idx)),
        ).fetchone()
        out.append({
            "episode_id": episode_id,
            "idx": int(idx),
            "text": meta["text"] if meta else None,
            "episode_title": meta["title"] if meta else None,
            "episode_ts": meta["ts"] if meta else None,
            "similarity": _sim(r["distance"]),
        })
    return out


# ── Reads ─────────────────────────────────────────────────────────────────────
def get_episode(conn: sqlite3.Connection, episode_id: str) -> sqlite3.Row | None:
    return conn.execute("SELECT * FROM episodes WHERE id = ?", (episode_id,)).fetchone()


def count_episodes(conn: sqlite3.Connection) -> int:
    return conn.execute("SELECT COUNT(*) AS n FROM episodes").fetchone()["n"]


def unconsolidated_episodes(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    return conn.execute(
        """SELECT e.* FROM episodes e
           LEFT JOIN episode_consolidations c ON c.episode_id = e.id
           WHERE c.episode_id IS NULL ORDER BY e.ts""").fetchall()


# ── Replay bookkeeping ────────────────────────────────────────────────────────
def replay_seen(conn: sqlite3.Connection, old_source_id: str) -> bool:
    return conn.execute("SELECT 1 FROM replay_map WHERE old_source_id = ?",
                        (old_source_id,)).fetchone() is not None


def mark_replayed(conn: sqlite3.Connection, old_source_id: str, episode_id: str) -> None:
    conn.execute(
        "INSERT INTO replay_map (old_source_id, episode_id, replayed_at) VALUES (?, ?, ?)",
        (old_source_id, episode_id, now_iso()),
    )


def stats(conn: sqlite3.Connection) -> dict:
    counts = {}
    for table in ("episodes", "episode_sentences", "claims", "concepts",
                  "relations", "events"):
        counts[table] = conn.execute(f"SELECT COUNT(*) AS n FROM {table}").fetchone()["n"]
    last_run = conn.execute(
        "SELECT * FROM consolidation_runs ORDER BY started_at DESC LIMIT 1").fetchone()
    counts["last_consolidation"] = dict(last_run) if last_run else None
    return counts
