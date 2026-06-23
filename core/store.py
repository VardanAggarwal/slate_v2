"""SQLite + sqlite-vec store. Schema init, event-log append/apply. The ONLY module that touches the DB file. See PLAN.md §4, AUTH.md.

Invariants (PLAN.md §10):
- Episodes are immutable: there is no UPDATE path for episodes here, and there must never be one.
- The semantic store (claims/concepts/relations) is only written by consolidate.py, only via events.
- Embeddings are stored normalized; vec distance is L2, so cosine = 1 - d²/2.

Multi-user invariants (AUTH.md):
- Every row is owned by a user_id; every function takes user_id as an explicit
  parameter (never ambient state) and filters on it. Passing user_id=None is
  reserved for the few admin paths that genuinely span users (rebuild).
- Ids are globally unique across users (ULIDs everywhere; claim ids are salted
  with user_id — see consolidate.claim_id_for) because vec0 PRIMARY KEYs are
  unique across partitions, not per-partition.
"""
import hashlib
import json
import os
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path

import sqlite_vec
from sqlite_vec import serialize_float32

from core import config

SCHEMA_VERSION = 2  # 2 = multi-user (PRAGMA user_version)

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
-- USERS (admin-provisioned via CLI; AUTH.md §2/§6)
CREATE TABLE IF NOT EXISTS users (
    id         TEXT PRIMARY KEY,             -- usr_<ulid>
    username   TEXT UNIQUE NOT NULL,
    pw_hash    TEXT NOT NULL,                -- bcrypt
    created_at TEXT NOT NULL,
    is_admin   INTEGER NOT NULL DEFAULT 0
);

-- EPISODIC STORE (append-only, immutable)
CREATE TABLE IF NOT EXISTS episodes (
    id           TEXT PRIMARY KEY,          -- ep_<ulid>
    user_id      TEXT NOT NULL,
    ts           TEXT NOT NULL,             -- ISO; replayed notes keep original created_at
    raw_text     TEXT NOT NULL,
    title        TEXT,
    source       TEXT,                      -- 'mcp' | 'replay' | 'import' | ...
    receipt_json TEXT
);

CREATE TABLE IF NOT EXISTS episode_sentences (
    episode_id   TEXT NOT NULL REFERENCES episodes(id),
    user_id      TEXT NOT NULL,
    idx          INTEGER NOT NULL,
    text         TEXT NOT NULL,
    PRIMARY KEY (episode_id, idx)
);

-- SEMANTIC STORE (derived; rebuildable from episodes + events)
CREATE TABLE IF NOT EXISTS claims (
    id            TEXT PRIMARY KEY,         -- clm_<md5 of user_id + canonical text>
    user_id       TEXT NOT NULL,
    text          TEXT NOT NULL,
    strength      REAL DEFAULT 1.0,
    created_at    TEXT,
    last_seen     TEXT,
    -- C8 versioning: a claim is 'current' | 'superseded' | 'version' (held).
    -- version_group ties the rival beliefs together; superseded_by points a
    -- past belief at the one that replaced it; qualifier scopes a conditional.
    status        TEXT DEFAULT 'current',
    superseded_by TEXT,
    qualifier     TEXT,
    version_group TEXT,
    -- C9 background: a claim re-predicted enough to become background (its theme
    -- now stands for it). Orthogonal to status; demotes standalone retrieval pull.
    background    INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS claim_support (
    claim_id          TEXT NOT NULL,
    user_id           TEXT NOT NULL,
    episode_id        TEXT NOT NULL,
    verbatim_sentence TEXT,
    PRIMARY KEY (claim_id, episode_id)
);

CREATE TABLE IF NOT EXISTS concepts (
    id            TEXT PRIMARY KEY,
    user_id       TEXT NOT NULL,
    label         TEXT,
    canonical     TEXT,
    state         TEXT DEFAULT 'active',    -- 'grounded'|'active'|'stale'|'dormant'
    strength      REAL DEFAULT 1.0,
    created_at    TEXT,
    last_activity TEXT
);

CREATE TABLE IF NOT EXISTS concept_members (
    concept_id TEXT NOT NULL,
    user_id    TEXT NOT NULL,
    claim_id   TEXT NOT NULL,
    weight     REAL DEFAULT 1.0,
    PRIMARY KEY (concept_id, claim_id)
);

-- WRITE-SIDE WORKING MEMORY (derived; rebuildable from episodes + FRAGMENTED events)
-- The predictor's output: the async Write refine pass (core/write.py, W2–W8)
-- segments a note into variable-resolution fragments, routes each against memory
-- (PRD: predicted/novel/ambiguous), and persists the NOVEL/AMBIGUOUS ones here.
-- This is working memory: it may be wrong and is fully re-derivable, so — like
-- claims/concepts — it is truncated and replayed by rebuild(). Written ONLY via
-- the FRAGMENTED event applier (write.apply_fragmented); ids are deterministic
-- (frg_<md5 of user+episode+span>) so retries and rebuild are idempotent.
CREATE TABLE IF NOT EXISTS fragments (
    id            TEXT PRIMARY KEY,        -- frg_<md5>
    user_id       TEXT NOT NULL,
    episode_id    TEXT NOT NULL,           -- provenance: the immutable raw episode
    text          TEXT NOT NULL,           -- the fragment text (joined sentence span)
    sent_start    INTEGER,                 -- inclusive episode_sentences.idx range …
    sent_end      INTEGER,                 -- … this fragment spans
    medoid_idx    INTEGER,                 -- the fragment's representative sentence (episode_sentences.idx):
                                           -- its vector IS this sentence's vector in vec_sentences, so NO
                                           -- duplicate fragment vector is stored — fragment_pool/knn_fragments
                                           -- REFERENCE vec_sentences via (episode_id, medoid_idx). NULL only
                                           -- on the legacy/bare-applier path (no sentences) → no referable vector.
    route         TEXT,                    -- 'NOVEL' | 'AMBIGUOUS' (PREDICTED is never stored)
    z             REAL,                    -- surprise: residual z-score vs memory's spread
    residual      REAL,
    weight        REAL,                    -- within-note residual share (relative salience)
    anchor_id     TEXT,                    -- prior fragment it sits on (AMBIGUOUS); NULL for NOVEL
    direction     TEXT,                    -- 'contradict'|'refine'|'reinforce' (resolver, AMBIGUOUS only)
    cluster       TEXT,                    -- region/cluster this fragment belongs to: the SCOPE the
                                           -- per-cluster calibration (z_echo/prox_margin) resolves on.
                                           -- Bootstrapped at write by anchor-inheritance; re-clustered
                                           -- by consolidation. NULL → global threshold (cold/sparse).
    is_centre     INTEGER DEFAULT 0,       -- note medoid (lowest-residual stored fragment)
    is_novel_peak INTEGER DEFAULT 0,       -- note's most-novel point (highest z)
    strength      REAL DEFAULT 1.0,        -- bumped when a later fragment is PREDICTED by it
    reinforced    INTEGER DEFAULT 0,       -- count of confirmed predictions (PRD: reinforcement)
    created_at    TEXT,
    last_seen     TEXT
);

-- Bookkeeping: which episodes the Write refine pass has processed (parallel to
-- episode_consolidations). NOT truncated by rebuild — it is the "done" marker;
-- the fragment rows themselves are rebuilt from FRAGMENTED events.
CREATE TABLE IF NOT EXISTS episode_fragmentations (
    episode_id    TEXT PRIMARY KEY,
    user_id       TEXT NOT NULL,
    fragmented_at TEXT NOT NULL,
    n_fragments   INTEGER
);

CREATE TABLE IF NOT EXISTS relations (
    from_id             TEXT NOT NULL,      -- claim or concept id
    to_id               TEXT NOT NULL,
    user_id             TEXT NOT NULL,
    relation            TEXT NOT NULL,      -- 'leads_to'|'contradicts'|'supports'|'bridges'|...
    weight              REAL DEFAULT 1.0,
    created_at          TEXT,
    evidence_episode_id TEXT,
    PRIMARY KEY (from_id, to_id, relation)
);

-- EVENT LOG (backbone; semantic store is a materialized view of this)
CREATE TABLE IF NOT EXISTS events (
    seq          INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id      TEXT NOT NULL,
    ts           TEXT NOT NULL,
    run_id       TEXT,
    type         TEXT NOT NULL,             -- ENCODED|FRAGMENTED|CANONICALIZED|CONCEPT_CREATED|
                                            -- MERGED|SPLIT|BRIDGED|RELATED|DECAYED|STRENGTHENED
    payload_json TEXT NOT NULL
);

-- C12 calibration: the fitted compression/budget profile per user (the bet over
-- Q,B that consolidation owns and pushes down to Write/Retrieve). NOT event-derived
-- and NOT truncated by rebuild — it is fitted config, like consolidation_runs.
CREATE TABLE IF NOT EXISTS calibration_profiles (
    user_id      TEXT PRIMARY KEY,
    profile_json TEXT NOT NULL,
    updated_at   TEXT
);

CREATE TABLE IF NOT EXISTS consolidation_runs (
    id            TEXT PRIMARY KEY,
    user_id       TEXT NOT NULL,
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
    user_id         TEXT NOT NULL,
    run_id          TEXT NOT NULL,
    consolidated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS replay_map (
    old_source_id TEXT NOT NULL,
    user_id       TEXT NOT NULL,
    episode_id    TEXT NOT NULL,
    replayed_at   TEXT NOT NULL,
    PRIMARY KEY (old_source_id, user_id)
);

-- OAUTH STATE (AUTH.md §2: DCR clients + token chain survive restarts.
-- token_json is the serialized AccessToken/RefreshToken; the user binding
-- rides in AccessToken.claims and is mirrored in user_id so the provider can
-- rebuild its refresh->user map on boot. Tokens are bearer capabilities
-- stored plaintext — same trust domain as the corpus they unlock.)
CREATE TABLE IF NOT EXISTS oauth_clients (
    client_id   TEXT PRIMARY KEY,
    client_json TEXT NOT NULL,
    created_at  TEXT
);

CREATE TABLE IF NOT EXISTS oauth_access_tokens (
    token         TEXT PRIMARY KEY,
    user_id       TEXT,
    token_json    TEXT NOT NULL,
    refresh_token TEXT,                     -- paired refresh token (revocation cascade)
    expires_at    INTEGER
);

CREATE TABLE IF NOT EXISTS oauth_refresh_tokens (
    token      TEXT PRIMARY KEY,
    user_id    TEXT,
    token_json TEXT NOT NULL,
    expires_at INTEGER                      -- NULL = never expires
);

CREATE INDEX IF NOT EXISTS idx_events_type ON events(type);
CREATE INDEX IF NOT EXISTS idx_events_user ON events(user_id, seq);
CREATE INDEX IF NOT EXISTS idx_episodes_ts ON episodes(ts);
CREATE INDEX IF NOT EXISTS idx_episodes_user ON episodes(user_id, ts);
CREATE INDEX IF NOT EXISTS idx_claims_user ON claims(user_id);
CREATE INDEX IF NOT EXISTS idx_concepts_user ON concepts(user_id);
CREATE INDEX IF NOT EXISTS idx_fragments_user ON fragments(user_id, episode_id);

-- PLAN.md §10: episodes are immutable — enforced mechanically, not by convention
CREATE TRIGGER IF NOT EXISTS episodes_no_update BEFORE UPDATE ON episodes
BEGIN SELECT RAISE(ABORT, 'episodes are immutable (PLAN.md §10)'); END;
CREATE TRIGGER IF NOT EXISTS episodes_no_delete BEFORE DELETE ON episodes
BEGIN SELECT RAISE(ABORT, 'episodes are immutable (PLAN.md §10)'); END;
"""

# AUTH.md §1: user_id is a vec0 PARTITION KEY, so kNN restricts to one user's
# partition (verified against sqlite-vec 0.1.9 — no cross-partition spill).
_VEC_SCHEMA = [
    # sent_key = "<episode_id>:<idx>"
    f"CREATE VIRTUAL TABLE IF NOT EXISTS vec_sentences USING vec0(user_id TEXT PARTITION KEY, sent_key TEXT PRIMARY KEY, embedding FLOAT[{config.EMBED_DIM}])",
    f"CREATE VIRTUAL TABLE IF NOT EXISTS vec_claims USING vec0(user_id TEXT PARTITION KEY, claim_id TEXT PRIMARY KEY, embedding FLOAT[{config.EMBED_DIM}])",
    f"CREATE VIRTUAL TABLE IF NOT EXISTS vec_concepts USING vec0(user_id TEXT PARTITION KEY, concept_id TEXT PRIMARY KEY, embedding FLOAT[{config.EMBED_DIM}])",
    # NOTE: there is deliberately NO vec_fragments table. A fragment's vector is its
    # medoid sentence's vector, already in vec_sentences; fragment_pool/knn_fragments
    # reference it via (episode_id, medoid_idx) instead of storing a second copy.
]

_FTS_SCHEMA = "CREATE VIRTUAL TABLE IF NOT EXISTS episodes_fts USING fts5(episode_id UNINDEXED, user_id UNINDEXED, title, raw_text)"


def _is_legacy_db(conn: sqlite3.Connection) -> bool:
    """A pre-multi-user database: has an episodes table without user_id."""
    has_episodes = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='episodes'").fetchone()
    if not has_episodes:
        return False
    cols = {r[1] for r in conn.execute("PRAGMA table_info(episodes)")}
    return "user_id" not in cols


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
    if _is_legacy_db(conn):
        conn.close()
        raise RuntimeError(
            f"{path} has the legacy single-user schema — run "
            "`python migrate_multiuser.py` to migrate it (AUTH.md §5)")
    conn.executescript(_SCHEMA)
    for stmt in _VEC_SCHEMA:
        conn.execute(stmt)
    conn.execute(_FTS_SCHEMA)
    _migrate_add_columns(conn)
    conn.execute(f"PRAGMA user_version={SCHEMA_VERSION}")
    conn.commit()
    return conn


# Idempotent additive column migrations for tables that predate a column. The
# schema is otherwise CREATE-IF-NOT-EXISTS, which never adds a column to an
# existing table — so a new column on an existing table is added here.
_ADD_COLUMNS = {"fragments": {"cluster": "TEXT", "medoid_idx": "INTEGER"},
                "claims": {"status": "TEXT DEFAULT 'current'", "superseded_by": "TEXT",
                           "qualifier": "TEXT", "version_group": "TEXT",
                           "background": "INTEGER DEFAULT 0"}}


def _migrate_add_columns(conn: sqlite3.Connection) -> None:
    # Reclaim the now-unused duplicate-vector index from pre-reference DBs. The
    # fragment vector is referenced from vec_sentences (via medoid_idx); a rebuild
    # repopulates fragments.medoid_idx. Harmless no-op once gone.
    conn.execute("DROP TABLE IF EXISTS vec_fragments")
    for table, cols in _ADD_COLUMNS.items():
        have = {r["name"] for r in conn.execute(f"PRAGMA table_info({table})")}
        for col, decl in cols.items():
            if col not in have:
                conn.execute(f"ALTER TABLE {table} ADD COLUMN {col} {decl}")


# ── Users (AUTH.md §2/§6) ─────────────────────────────────────────────────────
def create_user(conn: sqlite3.Connection, username: str, pw_hash: str,
                is_admin: bool = False) -> str:
    user_id = "usr_" + ulid()
    conn.execute(
        "INSERT INTO users (id, username, pw_hash, created_at, is_admin) VALUES (?, ?, ?, ?, ?)",
        (user_id, username, pw_hash, now_iso(), int(is_admin)))
    return user_id


def get_user(conn: sqlite3.Connection, user_id: str) -> sqlite3.Row | None:
    return conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()


def get_user_by_username(conn: sqlite3.Connection, username: str) -> sqlite3.Row | None:
    return conn.execute("SELECT * FROM users WHERE username = ?", (username,)).fetchone()


def list_users(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    return conn.execute("SELECT * FROM users ORDER BY created_at").fetchall()


def count_users(conn: sqlite3.Connection) -> int:
    return conn.execute("SELECT COUNT(*) AS n FROM users").fetchone()["n"]


def set_user_password(conn: sqlite3.Connection, username: str, pw_hash: str) -> bool:
    cur = conn.execute("UPDATE users SET pw_hash = ? WHERE username = ?",
                       (pw_hash, username))
    return cur.rowcount > 0


def delete_user(conn: sqlite3.Connection, username: str) -> bool:
    """Remove the login only. The user's corpus stays (episodes are immutable
    by trigger anyway) — orphaned data is unreachable without a user row."""
    cur = conn.execute("DELETE FROM users WHERE username = ?", (username,))
    return cur.rowcount > 0


# ── OAuth persistence (AUTH.md §2) ────────────────────────────────────────────
# Written by SlateOAuthProvider (mcp_server.py) so issued tokens, the refresh
# chain, and dynamically-registered clients survive server restarts.

def save_oauth_client(conn: sqlite3.Connection, client_id: str, client_json: str) -> None:
    conn.execute(
        """INSERT INTO oauth_clients (client_id, client_json, created_at)
           VALUES (?, ?, ?)
           ON CONFLICT(client_id) DO UPDATE SET client_json = excluded.client_json""",
        (client_id, client_json, now_iso()))


def load_oauth_clients(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    return conn.execute("SELECT * FROM oauth_clients").fetchall()


def save_oauth_access_token(conn: sqlite3.Connection, token: str, user_id: str | None,
                            token_json: str, refresh_token: str | None,
                            expires_at: int | None) -> None:
    conn.execute(
        "INSERT OR REPLACE INTO oauth_access_tokens (token, user_id, token_json, refresh_token, expires_at) VALUES (?, ?, ?, ?, ?)",
        (token, user_id, token_json, refresh_token, expires_at))


def save_oauth_refresh_token(conn: sqlite3.Connection, token: str, user_id: str | None,
                             token_json: str, expires_at: int | None) -> None:
    conn.execute(
        "INSERT OR REPLACE INTO oauth_refresh_tokens (token, user_id, token_json, expires_at) VALUES (?, ?, ?, ?)",
        (token, user_id, token_json, expires_at))


def load_oauth_access_tokens(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    return conn.execute("SELECT * FROM oauth_access_tokens").fetchall()


def load_oauth_refresh_tokens(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    return conn.execute("SELECT * FROM oauth_refresh_tokens").fetchall()


def delete_oauth_tokens(conn: sqlite3.Connection, access_token: str | None = None,
                        refresh_token: str | None = None) -> None:
    """Delete a token and its paired counterpart (mirrors the in-memory
    provider's pair revocation)."""
    if access_token:
        row = conn.execute(
            "SELECT refresh_token FROM oauth_access_tokens WHERE token = ?",
            (access_token,)).fetchone()
        if row and row["refresh_token"]:
            conn.execute("DELETE FROM oauth_refresh_tokens WHERE token = ?",
                         (row["refresh_token"],))
        conn.execute("DELETE FROM oauth_access_tokens WHERE token = ?", (access_token,))
    if refresh_token:
        conn.execute("DELETE FROM oauth_refresh_tokens WHERE token = ?", (refresh_token,))
        conn.execute("DELETE FROM oauth_access_tokens WHERE refresh_token = ?",
                     (refresh_token,))


def purge_expired_oauth_tokens(conn: sqlite3.Connection, now: int) -> None:
    """Boot-time cleanup. An expired refresh token kills its pair; an expired
    access token dies alone — its (non-expiring) refresh token can still mint
    a new pair, which is what lets clients outlive long downtime."""
    for r in conn.execute(
            "SELECT token FROM oauth_refresh_tokens WHERE expires_at IS NOT NULL AND expires_at < ?",
            (now,)).fetchall():
        conn.execute("DELETE FROM oauth_access_tokens WHERE refresh_token = ?",
                     (r["token"],))
    conn.execute("DELETE FROM oauth_refresh_tokens WHERE expires_at IS NOT NULL AND expires_at < ?", (now,))
    conn.execute("DELETE FROM oauth_access_tokens WHERE expires_at IS NOT NULL AND expires_at < ?", (now,))


# ── Event log ─────────────────────────────────────────────────────────────────
def append_event(conn: sqlite3.Connection, user_id: str, type_: str, payload: dict,
                 run_id: str | None = None) -> int:
    cur = conn.execute(
        "INSERT INTO events (user_id, ts, run_id, type, payload_json) VALUES (?, ?, ?, ?, ?)",
        (user_id, now_iso(), run_id, type_, json.dumps(payload, ensure_ascii=False)),
    )
    return cur.lastrowid


def events_since(conn: sqlite3.Connection, user_id: str | None, seq: int = 0,
                 types: list[str] | None = None,
                 include_rolled_back: bool = True) -> list[sqlite3.Row]:
    """user_id=None spans all users — reserved for rebuild (admin path).

    Default returns the FULL log (audit/inspection). Materialization (rebuild,
    re-derive) passes ``include_rolled_back=False`` so a rolled-back run's events
    are skipped — the run never happened as far as the semantic store is concerned
    (events stay on disk for audit; only their materialization is suppressed)."""
    q = "SELECT * FROM events WHERE seq > ?"
    args: list = [seq]
    if user_id is not None:
        q += " AND user_id = ?"
        args.append(user_id)
    if types:
        q += f" AND type IN ({','.join('?' * len(types))})"
        args.extend(types)
    if not include_rolled_back:
        q += " AND " + ACTIVE_RUN_PREDICATE
    return conn.execute(q + " ORDER BY seq", args).fetchall()


# A run's events are materialized only while the run is not rolled back. Write-side
# events (run_id IS NULL) are never consolidation runs, so always active.
ACTIVE_RUN_PREDICATE = (
    "(run_id IS NULL OR run_id NOT IN "
    "(SELECT id FROM consolidation_runs WHERE status = 'rolled_back'))"
)


# ── Episodic writes (called by encode.py inside one transaction) ──────────────
def new_episode_id(ts_unix: float | None = None) -> str:
    return "ep_" + ulid(ts_unix)


def insert_episode(conn: sqlite3.Connection, user_id: str, episode_id: str, ts: str,
                   raw_text: str, title: str | None, source: str, receipt: dict,
                   sentences: list[str], embeddings) -> None:
    """Write one episode + its sentences + vectors + FTS row. Caller owns the transaction."""
    conn.execute(
        "INSERT INTO episodes (id, user_id, ts, raw_text, title, source, receipt_json) VALUES (?, ?, ?, ?, ?, ?, ?)",
        (episode_id, user_id, ts, raw_text, title, source, json.dumps(receipt, ensure_ascii=False)),
    )
    for idx, sent in enumerate(sentences):
        conn.execute(
            "INSERT INTO episode_sentences (episode_id, user_id, idx, text) VALUES (?, ?, ?, ?)",
            (episode_id, user_id, idx, sent),
        )
        conn.execute(
            "INSERT INTO vec_sentences (user_id, sent_key, embedding) VALUES (?, ?, ?)",
            (user_id, f"{episode_id}:{idx}", serialize_float32([float(x) for x in embeddings[idx]])),
        )
    conn.execute(
        "INSERT INTO episodes_fts (episode_id, user_id, title, raw_text) VALUES (?, ?, ?, ?)",
        (episode_id, user_id, title or "", raw_text),
    )


# ── kNN lookups ───────────────────────────────────────────────────────────────
def _sim(l2_distance: float) -> float:
    """Normalized vectors: cosine similarity = 1 - (L2 distance)² / 2."""
    return 1.0 - (l2_distance * l2_distance) / 2.0


def knn_claims(conn: sqlite3.Connection, user_id: str, embedding, k: int = 5) -> list[dict]:
    """Nearest canonical claims to one sentence embedding, within one user's partition."""
    rows = conn.execute(
        "SELECT claim_id, distance FROM vec_claims WHERE embedding MATCH ? AND k = ? AND user_id = ?",
        (serialize_float32([float(x) for x in embedding]), k, user_id),
    ).fetchall()
    out = []
    for r in rows:
        claim = conn.execute("SELECT text, strength FROM claims WHERE id = ? AND user_id = ?",
                             (r["claim_id"], user_id)).fetchone()
        out.append({
            "claim_id": r["claim_id"],
            "text": claim["text"] if claim else None,
            "strength": claim["strength"] if claim else None,
            "similarity": _sim(r["distance"]),
        })
    return out


def knn_sentences(conn: sqlite3.Connection, user_id: str, embedding, k: int = 5) -> list[dict]:
    """Nearest prior episode sentences to one sentence embedding, within one user's partition."""
    rows = conn.execute(
        "SELECT sent_key, distance FROM vec_sentences WHERE embedding MATCH ? AND k = ? AND user_id = ?",
        (serialize_float32([float(x) for x in embedding]), k, user_id),
    ).fetchall()
    out = []
    for r in rows:
        episode_id, idx = r["sent_key"].rsplit(":", 1)
        meta = conn.execute(
            """SELECT s.text, e.title, e.ts FROM episode_sentences s
               JOIN episodes e ON e.id = s.episode_id
               WHERE s.episode_id = ? AND s.idx = ? AND e.user_id = ?""",
            (episode_id, int(idx), user_id),
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
def get_episode(conn: sqlite3.Connection, user_id: str, episode_id: str) -> sqlite3.Row | None:
    return conn.execute("SELECT * FROM episodes WHERE id = ? AND user_id = ?",
                        (episode_id, user_id)).fetchone()


def count_episodes(conn: sqlite3.Connection, user_id: str) -> int:
    return conn.execute("SELECT COUNT(*) AS n FROM episodes WHERE user_id = ?",
                        (user_id,)).fetchone()["n"]


def episode_sentences_with_vectors(conn: sqlite3.Connection, user_id: str,
                                   episode_id: str) -> list[dict]:
    """An episode's sentences in order, each with its W1-persisted embedding —
    so the Write refine pass segments/routes over the vectors already stored, with
    no re-embed. Returns [{idx, text, embedding(np)}] ordered by idx."""
    rows = conn.execute(
        """SELECT s.idx, s.text, v.embedding FROM episode_sentences s
           JOIN vec_sentences v ON v.sent_key = s.episode_id || ':' || s.idx
           WHERE s.episode_id = ? AND s.user_id = ? ORDER BY s.idx""",
        (episode_id, user_id)).fetchall()
    return [{"idx": r["idx"], "text": r["text"],
             "embedding": _deserialize(r["embedding"])} for r in rows]


def unconsolidated_episodes(conn: sqlite3.Connection, user_id: str) -> list[sqlite3.Row]:
    return conn.execute(
        """SELECT e.* FROM episodes e
           LEFT JOIN episode_consolidations c ON c.episode_id = e.id
           WHERE c.episode_id IS NULL AND e.user_id = ? ORDER BY e.ts""",
        (user_id,)).fetchall()


def users_with_unconsolidated(conn: sqlite3.Connection) -> list[str]:
    """Every user_id that has episodes awaiting consolidation (cron loops these,
    so a corpus saved under a user with no users row still gets consolidated)."""
    return [r["user_id"] for r in conn.execute(
        """SELECT DISTINCT e.user_id FROM episodes e
           LEFT JOIN episode_consolidations c ON c.episode_id = e.id
           WHERE c.episode_id IS NULL ORDER BY e.user_id""")]


def all_user_ids(conn: sqlite3.Connection) -> list[str]:
    """Every user_id present in the corpus (episodes ∪ users table)."""
    return [r["uid"] for r in conn.execute(
        """SELECT DISTINCT user_id AS uid FROM episodes
           UNION SELECT id AS uid FROM users ORDER BY uid""")]


# ── Replay bookkeeping ────────────────────────────────────────────────────────
def replay_seen(conn: sqlite3.Connection, user_id: str, old_source_id: str) -> bool:
    return conn.execute(
        "SELECT 1 FROM replay_map WHERE old_source_id = ? AND user_id = ?",
        (old_source_id, user_id)).fetchone() is not None


def mark_replayed(conn: sqlite3.Connection, user_id: str, old_source_id: str,
                  episode_id: str) -> None:
    conn.execute(
        "INSERT INTO replay_map (old_source_id, user_id, episode_id, replayed_at) VALUES (?, ?, ?, ?)",
        (old_source_id, user_id, episode_id, now_iso()),
    )


# ── Semantic store writes ─────────────────────────────────────────────────────
# Called ONLY by consolidate.py event appliers (PLAN.md §10). Any other caller
# is re-creating the v1 save-time-mutation architecture — don't.

def _deserialize(blob: bytes):
    import numpy as np
    return np.frombuffer(blob, dtype=np.float32)


def insert_claim(conn: sqlite3.Connection, user_id: str, claim_id: str, text: str,
                 embedding, ts: str) -> None:
    cur = conn.execute(
        """INSERT INTO claims (id, user_id, text, strength, created_at, last_seen)
           VALUES (?, ?, ?, 1.0, ?, ?) ON CONFLICT(id) DO NOTHING""",
        (claim_id, user_id, text, ts, ts))
    if cur.rowcount:  # only write the vector for a genuinely new claim
        conn.execute("INSERT INTO vec_claims (user_id, claim_id, embedding) VALUES (?, ?, ?)",
                     (user_id, claim_id, serialize_float32([float(x) for x in embedding])))


def add_claim_support(conn: sqlite3.Connection, user_id: str, claim_id: str,
                      episode_id: str, verbatim: str | None) -> None:
    conn.execute(
        """INSERT INTO claim_support (claim_id, user_id, episode_id, verbatim_sentence)
           VALUES (?, ?, ?, ?) ON CONFLICT DO NOTHING""",
        (claim_id, user_id, episode_id, verbatim))


def bump_claim_strength(conn: sqlite3.Connection, user_id: str, claim_id: str,
                        ts: str, delta: float) -> None:
    conn.execute(
        "UPDATE claims SET strength = strength + ?, last_seen = ? WHERE id = ? AND user_id = ?",
        (delta, ts, claim_id, user_id))


def get_claim(conn: sqlite3.Connection, user_id: str, claim_id: str) -> sqlite3.Row | None:
    return conn.execute("SELECT * FROM claims WHERE id = ? AND user_id = ?",
                        (claim_id, user_id)).fetchone()


def claim_embedding(conn: sqlite3.Connection, user_id: str, claim_id: str):
    row = conn.execute(
        "SELECT embedding FROM vec_claims WHERE claim_id = ? AND user_id = ?",
        (claim_id, user_id)).fetchone()
    return _deserialize(row["embedding"]) if row else None


def claim_in_any_concept(conn: sqlite3.Connection, user_id: str, claim_id: str) -> bool:
    return conn.execute(
        "SELECT 1 FROM concept_members WHERE claim_id = ? AND user_id = ? LIMIT 1",
        (claim_id, user_id)).fetchone() is not None


def delete_claim(conn: sqlite3.Connection, user_id: str, claim_id: str) -> None:
    """Drop a claim from working memory (claim + support + vector). The raw
    episodes it was derived from are untouched, so it stays re-derivable — this
    is only ever called on a guard-confirmed reconstructable claim (C7 prune)."""
    conn.execute("DELETE FROM claim_support WHERE claim_id = ? AND user_id = ?",
                 (claim_id, user_id))
    conn.execute("DELETE FROM concept_members WHERE claim_id = ? AND user_id = ?",
                 (claim_id, user_id))
    conn.execute("DELETE FROM claims WHERE id = ? AND user_id = ?", (claim_id, user_id))
    conn.execute("DELETE FROM vec_claims WHERE claim_id = ?", (claim_id,))


def set_claim_version(conn: sqlite3.Connection, user_id: str, claim_id: str,
                      **fields) -> None:
    """C8 — set a claim's version state. Additive; only the named columns move."""
    allowed = {"status", "superseded_by", "qualifier", "version_group"}
    sets = {k: v for k, v in fields.items() if k in allowed}
    if not sets:
        return
    assign = ", ".join(f"{k} = ?" for k in sets)
    conn.execute(f"UPDATE claims SET {assign} WHERE id = ? AND user_id = ?",
                 (*sets.values(), claim_id, user_id))


def set_claim_background(conn: sqlite3.Connection, user_id: str, claim_id: str,
                         value: int = 1) -> None:
    """C9 — mark a claim background (its theme now represents it). Reversible."""
    conn.execute("UPDATE claims SET background = ? WHERE id = ? AND user_id = ?",
                 (int(value), claim_id, user_id))


def claim_support_count(conn: sqlite3.Connection, user_id: str, claim_id: str) -> int:
    return conn.execute(
        "SELECT COUNT(*) AS n FROM claim_support WHERE claim_id = ? AND user_id = ?",
        (claim_id, user_id)).fetchone()["n"]


def claim_source_episodes(conn: sqlite3.Connection, user_id: str, claim_id: str) -> list[str]:
    """Episodes that support a claim (the reverse of `claims_for_episode`). Used by
    consolidation's fragment→concept prior to find a claim's Write-time source note."""
    return [r["episode_id"] for r in conn.execute(
        "SELECT DISTINCT episode_id FROM claim_support WHERE claim_id = ? AND user_id = ?",
        (claim_id, user_id))]


def episode_fragments(conn: sqlite3.Connection, user_id: str, episode_id: str) -> list[dict]:
    """Fragments of ONE episode as {id, route, anchor_id, cluster, embedding} — the
    medoid vector REFERENCED from vec_sentences (no duplicate stored), same join as
    `fragment_pool` but scoped + carrying route/anchor. Empty when the episode hasn't
    been refined yet (Write async), so consolidation's prior degrades to a no-op."""
    rows = conn.execute(
        """SELECT f.id, f.route, f.anchor_id, f.cluster, v.embedding FROM fragments f
           JOIN vec_sentences v ON v.sent_key = f.episode_id || ':' || f.medoid_idx
           WHERE f.user_id = ? AND f.episode_id = ? AND f.medoid_idx IS NOT NULL""",
        (user_id, episode_id)).fetchall()
    return [{"id": r["id"], "route": r["route"], "anchor_id": r["anchor_id"],
             "cluster": r["cluster"], "embedding": _deserialize(r["embedding"])}
            for r in rows]


def fragment_episode(conn: sqlite3.Connection, user_id: str, frag_id: str) -> str | None:
    """The source episode of a fragment — the C13 bridge from a retrieval signal
    (which keys on fragment ids) to the claims derived from the same episode."""
    row = conn.execute(
        "SELECT episode_id FROM fragments WHERE id = ? AND user_id = ?",
        (frag_id, user_id)).fetchone()
    return row["episode_id"] if row else None


def claims_for_episode(conn: sqlite3.Connection, user_id: str, episode_id: str) -> list[str]:
    return [r["claim_id"] for r in conn.execute(
        "SELECT DISTINCT claim_id FROM claim_support WHERE episode_id = ? AND user_id = ?",
        (episode_id, user_id))]


def contradiction_pairs(conn: sqlite3.Connection, user_id: str,
                        claim_ids: list[str] | None = None) -> list[tuple[str, str]]:
    """(from_id, to_id) for every 'contradicts' edge, optionally restricted to
    edges touching `claim_ids`. The conflict signal C8 reconciles."""
    q = ("SELECT from_id, to_id FROM relations WHERE relation = 'contradicts' "
         "AND user_id = ?")
    rows = conn.execute(q, (user_id,)).fetchall()
    pairs = [(r["from_id"], r["to_id"]) for r in rows]
    if claim_ids is not None:
        s = set(claim_ids)
        pairs = [p for p in pairs if p[0] in s or p[1] in s]
    return pairs


def claim_versions(conn: sqlite3.Connection, user_id: str,
                   version_group: str) -> list[sqlite3.Row]:
    """All claims belonging to one belief's version group (current + held +
    superseded). Used to surface 'contested' at retrieval."""
    return conn.execute(
        "SELECT * FROM claims WHERE version_group = ? AND user_id = ? ORDER BY id",
        (version_group, user_id)).fetchall()


def insert_concept(conn: sqlite3.Connection, user_id: str, concept_id: str,
                   label: str, canonical: str, ts: str) -> None:
    conn.execute(
        """INSERT INTO concepts (id, user_id, label, canonical, state, strength, created_at, last_activity)
           VALUES (?, ?, ?, ?, 'active', 1.0, ?, ?) ON CONFLICT(id) DO NOTHING""",
        (concept_id, user_id, label, canonical, ts, ts))


def update_concept(conn: sqlite3.Connection, user_id: str, concept_id: str, **fields) -> None:
    allowed = {"label", "canonical", "state", "strength", "last_activity"}
    sets = {k: v for k, v in fields.items() if k in allowed and v is not None}
    if not sets:
        return
    assign = ", ".join(f"{k} = ?" for k in sets)
    conn.execute(f"UPDATE concepts SET {assign} WHERE id = ? AND user_id = ?",
                 (*sets.values(), concept_id, user_id))


def delete_concept(conn: sqlite3.Connection, user_id: str, concept_id: str) -> None:
    """Remove a concept from the materialized view (history stays in events).
    Ownership is checked first; concept ids are globally unique ULIDs, so the
    vec delete by PK cannot touch another user's partition."""
    if get_concept(conn, user_id, concept_id) is None:
        return
    conn.execute("DELETE FROM concept_members WHERE concept_id = ? AND user_id = ?",
                 (concept_id, user_id))
    conn.execute("DELETE FROM concepts WHERE id = ? AND user_id = ?",
                 (concept_id, user_id))
    conn.execute("DELETE FROM vec_concepts WHERE concept_id = ?", (concept_id,))


def add_concept_member(conn: sqlite3.Connection, user_id: str, concept_id: str,
                       claim_id: str, weight: float = 1.0) -> None:
    conn.execute(
        """INSERT INTO concept_members (concept_id, user_id, claim_id, weight)
           VALUES (?, ?, ?, ?) ON CONFLICT DO NOTHING""",
        (concept_id, user_id, claim_id, weight))


def remove_concept_member(conn: sqlite3.Connection, user_id: str, concept_id: str,
                          claim_id: str) -> None:
    conn.execute(
        "DELETE FROM concept_members WHERE concept_id = ? AND claim_id = ? AND user_id = ?",
        (concept_id, claim_id, user_id))


def concept_member_ids(conn: sqlite3.Connection, user_id: str, concept_id: str) -> list[str]:
    return [r["claim_id"] for r in conn.execute(
        "SELECT claim_id FROM concept_members WHERE concept_id = ? AND user_id = ? ORDER BY claim_id",
        (concept_id, user_id))]


def get_concept(conn: sqlite3.Connection, user_id: str, concept_id: str) -> sqlite3.Row | None:
    return conn.execute("SELECT * FROM concepts WHERE id = ? AND user_id = ?",
                        (concept_id, user_id)).fetchone()


def all_concepts(conn: sqlite3.Connection, user_id: str) -> list[sqlite3.Row]:
    return conn.execute("SELECT * FROM concepts WHERE user_id = ? ORDER BY id",
                        (user_id,)).fetchall()


def recompute_concept_embedding(conn: sqlite3.Connection, user_id: str,
                                concept_id: str) -> None:
    """Concept vector = normalized mean of member claim vectors (deterministic,
    so event-log rebuild reproduces it exactly)."""
    import numpy as np
    members = concept_member_ids(conn, user_id, concept_id)
    vecs = [claim_embedding(conn, user_id, c) for c in members]
    vecs = [v for v in vecs if v is not None]
    conn.execute("DELETE FROM vec_concepts WHERE concept_id = ?", (concept_id,))
    if not vecs:
        return
    mean = np.mean(vecs, axis=0)
    norm = np.linalg.norm(mean)
    if norm > 0:
        mean = mean / norm
    conn.execute("INSERT INTO vec_concepts (user_id, concept_id, embedding) VALUES (?, ?, ?)",
                 (user_id, concept_id, serialize_float32([float(x) for x in mean])))


def knn_concepts(conn: sqlite3.Connection, user_id: str, embedding, k: int = 5) -> list[dict]:
    rows = conn.execute(
        "SELECT concept_id, distance FROM vec_concepts WHERE embedding MATCH ? AND k = ? AND user_id = ?",
        (serialize_float32([float(x) for x in embedding]), k, user_id)).fetchall()
    out = []
    for r in rows:
        c = get_concept(conn, user_id, r["concept_id"])
        if c:
            out.append({**dict(c), "similarity": _sim(r["distance"])})
    return out


def insert_relation(conn: sqlite3.Connection, user_id: str, from_id: str, to_id: str,
                    relation: str, weight: float, ts: str,
                    evidence_episode_id: str | None = None) -> None:
    conn.execute(
        """INSERT INTO relations (from_id, to_id, user_id, relation, weight, created_at, evidence_episode_id)
           VALUES (?, ?, ?, ?, ?, ?, ?)
           ON CONFLICT(from_id, to_id, relation) DO UPDATE SET weight = excluded.weight""",
        (from_id, to_id, user_id, relation, weight, ts, evidence_episode_id))


def truncate_semantic(conn: sqlite3.Connection) -> None:
    """Wipe the materialized semantic store (claims/concepts/relations + the
    Write-side fragments, plus their vectors) for ALL users. Episodes, events,
    runs, and bookkeeping (episode_consolidations / episode_fragmentations) are
    untouched — this is the first half of `rebuild`, which then re-applies the
    event log (FRAGMENTED events rebuild the fragments)."""
    for table in ("claim_support", "concept_members", "relations",
                  "claims", "concepts", "vec_claims", "vec_concepts",
                  "fragments"):
        conn.execute(f"DELETE FROM {table}")


# ── Write-side working memory: fragments (called ONLY by write.apply_fragmented;
#    PLAN.md §10 — derived store written via events, never at the API boundary) ──
def fragment_id_for(user_id: str, episode_id: str, sent_start: int, sent_end: int) -> str:
    """Deterministic fragment id: same (user, episode, span) → same id, so a
    refine retry or a full event-log rebuild re-inserts the same row (ON CONFLICT
    DO NOTHING) rather than minting a duplicate. Mirrors consolidate.claim_id_for."""
    raw = f"{user_id}\x00{episode_id}\x00{sent_start}-{sent_end}".encode("utf-8")
    return "frg_" + hashlib.md5(raw).hexdigest()[:24]


def insert_fragment(conn: sqlite3.Connection, user_id: str, frag_id: str,
                    episode_id: str, text: str, sent_start: int, sent_end: int,
                    route: str, z: float | None, residual: float | None,
                    weight: float | None, anchor_id: str | None,
                    direction: str | None, is_centre: bool, is_novel_peak: bool,
                    ts: str, strength: float = 1.0, cluster: str | None = None,
                    medoid_idx: int | None = None) -> None:
    """Insert one routed fragment. Idempotent on the deterministic id. NO fragment
    vector is stored — the routing/retrieval vector is the medoid SENTENCE's vector,
    already in vec_sentences and referenced via (episode_id, medoid_idx). `strength`
    is the initial hold — a contradiction is born held STRONGER than a refine (PRD W6:
    "store, held strongest, flag"). `cluster` is the region the per-cluster calibration
    resolves on (NULL → global threshold)."""
    conn.execute(
        """INSERT INTO fragments
             (id, user_id, episode_id, text, sent_start, sent_end, medoid_idx, route,
              z, residual, weight, anchor_id, direction, cluster, is_centre,
              is_novel_peak, strength, reinforced, created_at, last_seen)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, ?, ?)
           ON CONFLICT(id) DO NOTHING""",
        (frag_id, user_id, episode_id, text, sent_start, sent_end, medoid_idx, route,
         z, residual, weight, anchor_id, direction, cluster, int(is_centre),
         int(is_novel_peak), strength, ts, ts))


def bump_fragment_strength(conn: sqlite3.Connection, user_id: str, frag_id: str,
                           ts: str, delta: float) -> None:
    """Reinforce a fragment a later one was PREDICTED by (PRD: a confirmed
    prediction strengthens what predicted it, stores nothing new). No-op if the
    anchor isn't a fragment row (e.g. it was itself PREDICTED and never stored)."""
    conn.execute(
        """UPDATE fragments SET strength = strength + ?, reinforced = reinforced + 1,
           last_seen = ? WHERE id = ? AND user_id = ?""",
        (delta, ts, frag_id, user_id))


def fragment_pool(conn: sqlite3.Connection, user_id: str) -> list[dict]:
    """All of a user's fragments as measure() corpus rows {id,text,embedding,
    cluster}. This is the memory M the Write match pass routes against. The
    embedding is the fragment's medoid SENTENCE vector, REFERENCED from vec_sentences
    via (episode_id, medoid_idx) — no duplicate fragment vector is stored. `cluster`
    is the region a fragment was assigned (anchor-inheritance at write, re-clustered
    by consolidation); NULL → measure falls back to its local-neighbour spread and
    decide() resolves the GLOBAL threshold (cold/sparse regions)."""
    rows = conn.execute(
        """SELECT f.id, f.text, f.cluster, v.embedding FROM fragments f
           JOIN vec_sentences v ON v.sent_key = f.episode_id || ':' || f.medoid_idx
           WHERE f.user_id = ? AND f.medoid_idx IS NOT NULL""", (user_id,)).fetchall()
    return [{"id": r["id"], "text": r["text"],
             "embedding": _deserialize(r["embedding"]), "cluster": r["cluster"]}
            for r in rows]


def knn_fragments(conn: sqlite3.Connection, user_id: str, embedding, k: int = 5) -> list[dict]:
    """Nearest stored fragments to an embedding, within one user's partition. A
    fragment's vector IS its medoid sentence's vector (no duplicate is stored), so
    this searches vec_sentences and maps medoid hits back to fragments. vec0 returns
    hits in ascending-distance order, so the first k that are medoids are exactly the
    k nearest fragments; we over-fetch and widen to all sentences if a batch is short."""
    medoids = {f"{r['episode_id']}:{r['medoid_idx']}": r for r in conn.execute(
        "SELECT id, episode_id, medoid_idx, text, route, strength FROM fragments "
        "WHERE user_id = ? AND medoid_idx IS NOT NULL", (user_id,))}
    if not medoids:
        return []
    total = conn.execute("SELECT COUNT(*) AS n FROM episode_sentences WHERE user_id = ?",
                         (user_id,)).fetchone()["n"]
    emb = serialize_float32([float(x) for x in embedding])
    kq = min(total, max(k * 8, 32))
    while True:
        rows = conn.execute(
            "SELECT sent_key, distance FROM vec_sentences WHERE embedding MATCH ? AND k = ? AND user_id = ?",
            (emb, kq, user_id)).fetchall()
        out = []
        for r in rows:
            f = medoids.get(r["sent_key"])
            if f:
                out.append({"frag_id": f["id"], "text": f["text"], "route": f["route"],
                            "strength": f["strength"], "similarity": _sim(r["distance"])})
                if len(out) >= k:
                    break
        if len(out) >= k or kq >= total:
            return out
        kq = min(total, kq * 2)


def fragment_candidates(conn: sqlite3.Connection, user_id: str, embedding,
                        k: int = 40) -> list[dict]:
    """Nearest fragments to a query embedding, enriched for RETRIEVE: each carries
    its medoid SENTENCE vector (the retrieval vector, referenced from vec_sentences)
    plus episode provenance, so core/retrieve.py can run the assembly wrapper over
    them and format verbatim spans with sources. Like knn_fragments but returns the
    embedding + episode_id/title/ts/cluster — the seed candidate set for assembly.

    vec0 returns sentence hits in ascending-distance order; we map the medoid hits
    back to fragments and widen the probe until we have k (or exhaust the corpus)."""
    medoids = {f"{r['episode_id']}:{r['medoid_idx']}": dict(r) for r in conn.execute(
        """SELECT f.id, f.episode_id, f.medoid_idx, f.text, f.route, f.strength,
                  f.cluster, f.weight, e.title, e.ts
           FROM fragments f JOIN episodes e ON e.id = f.episode_id
           WHERE f.user_id = ? AND f.medoid_idx IS NOT NULL""", (user_id,))}
    if not medoids:
        return []
    total = conn.execute("SELECT COUNT(*) AS n FROM episode_sentences WHERE user_id = ?",
                         (user_id,)).fetchone()["n"]
    emb = serialize_float32([float(x) for x in embedding])
    kq = min(total, max(k * 8, 32))
    while True:
        rows = conn.execute(
            "SELECT sent_key, embedding, distance FROM vec_sentences "
            "WHERE embedding MATCH ? AND k = ? AND user_id = ?",
            (emb, kq, user_id)).fetchall()
        out = []
        for r in rows:
            f = medoids.get(r["sent_key"])
            if f:
                out.append({"id": f["id"], "frag_id": f["id"], "text": f["text"],
                            "embedding": _deserialize(r["embedding"]),
                            "episode_id": f["episode_id"], "title": f["title"],
                            "ts": f["ts"], "cluster": f["cluster"],
                            "route": f["route"], "strength": f["strength"],
                            "weight": f["weight"], "similarity": _sim(r["distance"])})
                if len(out) >= k:
                    break
        if len(out) >= k or kq >= total:
            return out
        kq = min(total, kq * 2)


def fragment_count(conn: sqlite3.Connection, user_id: str) -> int:
    return conn.execute("SELECT COUNT(*) AS n FROM fragments WHERE user_id = ?",
                        (user_id,)).fetchone()["n"]


def unfragmented_episodes(conn: sqlite3.Connection, user_id: str) -> list[sqlite3.Row]:
    """Episodes the Write refine pass hasn't processed yet (the retry queue —
    parallel to unconsolidated_episodes). Oldest first."""
    return conn.execute(
        """SELECT e.* FROM episodes e
           LEFT JOIN episode_fragmentations f ON f.episode_id = e.id
           WHERE f.episode_id IS NULL AND e.user_id = ? ORDER BY e.ts""",
        (user_id,)).fetchall()


def users_with_unfragmented(conn: sqlite3.Connection) -> list[str]:
    return [r["user_id"] for r in conn.execute(
        """SELECT DISTINCT e.user_id FROM episodes e
           LEFT JOIN episode_fragmentations f ON f.episode_id = e.id
           WHERE f.episode_id IS NULL ORDER BY e.user_id""")]


def mark_fragmented(conn: sqlite3.Connection, user_id: str, episode_id: str,
                    n_fragments: int) -> bool:
    """Atomically CLAIM this episode for fragmentation. Returns True iff this
    caller won the claim (inserted the marker), False if it already existed.

    This is the race guard: refine_episode runs it as the FIRST statement of its
    commit transaction, so when a background trigger and a refine_pending sweep
    process the same episode concurrently, SQLite serialises the two writes and
    only the winner gets True — the loser sees the committed marker, returns
    False, and skips append_event/apply_fragmented, so no duplicate FRAGMENTED
    event is logged and the (non-idempotent) reinforce bump runs exactly once."""
    cur = conn.execute(
        """INSERT INTO episode_fragmentations (episode_id, user_id, fragmented_at, n_fragments)
           VALUES (?, ?, ?, ?) ON CONFLICT(episode_id) DO NOTHING""",
        (episode_id, user_id, now_iso(), n_fragments))
    return cur.rowcount > 0


# ── Consolidation runs + bookkeeping ──────────────────────────────────────────
def start_run(conn: sqlite3.Connection, user_id: str, run_id: str, n_episodes: int) -> None:
    conn.execute(
        """INSERT INTO consolidation_runs (id, user_id, started_at, n_episodes, status)
           VALUES (?, ?, ?, ?, 'running')""",
        (run_id, user_id, now_iso(), n_episodes))


def finish_run(conn: sqlite3.Connection, run_id: str, status: str,
               cost_estimate: float, batch_id: str | None = None) -> None:
    conn.execute(
        """UPDATE consolidation_runs SET finished_at = ?, status = ?,
           cost_estimate = ?, batch_id = ? WHERE id = ?""",
        (now_iso(), status, cost_estimate, batch_id, run_id))


def last_run(conn: sqlite3.Connection, user_id: str) -> sqlite3.Row | None:
    return conn.execute(
        """SELECT * FROM consolidation_runs WHERE user_id = ?
           ORDER BY started_at DESC LIMIT 1""", (user_id,)).fetchone()


def get_calibration(conn: sqlite3.Connection, user_id: str) -> dict:
    """C12 — the fitted calibration profile for a user, or {} if none fitted."""
    row = conn.execute(
        "SELECT profile_json FROM calibration_profiles WHERE user_id = ?",
        (user_id,)).fetchone()
    return json.loads(row["profile_json"]) if row else {}


def set_calibration(conn: sqlite3.Connection, user_id: str, profile: dict) -> None:
    conn.execute(
        """INSERT INTO calibration_profiles (user_id, profile_json, updated_at)
           VALUES (?, ?, ?) ON CONFLICT(user_id) DO UPDATE SET
           profile_json = excluded.profile_json, updated_at = excluded.updated_at""",
        (user_id, json.dumps(profile), now_iso()))


def get_run(conn: sqlite3.Connection, run_id: str) -> sqlite3.Row | None:
    return conn.execute(
        "SELECT * FROM consolidation_runs WHERE id = ?", (run_id,)).fetchone()


def mark_run_rolled_back(conn: sqlite3.Connection, run_id: str) -> None:
    """Flag a run rolled back. Its events stay on disk (audit) but are excluded
    from materialization by ACTIVE_RUN_PREDICATE — the reversible commit."""
    conn.execute(
        "UPDATE consolidation_runs SET status = 'rolled_back' WHERE id = ?",
        (run_id,))


def unconsolidate_run_episodes(conn: sqlite3.Connection, run_id: str) -> int:
    """Drop the consolidated-bookkeeping rows for a run so its episodes are
    re-eligible for the next consolidate() pass (re-derive from raw)."""
    cur = conn.execute(
        "DELETE FROM episode_consolidations WHERE run_id = ?", (run_id,))
    return cur.rowcount


def mark_consolidated(conn: sqlite3.Connection, user_id: str, episode_id: str,
                      run_id: str) -> None:
    conn.execute(
        """INSERT INTO episode_consolidations (episode_id, user_id, run_id, consolidated_at)
           VALUES (?, ?, ?, ?) ON CONFLICT DO NOTHING""",
        (episode_id, user_id, run_id, now_iso()))


def stats(conn: sqlite3.Connection, user_id: str) -> dict:
    counts = {}
    for table in ("episodes", "episode_sentences", "claims", "concepts",
                  "relations", "events"):
        counts[table] = conn.execute(
            f"SELECT COUNT(*) AS n FROM {table} WHERE user_id = ?",
            (user_id,)).fetchone()["n"]
    run = last_run(conn, user_id)
    counts["last_consolidation"] = dict(run) if run else None
    return counts
