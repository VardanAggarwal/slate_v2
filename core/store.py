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
    id         TEXT PRIMARY KEY,            -- clm_<md5 of user_id + canonical text>
    user_id    TEXT NOT NULL,
    text       TEXT NOT NULL,
    strength   REAL DEFAULT 1.0,
    created_at TEXT,
    last_seen  TEXT
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
    type         TEXT NOT NULL,             -- ENCODED|CANONICALIZED|CONCEPT_CREATED|MERGED|
                                            -- SPLIT|BRIDGED|RELATED|DECAYED|STRENGTHENED
    payload_json TEXT NOT NULL
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
    conn.execute(f"PRAGMA user_version={SCHEMA_VERSION}")
    conn.commit()
    return conn


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
                 types: list[str] | None = None) -> list[sqlite3.Row]:
    """user_id=None spans all users — reserved for rebuild (admin path)."""
    q = "SELECT * FROM events WHERE seq > ?"
    args: list = [seq]
    if user_id is not None:
        q += " AND user_id = ?"
        args.append(user_id)
    if types:
        q += f" AND type IN ({','.join('?' * len(types))})"
        args.extend(types)
    return conn.execute(q + " ORDER BY seq", args).fetchall()


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
    """Wipe the materialized semantic store (claims/concepts/relations + vectors)
    for ALL users. Episodes, events, runs, and bookkeeping are untouched — this
    is the first half of `rebuild`, which then re-applies the event log."""
    for table in ("claim_support", "concept_members", "relations",
                  "claims", "concepts", "vec_claims", "vec_concepts"):
        conn.execute(f"DELETE FROM {table}")


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
