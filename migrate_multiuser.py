"""One-shot migration: legacy single-user engine.db → multi-user schema (AUTH.md §5).

Designates the existing corpus as the first user, backfills user_id on every
row, and rebuilds the vec/FTS virtual tables in their partitioned form. The
old file is kept as <db>.pre-multiuser.bak; the migrated copy atomically
replaces the original. Idempotent: refuses to run on an already-migrated DB.

Claim ids are NOT rewritten: the legacy unsalted ids stay globally unique while
they belong to one user, and they ride inside immutable event payloads. Only
claims minted after migration use the user-salted form (consolidate.claim_id_for).

Usage:
    python migrate_multiuser.py [--db data/engine.db] --username <name>
        [--password <pw>] [--dry-run]

The password seeds the first (admin) users row; omit it to be prompted.
"""
import argparse
import getpass
import shutil
import sqlite3
import sys
from pathlib import Path

import sqlite_vec

from core import config


def _open(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(path), timeout=30)
    conn.row_factory = sqlite3.Row
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    return conn


def migrate(db_path: str | Path, username: str, password: str,
            dry_run: bool = False) -> dict:
    src_path = Path(db_path)
    if not src_path.exists():
        raise SystemExit(f"no database at {src_path}")

    old = _open(src_path)
    cols = {r[1] for r in old.execute("PRAGMA table_info(episodes)")}
    if not cols:
        raise SystemExit(f"{src_path} has no episodes table — nothing to migrate")
    if "user_id" in cols:
        raise SystemExit(f"{src_path} is already multi-user — nothing to do")
    old.execute("PRAGMA wal_checkpoint(TRUNCATE)")

    # Build the migrated copy beside the original, then swap.
    new_path = src_path.with_suffix(".new")
    new_path.unlink(missing_ok=True)
    from core import store
    new = store.connect(new_path)

    import bcrypt
    pw_hash = bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()
    with new:
        user_id = store.create_user(new, username, pw_hash, is_admin=True)

    counts = {}
    with new:
        # Plain tables: copy with the user_id backfilled. Order matters only
        # for readability; FKs are not enforced across the copy.
        copies = [
            ("episodes",
             "INSERT INTO episodes (id, user_id, ts, raw_text, title, source, receipt_json) "
             "SELECT id, ?, ts, raw_text, title, source, receipt_json FROM old.episodes"),
            ("episode_sentences",
             "INSERT INTO episode_sentences (episode_id, user_id, idx, text) "
             "SELECT episode_id, ?, idx, text FROM old.episode_sentences"),
            ("claims",
             "INSERT INTO claims (id, user_id, text, strength, created_at, last_seen) "
             "SELECT id, ?, text, strength, created_at, last_seen FROM old.claims"),
            ("claim_support",
             "INSERT INTO claim_support (claim_id, user_id, episode_id, verbatim_sentence) "
             "SELECT claim_id, ?, episode_id, verbatim_sentence FROM old.claim_support"),
            ("concepts",
             "INSERT INTO concepts (id, user_id, label, canonical, state, strength, created_at, last_activity) "
             "SELECT id, ?, label, canonical, state, strength, created_at, last_activity FROM old.concepts"),
            ("concept_members",
             "INSERT INTO concept_members (concept_id, user_id, claim_id, weight) "
             "SELECT concept_id, ?, claim_id, weight FROM old.concept_members"),
            ("relations",
             "INSERT INTO relations (from_id, to_id, user_id, relation, weight, created_at, evidence_episode_id) "
             "SELECT from_id, to_id, ?, relation, weight, created_at, evidence_episode_id FROM old.relations"),
            # seq preserved verbatim: digest/timeline order and rebuild depend on it
            ("events",
             "INSERT INTO events (seq, user_id, ts, run_id, type, payload_json) "
             "SELECT seq, ?, ts, run_id, type, payload_json FROM old.events"),
            ("consolidation_runs",
             "INSERT INTO consolidation_runs (id, user_id, started_at, finished_at, n_episodes, batch_id, status, cost_estimate) "
             "SELECT id, ?, started_at, finished_at, n_episodes, batch_id, status, cost_estimate FROM old.consolidation_runs"),
            ("episode_consolidations",
             "INSERT INTO episode_consolidations (episode_id, user_id, run_id, consolidated_at) "
             "SELECT episode_id, ?, run_id, consolidated_at FROM old.episode_consolidations"),
            ("replay_map",
             "INSERT INTO replay_map (old_source_id, user_id, episode_id, replayed_at) "
             "SELECT old_source_id, ?, episode_id, replayed_at FROM old.replay_map"),
        ]
        new.execute("ATTACH DATABASE ? AS old", (str(src_path),))
        # episodes are immutable by trigger — drop them around the bulk INSERT?
        # No: INSERT is allowed; only UPDATE/DELETE are blocked.
        for table, sql in copies:
            new.execute(sql, (user_id,))
            counts[table] = new.execute(
                f"SELECT COUNT(*) AS n FROM {table}").fetchone()["n"]

        # Vec tables: blobs copy verbatim into the partitioned declarations.
        for table, key in (("vec_sentences", "sent_key"),
                           ("vec_claims", "claim_id"),
                           ("vec_concepts", "concept_id")):
            for r in new.execute(f"SELECT {key} AS k, embedding FROM old.{table}"):
                new.execute(
                    f"INSERT INTO {table} (user_id, {key}, embedding) VALUES (?, ?, ?)",
                    (user_id, r["k"], r["embedding"]))
            counts[table] = new.execute(
                f"SELECT COUNT(*) AS n FROM {table}").fetchone()["n"]

        # FTS: rebuild from episodes (contentless-style re-insert).
        new.execute(
            """INSERT INTO episodes_fts (episode_id, user_id, title, raw_text)
               SELECT id, user_id, COALESCE(title, ''), raw_text FROM episodes""")
        counts["episodes_fts"] = new.execute(
            "SELECT COUNT(*) AS n FROM episodes_fts").fetchone()["n"]

    # Verify row parity before swapping anything.
    for table in ("episodes", "episode_sentences", "claims", "claim_support",
                  "concepts", "concept_members", "relations", "events",
                  "consolidation_runs", "episode_consolidations", "replay_map",
                  "vec_sentences", "vec_claims", "vec_concepts"):
        n_old = old.execute(f"SELECT COUNT(*) AS n FROM {table}").fetchone()["n"]
        n_new = new.execute(f"SELECT COUNT(*) AS n FROM {table}").fetchone()["n"]
        if n_old != n_new:
            raise SystemExit(f"row-count mismatch on {table}: old={n_old} new={n_new} — aborting, original untouched")
    max_seq = (old.execute("SELECT MAX(seq) AS m FROM events").fetchone()["m"] or 0)
    new_max = (new.execute("SELECT MAX(seq) AS m FROM events").fetchone()["m"] or 0)
    if max_seq != new_max:
        raise SystemExit(f"events.seq mismatch: old={max_seq} new={new_max}")

    new.execute("DETACH DATABASE old")
    new.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    new.close()
    old.close()

    if dry_run:
        new_path.unlink(missing_ok=True)
        Path(str(new_path) + "-wal").unlink(missing_ok=True)
        Path(str(new_path) + "-shm").unlink(missing_ok=True)
        return {"dry_run": True, "user_id": user_id, "username": username, **counts}

    bak = src_path.with_suffix(".pre-multiuser.bak")
    shutil.copy2(src_path, bak)
    for ext in ("-wal", "-shm"):  # stale sidecars must not shadow the new file
        Path(str(src_path) + ext).unlink(missing_ok=True)
    new_path.replace(src_path)
    for ext in ("-wal", "-shm"):
        p = Path(str(new_path) + ext)
        if p.exists():
            p.replace(Path(str(src_path) + ext))
    return {"backup": str(bak), "user_id": user_id, "username": username, **counts}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--db", default=str(config.DB_PATH))
    ap.add_argument("--username", default=config.AUTH_USER or None,
                    help="first user's username (default: $AUTH_USER)")
    ap.add_argument("--password", default=None,
                    help="first user's password (omit to be prompted)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    if not args.username:
        sys.exit("--username required (no AUTH_USER in env)")
    password = args.password or getpass.getpass(f"Password for {args.username}: ")
    if not password:
        sys.exit("empty password")
    result = migrate(args.db, args.username, password, dry_run=args.dry_run)
    print(result)
