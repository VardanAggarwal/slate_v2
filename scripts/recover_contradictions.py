"""Recover the belief-versioning that never ran, without re-consolidating.

WHY RE-CONSOLIDATING DOES NOT WORK
----------------------------------
`_relations` (consolidate.py) is the only producer of claim-level 'contradicts'
edges, and it reads them out of `episodes.receipt_json["contradictions"]`. Those
receipts were written while classify_stance() was silently returning "neutral",
so every one says `contradictions: []` — and `episodes` is immutable by DB
trigger (store.py: episodes_no_update / episodes_no_delete). Replaying the
pipeline re-reads the same empty lists and reproduces the same nothing.

Consequence measured on prod: 1 'contradicts' relation (concept→concept, from an
LLM blueprint spine link, which _reconcile skips because get_claim returns None),
0 VERSIONED events, 0 of 3533 claims carrying a version_group. The whole
supersede / scope / version subsystem has never produced a row.

WHAT THIS DOES INSTEAD
----------------------
Goes straight at the layer re-consolidation was supposed to reach: appends
RELATED{relation:'contradicts'} events for confirmed claim pairs, then runs
consolidate._reconcile over the touched claims so the normal C8 path emits
VERSIONED. Events are the source of truth (apply_event / rebuild), so this is
purely ADDITIVE — no receipt rewrite, no DELETE anywhere.

TWO CANDIDATE SOURCES
---------------------
--from-sweep         A fresh sentence-vs-canonical-claim sweep over every
                     episode, gated at similarity >= ECHO_THRESHOLD exactly like
                     the receipt path. This is where the zero-shot classifier is
                     actually sound (near-duplicate premise/hypothesis), unlike
                     the ungated fragment path where it measured 0.125 precision.
                     Note this is a NEW capability, not a faithful repair: each
                     episode is compared against today's full claim pool, which
                     is far larger than the pool that existed at its save time.

--from-fragments P   The confirmed rows of a backfill_stance.py report, mapped
                     fragment -> episode -> claims. Coarser (claim granularity is
                     per-episode), but that is the same granularity _relations
                     already uses.

Both feed one LLM adjudication stage and one emit path.

ORDERING MATTERS: _reconcile treats (from_id, to_id) as (newer challenger, older
incumbent) and only flips the current view when the challenger outweighs the
incumbent. Pairs are therefore always oriented by the earliest supporting
episode's timestamp; a pair with no resolvable order is dropped, not guessed.

REVERSIBILITY: claim version state is snapshotted into an insert-only
`contradiction_recovery` table before any write, and --revert restores it with an
UPDATE. The 'contradicts' relation rows are deliberately LEFT in place: the
contradiction is a true finding about the corpus, and removing rows would mean a
DELETE. A later consolidation is free to re-reconcile them.

USAGE
    python scripts/recover_contradictions.py --from-sweep --count-only
    python scripts/recover_contradictions.py --from-sweep
    python scripts/recover_contradictions.py --from-sweep --apply
    python scripts/recover_contradictions.py --from-fragments /tmp/bf_full.json --apply
    python scripts/recover_contradictions.py --revert <run_id>
"""
import argparse
import json
import sqlite3
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backfill_stance import (  # noqa: E402  — shared stages
    LLM_STANCE_PROVIDERS, adjudicate, classify_strict)
from core import config, consolidate, encode, store  # noqa: E402

AUDIT_DDL = """
CREATE TABLE IF NOT EXISTS contradiction_recovery (
    run_id       TEXT NOT NULL,
    claim_id     TEXT NOT NULL,
    old_status   TEXT,
    old_version_group TEXT,
    old_superseded_by TEXT,
    old_qualifier     TEXT,
    applied_at   TEXT NOT NULL,
    PRIMARY KEY (run_id, claim_id)
)
"""


def _claim_first_seen(conn, user_id):
    """claim_id -> earliest supporting episode ts. That is when the belief first
    appeared, which is what orients challenger vs incumbent."""
    rows = conn.execute(
        """SELECT cs.claim_id, MIN(e.ts) AS first_ts
             FROM claim_support cs JOIN episodes e ON e.id = cs.episode_id
            WHERE cs.user_id = ? GROUP BY cs.claim_id""", (user_id,)).fetchall()
    return {r["claim_id"]: r["first_ts"] for r in rows}


def _episode_claims(conn, user_id):
    """episode_id -> [claim_id]."""
    out = {}
    for r in conn.execute(
            "SELECT episode_id, claim_id FROM claim_support WHERE user_id = ?",
            (user_id,)).fetchall():
        out.setdefault(r["episode_id"], []).append(r["claim_id"])
    return out


def _users(conn):
    return [r["user_id"] for r in conn.execute(
        "SELECT DISTINCT user_id FROM claims").fetchall()]


def _nearest_claim(conn, user_id, claim_ids, emb, cache):
    """The ONE claim among claim_ids closest to emb.

    Without this, a contradicting sentence expands to every claim in its episode:
    12 confirmed findings became 250 claim pairs, which would write 250
    'contradicts' edges and version-group unrelated beliefs. _relations has the
    same flaw (it takes episode_claim_ids[0], i.e. an arbitrary one) — nearest is
    strictly better than either.
    """
    import numpy as np
    best, best_sim = None, -2.0
    v = np.asarray(emb, dtype=float)
    nv = np.linalg.norm(v) or 1.0
    for cid in claim_ids:
        if cid not in cache:
            cache[cid] = store.claim_embedding(conn, user_id, cid)
        ce = cache[cid]
        if ce is None:
            continue
        c = np.asarray(ce, dtype=float)
        sim = float(v @ c / (nv * (np.linalg.norm(c) or 1.0)))
        if sim > best_sim:
            best, best_sim = cid, sim
    return best, (round(best_sim, 3) if best else None)


def sweep_candidates(conn, user_id, first_seen, ep_claims, count_only=False,
                     limit=None, verbose=True, gate=None, k=3):
    """Option 2: every episode sentence vs canonical claims, gated at ECHO_THRESHOLD.

    Mirrors _build_receipt's comparison, with two exclusions it also needs:
    claims supported by this same episode (self-match), and claims that are not
    strictly older than the sentence's episode (no orientable challenge).
    """
    episodes = conn.execute(
        "SELECT id, ts, title FROM episodes WHERE user_id = ? ORDER BY ts",
        (user_id,)).fetchall()
    if limit:
        episodes = episodes[:limit]

    gate = config.ECHO_THRESHOLD if gate is None else gate
    gated, cands = 0, []
    emb_cache = {}
    for n, ep in enumerate(episodes, 1):
        own = set(ep_claims.get(ep["id"], []))
        sents = store.episode_sentences_with_vectors(conn, user_id, ep["id"])
        for s in sents:
            for hit in store.knn_claims(conn, user_id, s["embedding"], k=k):
                if hit["similarity"] < gate or not hit["text"]:
                    continue
                cid = hit["claim_id"]
                if cid in own:
                    continue                       # a claim distilled from this note
                older = first_seen.get(cid)
                if not older or older >= ep["ts"]:
                    continue                       # not an older incumbent
                gated += 1
                if count_only:
                    continue
                # ONE challenger: the episode's claim nearest this sentence, not
                # all 18-44 of them (see _nearest_claim).
                newer, newer_sim = _nearest_claim(conn, user_id, own,
                                                  s["embedding"], emb_cache)
                if not newer:
                    continue
                cands.append({
                    "id": f"{ep['id']}:{s['idx']}",
                    "episode_id": ep["id"], "ts": ep["ts"], "title": ep["title"],
                    "older_claim_id": cid,
                    "newer_claim_ids": [newer],
                    "newer_claim_sim": newer_sim,
                    # adjudicate() reads these two keys
                    "anchor_text": hit["text"], "text": s["text"],
                    "similarity": round(hit["similarity"], 3),
                    "anchor_ts": older, "intra_note": False,
                    "source": "sweep",
                })
        if verbose and n % 50 == 0:
            print(f"  swept {n}/{len(episodes)} episodes — {gated} gated pairs",
                  file=sys.stderr, flush=True)
    return gated, cands


def gate_curve(conn, user_id, first_seen, ep_claims, gates, k=3, limit=None):
    """How many pairs each ECHO_THRESHOLD would admit. Pure vector search — no
    model calls — so the reach/cost tradeoff can be priced before spending any.

    Reported alongside the retrieval-side meaning of the same knob: ECHO_THRESHOLD
    also decides what the live receipt calls an echo, so moving it is not a
    sweep-only change.
    """
    lowest = min(gates)
    episodes = conn.execute(
        "SELECT id, ts FROM episodes WHERE user_id = ? ORDER BY ts", (user_id,)).fetchall()
    if limit:
        episodes = episodes[:limit]
    sims = []
    for ep in episodes:
        own = set(ep_claims.get(ep["id"], []))
        for s in store.episode_sentences_with_vectors(conn, user_id, ep["id"]):
            for hit in store.knn_claims(conn, user_id, s["embedding"], k=k):
                if not hit["text"] or hit["claim_id"] in own:
                    continue
                older = first_seen.get(hit["claim_id"])
                if not older or older >= ep["ts"]:
                    continue
                if hit["similarity"] >= lowest:
                    sims.append(hit["similarity"])
    return {f"{g:.2f}": sum(1 for x in sims if x >= g) for g in sorted(gates, reverse=True)}


def fragment_candidates(conn, user_id, report_path, first_seen, ep_claims):
    """Option 1: a backfill_stance report's confirmed rows -> claim pairs."""
    report = json.loads(Path(report_path).read_text())
    out, cache = [], {}
    for r in report.get("confirmed_rows", []):
        if r.get("user_id") != user_id:
            continue
        newer_pool = sorted(ep_claims.get(r["episode_id"], []))
        older_pool = [c for c in ep_claims.get(r["anchor_episode_id"], [])
                      if c not in set(newer_pool)]
        if not newer_pool or not older_pool:
            continue
        # orient by the anchor episode actually being older
        if not (r.get("anchor_ts") and r["anchor_ts"] < r["ts"]):
            continue
        # Fragments carry no stored vector (store.py: "deliberately NO
        # vec_fragments"), so embed the two texts once to pick ONE claim a side.
        # Without this the pool crossed both episodes' claims — a double fan-out.
        try:
            embs = encode.get_embedder().encode(
                [r["text"], r["anchor_text"]], normalize_embeddings=True,
                show_progress_bar=False)
        except Exception as e:                     # noqa: BLE001
            print(f"  embed failed for {r['id']}: {type(e).__name__}", file=sys.stderr)
            continue
        newer, n_sim = _nearest_claim(conn, user_id, newer_pool, embs[0], cache)
        older, o_sim = _nearest_claim(conn, user_id, older_pool, embs[1], cache)
        if newer and older and newer != older:
            out.append({
                "id": r["id"], "episode_id": r["episode_id"], "ts": r["ts"],
                "title": r["title"], "older_claim_id": older,
                "newer_claim_ids": [newer],
                "newer_claim_sim": n_sim, "older_claim_sim": o_sim,
                "anchor_text": r["anchor_text"], "text": r["text"],
                "similarity": None, "anchor_ts": r["anchor_ts"],
                "intra_note": r.get("intra_note", False),
                "source": "fragments",
                # already LLM-confirmed upstream; do not re-adjudicate
                "llm_stance": "contradiction", "llm_why": r.get("llm_why", ""),
            })
    return out


def stage1_filter(cands, verbose=True):
    """Cheap zero-shot recall filter. Sound here because the sweep gate keeps
    premise/hypothesis near-duplicates — unlike the ungated fragment path."""
    keep, failures = [], []
    for i, c in enumerate(cands, 1):
        try:
            # classify_strict, NOT classify_stance: the latter swallows provider
            # failures into "neutral", which silently suppresses findings while
            # reporting zero failures (see backfill_stance.classify_strict).
            if classify_strict(c["anchor_text"], c["text"]) == "contradiction":
                keep.append(c)
        except Exception as e:                     # noqa: BLE001 — never silent
            failures.append({"id": c["id"], "error": f"{type(e).__name__}: {e}"})
        if verbose and i % 50 == 0:
            print(f"  stage1 {i}/{len(cands)} — {len(keep)} shortlisted",
                  file=sys.stderr, flush=True)
    return keep, failures


def apply_recovery(conn, user_id, confirmed, dry=True):
    """Emit RELATED{contradicts} for each confirmed pair, then let the normal C8
    path (consolidate._reconcile) decide supersede / scope / version."""
    pairs = []
    seen = set()
    for c in confirmed:
        for newer in c["newer_claim_ids"]:
            if newer == c["older_claim_id"]:
                continue
            key = (newer, c["older_claim_id"])
            if key in seen:
                continue
            seen.add(key)
            pairs.append((newer, c["older_claim_id"], c))
    if dry or not pairs:
        return {"pairs": len(pairs), "applied": False}

    run_id = "run_" + store.ulid()
    ts = time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime())
    touched = sorted({p for a, b, _ in pairs for p in (a, b)})

    conn.execute(AUDIT_DDL)
    with conn:
        store.start_run(conn, user_id, run_id, 0)
        for cid in touched:                        # snapshot BEFORE anything moves
            row = conn.execute(
                "SELECT status, version_group, superseded_by, qualifier FROM claims"
                " WHERE id = ? AND user_id = ?", (cid, user_id)).fetchone()
            if row is None:
                continue
            conn.execute(
                "INSERT OR IGNORE INTO contradiction_recovery (run_id, claim_id,"
                " old_status, old_version_group, old_superseded_by, old_qualifier,"
                " applied_at) VALUES (?,?,?,?,?,?,?)",
                (run_id, cid, row["status"], row["version_group"],
                 row["superseded_by"], row["qualifier"], ts))
        for newer, older, c in pairs:
            consolidate.emit(conn, user_id, run_id, "RELATED", {
                "from_id": newer, "to_id": older, "relation": "contradicts",
                "weight": 1.0, "evidence_episode_id": c["episode_id"], "ts": ts})

    # _reconcile is an LLM stage; pin the provider so a failure cannot fall
    # through to a billed rung mid-bulk-run.
    saved = config.LLM_FALLBACK_ORDER
    config.LLM_FALLBACK_ORDER = ["openrouter"]
    try:
        cost = consolidate._reconcile(conn, user_id, run_id, touched, ts)
    finally:
        config.LLM_FALLBACK_ORDER = saved

    versioned = conn.execute(
        "SELECT COUNT(*) n FROM events WHERE type='VERSIONED' AND run_id=?",
        (run_id,)).fetchone()["n"]
    return {"pairs": len(pairs), "applied": True, "run_id": run_id,
            "claims_touched": len(touched), "versioned_events": versioned,
            "reconcile_cost": round(cost, 4)}


def revert(conn, run_id):
    conn.execute(AUDIT_DDL)
    rows = conn.execute(
        "SELECT claim_id, old_status, old_version_group, old_superseded_by,"
        " old_qualifier FROM contradiction_recovery WHERE run_id = ?",
        (run_id,)).fetchall()
    with conn:
        for r in rows:
            conn.execute(
                "UPDATE claims SET status=?, version_group=?, superseded_by=?,"
                " qualifier=? WHERE id=?",
                (r["old_status"], r["old_version_group"], r["old_superseded_by"],
                 r["old_qualifier"], r["claim_id"]))
        store.mark_run_rolled_back(conn, run_id)
    print(f"restored {len(rows)} claims; run {run_id} flagged rolled-back.")
    print("NOTE: the 'contradicts' relation rows are left in place by design "
          "(removing them would be a DELETE, and the finding itself is true).")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--from-sweep", action="store_true")
    ap.add_argument("--from-fragments", metavar="REPORT")
    ap.add_argument("--count-only", action="store_true",
                    help="how many pairs clear the gate, with zero model calls")
    ap.add_argument("--gate", type=float,
                    help=f"override ECHO_THRESHOLD for the sweep only "
                         f"(default {config.ECHO_THRESHOLD})")
    ap.add_argument("--gate-curve", metavar="G", type=float, nargs="+",
                    help="pairs admitted at each threshold; no model calls")
    ap.add_argument("--k", type=int, default=3, help="claims per sentence (knn)")
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--limit", type=int, help="first N episodes (sweep only)")
    ap.add_argument("--report", default="contradiction_recovery_report.json")
    ap.add_argument("--revert", metavar="RUN_ID")
    args = ap.parse_args()

    conn = store.connect(config.DB_PATH)
    conn.row_factory = sqlite3.Row

    if args.revert:
        return revert(conn, args.revert)

    if args.gate_curve:
        for user_id in _users(conn):
            curve = gate_curve(conn, user_id, _claim_first_seen(conn, user_id),
                               _episode_claims(conn, user_id), args.gate_curve,
                               k=args.k, limit=args.limit)
            print(f"\n[{user_id}]  (k={args.k})")
            for g, n in curve.items():
                print(f"   >= {g}   {n:6d} pairs")
        print(f"\nlive ECHO_THRESHOLD = {config.ECHO_THRESHOLD} "
              f"(NOVELTY_THRESHOLD = {config.NOVELTY_THRESHOLD})")
        return

    if not (args.from_sweep or args.from_fragments):
        sys.exit("pick a source: --from-sweep and/or --from-fragments REPORT")

    if not args.count_only:
        probe = encode.classify_stance("I love working in the office.",
                                       "I hate working in the office.")
        if probe != "contradiction":
            sys.exit(f"STANCE_PROVIDER={config.STANCE_PROVIDER!r} self-check FAILED "
                     f"(got {probe!r}) — fix the provider before recovering.")
        print(f"provider {config.STANCE_PROVIDER!r} ok", file=sys.stderr)

    report = {"users": {}, "applied": bool(args.apply)}
    for user_id in _users(conn):
        first_seen = _claim_first_seen(conn, user_id)
        ep_claims = _episode_claims(conn, user_id)
        u = {"gated": 0, "stage1": 0, "confirmed": 0}

        cands = []
        if args.from_sweep:
            gated, sc = sweep_candidates(conn, user_id, first_seen, ep_claims,
                                         count_only=args.count_only, limit=args.limit,
                                         gate=args.gate, k=args.k)
            u["gated"] = gated
            u["gate"] = args.gate if args.gate is not None else config.ECHO_THRESHOLD
            print(f"[{user_id}] sweep gated {gated} sentence/claim pairs",
                  file=sys.stderr)
            if args.count_only:
                report["users"][user_id] = u
                continue
            if config.STANCE_PROVIDER in LLM_STANCE_PROVIDERS:
                # Stage 1 is a CHEAP recall filter; with an LLM provider it is not
                # cheap, and it charges the same 20 req/min budget stage 2 needs.
                # Adjudicate every gated pair once, with the stricter rubric.
                print(f"[{user_id}] provider {config.STANCE_PROVIDER!r} is LLM-backed "
                      f"— skipping stage 1, adjudicating all {len(sc)} gated pairs",
                      file=sys.stderr)
                u["stage1"] = None
                u["stage1_skipped_reason"] = f"{config.STANCE_PROVIDER} is LLM-backed"
                cands += sc
            else:
                shortlist, f1 = stage1_filter(sc)
                u["stage1"], u["stage1_failed"] = len(shortlist), len(f1)
                u["stage1_failure_rows"] = f1          # detail, not just a count
                cands += shortlist

        pre_confirmed = []
        if args.from_fragments:
            fc = fragment_candidates(conn, user_id, args.from_fragments,
                                     first_seen, ep_claims)
            pre_confirmed = fc          # already adjudicated upstream
            u["from_fragments"] = len(fc)

        confirmed = list(pre_confirmed)
        if cands:
            print(f"[{user_id}] stage2 over {len(cands)} — OpenRouter only",
                  file=sys.stderr)
            ok, rejected, f2 = adjudicate(cands)
            confirmed += ok
            u.update(stage2_confirmed=len(ok), stage2_rejected=len(rejected),
                     stage2_failed=len(f2), stage2_failure_rows=f2,
                     precision_of_input=(round(len(ok) / len(cands), 3) if cands else None))
            # Keep the rejections: when the run confirms nothing, these ARE the
            # result, and a reader has to be able to check the adjudicator was
            # right rather than take a zero on faith.
            u["rejected_rows"] = rejected
        u["confirmed"] = len(confirmed)

        res = apply_recovery(conn, user_id, confirmed, dry=not args.apply)
        u["emit"] = res
        u["confirmed_rows"] = confirmed
        report["users"][user_id] = u

    Path(args.report).write_text(json.dumps(report, indent=2, ensure_ascii=False))
    slim = {uid: {k: v for k, v in u.items()
                  if k not in ("confirmed_rows", "rejected_rows")}
            for uid, u in report["users"].items()}
    print(json.dumps({"applied": report["applied"], "users": slim}, indent=2))

    def show(label, rows, n):
        for r in rows[:n]:
            print(f"\n  {label} [{r['ts'][:10]}] {(r['title'] or '')[:52]}  ({r['source']}"
                  f"{', sim ' + str(r['similarity']) if r['similarity'] else ''})"
                  f"\n  OLDER : {r['anchor_text'][:88]}"
                  f"\n  NEWER : {r['text'][:88]}"
                  f"\n  WHY   : {r.get('llm_why', '')}")

    for uid, u in report["users"].items():
        show("CONFIRMED", u.get("confirmed_rows", []), 12)
        show("REJECTED ", u.get("rejected_rows", []), 8)
    print(f"\nfull report -> {args.report}")


if __name__ == "__main__":
    main()
