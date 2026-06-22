"""Write pipeline — W2–W8 of the execution plan: the predictor-driven layer.

W1 (persist the immutable raw episode + its sentence vectors + the cheap novelty
receipt + the ENCODED event) is synchronous and lives in `core/encode.py`. THIS
module is the asynchronous, retryable remainder. It:

  W2+W3  segment the persisted note into VARIABLE-RESOLUTION fragments
         (`scan.fragments` — boundary + granularity off one surprise signal), each
         represented by its MEDOID sentence (selection, not generation — reusing
         the encode-time sentence vectors, NO second embedding pass),
  W4+W5  route the fragments against memory in ONE BATCHED pass (single gemm vs the
         fixed corpus), plus a cheap causal sibling check for intra-note dedup: a
         within-note echo collapses to a PREDICTED route whose anchor is a sibling
         fragment, re-measured against memory ∪ kept-siblings only when a sibling
         actually competes,
  W6     label store/reinforce/resolve (`predict.decide`) and resolve the
         DIRECTION of the ambiguous ones (`predict.resolve_direction`, the LLM —
         only those),
  W7     mark the note's centre (lowest-residual stored fragment) and most-novel
         point (highest z), read off the same measurements,
  W8     persist the NOVEL/AMBIGUOUS fragments as working memory via a FRAGMENTED
         event, and reinforce the anchors the PREDICTED ones confirmed.

Why async + event-sourced:
  - It may call the LLM resolver for the AMBIGUOUS fragments — network work that
    can fail and must be retryable; W1 must never wait on it. (Fragment vectors are
    medoid sentences, so refine no longer makes an embedding call.) `save_note` kicks
    this off in a thread (`trigger_refine_async`); `refine_pending` sweeps
    anything HF/LLM failures left unfragmented.
  - Like consolidation, the routing is nondeterministic (it depends on memory at
    the time + the LLM), so it is decided ONCE, frozen in the FRAGMENTED payload,
    and a deterministic applier (`apply_fragmented`) materializes it — `rebuild()`
    replays the frozen decision exactly. Fragment ids are content-addressed
    (`store.fragment_id_for`), so a retry or a rebuild re-inserts the same rows
    instead of minting duplicates.

Layering: a THIN orchestrator over the spine (`scan` + `predict`), never
re-derived geometry. The pure core (`plan_fragments`, `route_fragments`) takes
embeddings + an in-memory corpus and is fully unit-testable with no DB and no
network; only `refine_episode` touches the store and the embedder.
"""
from __future__ import annotations

import logging
import threading

import numpy as np

from core import config, predict, scan, store

log = logging.getLogger("slate.write")

# Route labels are predict.decide()'s public contract.
PREDICTED, NOVEL, AMBIGUOUS = "PREDICTED", "NOVEL", "AMBIGUOUS"


# ── injected defaults (production wiring; lazy so importing write is cheap) ────
def _memory_pool(conn, user_id: str) -> list[dict]:
    """The memory M the Write match pass routes against. Currently the Write-side
    working layer (prior fragments) — self-contained and useful before any
    consolidation has run. This is the single seam to widen M (e.g. union the
    canonical claims) without touching the routing logic."""
    return store.fragment_pool(conn, user_id)


def _load_calibration(conn, user_id: str) -> dict:
    """The calibration profile the bets read from. Seam for C12 (consolidation
    fits z_echo / drop_z per corpus from SR@B and persists it, then pushes it
    down here). For now the module defaults, merged so the scan lookups
    (drop_z/fold_z) and the route lookups (z_echo/prox_margin) resolve from one
    profile."""
    return {**scan.DEFAULT_CALIBRATION, **predict.DEFAULT_CALIBRATION,
            "per_cluster": {}}


# ══════════════════════════════════════════════════════════════════════════════
# PURE CORE — no DB, no network. Embeddings + corpus in, routing plan out.
# ══════════════════════════════════════════════════════════════════════════════
def plan_fragments(sent_texts: list[str], sent_embs, *,
                   calibration: dict | None = None) -> list[tuple[str, int, int]]:
    """W2+W3 — segment a note into variable-resolution fragments over its sentence
    vectors. Returns [(text, sent_start, sent_end)] partitioning the note, where
    text is the joined sentence span and the indices are inclusive.

    Intra-note dedup is NOT done here — it falls out of the vs-memory routing in
    route_fragments, where a fragment that a sibling already covers takes the
    PREDICTED route (residual-based, the predictor's spread-relative z_echo)."""
    E = np.atleast_2d(np.asarray(sent_embs, dtype=float))
    groups = scan.fragments(E, calibration=calibration)
    specs = []
    for g in groups:
        if not g:
            continue
        specs.append((" ".join(sent_texts[i] for i in g), g[0], g[-1]))
    return specs


def _nearest_kept_sibling(sib_sims, kept_idx: list[int]) -> tuple[int | None, float]:
    """Among earlier KEPT siblings, the one most similar to this fragment (and how
    similar). (None, -inf) when nothing has been kept yet."""
    best_j, best = None, float("-inf")
    for j in kept_idx:
        if sib_sims[j] > best:
            best_j, best = j, float(sib_sims[j])
    return best_j, best


def _region_for(route: str, anchor_id: str | None, fid: str,
                memory_by_id: dict, kept_by_fid: dict) -> str:
    """The cluster/region a kept fragment is assigned — the scope its per-cluster
    z_echo/prox_margin resolves on. NOVEL opens its OWN region (open territory);
    AMBIGUOUS INHERITS the region of the anchor it sits on (it refines/contests
    that region). An anchor with no region yet (cold) seeds a fresh one. This is a
    bootstrap over the anchor graph — consolidation re-clusters later."""
    if route == NOVEL or anchor_id is None:
        return fid
    anchor_row = memory_by_id.get(anchor_id) or kept_by_fid.get(anchor_id)
    return (anchor_row.get("cluster") if anchor_row else None) or fid


def _fragment_medoid(sent_texts: list[str], sent_embs, s: int, e: int) -> tuple[int, np.ndarray]:
    """A fragment's representative = its MEDOID sentence: the one its own siblings
    in the span reconstruct best (lowest within-span residual), via the spine's
    `measure(exclude_self)`. Selection, not generation — the routing vector points
    at a REAL statement, reusing the encode-time sentence vectors (no second
    embedding pass). A single-sentence span is its own representative; for a fully
    redundant pair either sentence serves (ties break to the first). Returns the
    absolute episode-sentence index of the medoid and its vector.

    Mirrors W7's note-centre one level down: note-centre = lowest-residual fragment;
    fragment-centre = lowest-residual sentence. A folded (large) fragment is large
    BECAUSE scan judged its sentences mutually predictable, so its medoid faithfully
    stands for the whole span — the surprising sentences already left as their own
    fine fragments."""
    if e <= s:
        return s, np.asarray(sent_embs[s], dtype=float)
    items = [{"id": j, "text": sent_texts[j], "embedding": sent_embs[j]}
             for j in range(s, e + 1)]
    ms = predict.measure(items, items, exclude_self=True)
    best = min(range(len(items)), key=lambda k: ms[k]["residual"])
    return s + best, np.asarray(sent_embs[s + best], dtype=float)


def fragment_representatives(specs: list[tuple[str, int, int]], sent_texts: list[str],
                             sent_embs) -> tuple[np.ndarray, list[int]]:
    """Medoid vector + medoid sentence index for each fragment spec. The vectors
    are what routing/storage use (no re-embed); the indices are provenance."""
    sent_embs = np.atleast_2d(np.asarray(sent_embs, dtype=float))
    reps = [_fragment_medoid(sent_texts, sent_embs, s, e) for _, s, e in specs]
    vecs = np.vstack([v for _, v in reps]) if reps else np.empty((0, sent_embs.shape[1]))
    return vecs, [j for j, _ in reps]


def route_fragments(specs: list[tuple[str, int, int]], frag_embs, memory: list[dict],
                    *, classify=None, calibration: dict | None = None,
                    frag_id_fn=None, medoid_idxs: list[int] | None = None) -> dict:
    """W4–W7 — route each fragment against memory ∪ siblings, in note order.

    PREDICTED  → reinforce the anchor (a prior or sibling fragment); store nothing.
    NOVEL      → store, no anchor (open territory).
    AMBIGUOUS  → store + resolve DIRECTION (contradict/refine/reinforce) via the
                 LLM, the ONLY route that reaches it; flag the anchor it sits on.

    Batched: every fragment is measured against the FIXED memory in ONE gemm
    (`measure` processes the whole note together — no per-fragment corpus rebuild).
    The only sequential work is intra-note dedup, which depends on which earlier
    siblings were KEPT: a fragment is re-measured against memory ∪ kept-siblings
    ONLY when an earlier kept sibling is nearer than its best memory match (i.e. it
    could be a within-note echo). The common case reuses the batched measurement,
    so the old per-fragment full-corpus rescan is gone; the union-z it computed is
    preserved exactly for the fragments where a sibling actually competes.

    Pure: `frag_embs` are the fragment vectors (caller embeds/pools, outside any
    txn), `memory` is the measure() corpus. `frag_id_fn(start, end)` assigns each
    kept fragment its stable id up front, so an anchor reference is the same id
    whether it points at a prior-note fragment or a sibling in this note."""
    calibration = calibration or predict.DEFAULT_CALIBRATION
    frag_id_fn = frag_id_fn or (lambda s, e: f"tmp:{s}-{e}")
    frag_embs = np.atleast_2d(np.asarray(frag_embs, dtype=float))
    # medoid_idx per fragment = the episode-sentence its vector references; absent
    # (direct callers passing raw frag_embs) it defaults to the span start, a real
    # sentence index. Production passes the true medoids from fragment_representatives.
    medoid_of = (lambda i, s: medoid_idxs[i]) if medoid_idxs is not None else (lambda i, s: s)
    baselines = (predict.compute_baselines(memory)
                 if len(memory) >= predict.WARMUP_MIN_CORPUS else None)

    # one batched pass: all fragments vs the fixed memory (single gemm inside measure)
    frag_dicts = [{"text": t, "embedding": e} for (t, _, _), e in zip(specs, frag_embs)]
    M = predict.measure(frag_dicts, memory, baselines=baselines)
    sib_gram = frag_embs @ frag_embs.T     # k×k sibling similarities

    memory_by_id = {r["id"]: r for r in memory}
    kept_by_fid: dict = {}                 # fid → kept sibling row (anchor lookup, n_intra)
    kept_rows: list[dict] = []             # kept siblings in causal order (the union Y)
    kept_idx: list[int] = []               # their fragment indices (sib_gram lookup)
    planned: list[dict] = []
    reinforced: list[str] = []
    n_intra = 0

    for i, ((text, s, e), emb) in enumerate(zip(specs, frag_embs)):
        fid = frag_id_fn(s, e)
        m = M[i]
        sib_j, sib_sim = _nearest_kept_sibling(sib_gram[i], kept_idx)
        m_near = m["nearest_sim"] if m["nearest_sim"] is not None else float("-inf")
        if sib_j is not None and sib_sim > m_near:    # a sibling belongs in the neighbourhood
            m = predict.measure([{"text": text, "embedding": emb}],
                                memory + kept_rows, baselines=baselines)[0]

        route = predict.decide([m], calibration)["fragments"][0]["route"]
        if route == PREDICTED:
            anchor = m["anchor_id"]
            if anchor is not None:
                reinforced.append(anchor)
                if anchor in kept_by_fid:
                    n_intra += 1     # the echo collapsed against a sibling, not memory
            continue

        direction, strength = None, 1.0
        # anchor is meaningful only for AMBIGUOUS (it sits ON something);
        # NOVEL is open territory — decide() already nulls it.
        anchor_id = m["anchor_id"] if route == AMBIGUOUS else None
        if route == AMBIGUOUS and m.get("anchor_text"):
            direction = predict.resolve_direction(
                text, m["anchor_text"], classify=classify)["direction"]
            # A contradiction is born held STRONGER than a refine/novel (PRD W6:
            # "store, held strongest, flag") — the resolver's sign, carried here.
            if direction == "contradict":
                strength = config.WRITE_CONTRADICT_HOLD
        region = _region_for(route, anchor_id, fid, memory_by_id, kept_by_fid)
        planned.append({
            "frag_id": fid, "text": text, "sent_start": s, "sent_end": e,
            "medoid_idx": medoid_of(i, s),
            "route": route, "z": m["z"], "residual": m["residual"],
            "anchor_id": anchor_id, "direction": direction, "strength": strength,
            "cluster": region, "embedding": emb})
        row = {"id": fid, "text": text, "embedding": emb, "cluster": region}
        kept_by_fid[fid] = row
        kept_rows.append(row)
        kept_idx.append(i)

    # within-note residual share — relative salience, never an absolute magnitude
    total = sum(p["residual"] for p in planned) or 1.0
    for p in planned:
        p["weight"] = round(p["residual"] / total, 4)
    # W7 — centre = the note's core (lowest residual); peak = its most novel point
    centre_id = min(planned, key=lambda p: p["residual"])["frag_id"] if planned else None
    peak_id = max(planned, key=lambda p: p["z"])["frag_id"] if planned else None
    return {"planned": planned, "reinforced": reinforced, "centre_id": centre_id,
            "novel_peak_id": peak_id, "n_intra_echo": n_intra}


def _build_payload(episode_id: str, ts: str, routed: dict, memory_size: int) -> dict:
    """The FRAGMENTED event payload — JSON-only. Neither the fragment TEXT nor its
    vector rides in the log: both are references into the immutable episode. Text is
    reconstructed from (sent_start, sent_end) and the vector from medoid_idx against
    the episode's sentences at apply time. (Legacy payloads that still carry "text"
    are honoured by apply_fragmented.)"""
    frags = [{
        "frag_id": p["frag_id"],
        "sent_start": p["sent_start"], "sent_end": p["sent_end"],
        "medoid_idx": p["medoid_idx"],
        "route": p["route"], "z": p["z"], "residual": p["residual"],
        "weight": p["weight"], "anchor_id": p["anchor_id"], "direction": p["direction"],
        "strength": p["strength"], "cluster": p.get("cluster"),
        "is_centre": p["frag_id"] == routed["centre_id"],
        "is_novel_peak": p["frag_id"] == routed["novel_peak_id"],
    } for p in routed["planned"]]
    return {"episode_id": episode_id, "ts": ts, "memory_size": memory_size,
            "fragments": frags, "reinforced": routed["reinforced"],
            "n_intra_echo": routed["n_intra_echo"]}


# ══════════════════════════════════════════════════════════════════════════════
# APPLIER — materialize a FRAGMENTED event. Deterministic; called by refine (live)
# and by consolidate.apply_event during rebuild.
#
# Idempotency is PARTIAL: the fragment-row inserts are idempotent (content-addressed
# id + ON CONFLICT DO NOTHING), but the reinforce bumps are ADDITIVE and would
# double-count if a single event were applied twice. So this must be called
# AT MOST ONCE per FRAGMENTED event. Both call sites guarantee that: refine_episode
# claims the episode atomically (mark_fragmented) before emitting, so only one
# FRAGMENTED event per episode ever exists; rebuild() truncates fragments first and
# replays each logged event exactly once.
# ══════════════════════════════════════════════════════════════════════════════
def apply_fragmented(conn, user_id: str, payload: dict) -> None:
    """Insert the routed fragments + bump the reinforced anchors. NOTHING heavy
    rides in the payload: each fragment's TEXT is reconstructed from its
    (sent_start, sent_end) span and its VECTOR is referenced from vec_sentences via
    medoid_idx — both are slices of the immutable episode, so no re-embed and no
    duplicate storage. The same path serves the live refine and rebuild() (the
    fragment rows are a deterministic function of the episode + the frozen routing).

    Backward-compat: a legacy payload that carries "text" and lacks "medoid_idx" is
    honoured — text falls back to the stored value and the medoid is re-derived from
    the span. A payload for an episode with no stored sentences inserts the row with
    a NULL medoid_idx (no referable vector); this only arises off the real pipeline."""
    p = payload
    sents = store.episode_sentences_with_vectors(conn, user_id, p["episode_id"])
    sent_texts = [s["text"] for s in sents]
    sent_embs = np.vstack([s["embedding"] for s in sents]) if sents else None

    for f in p.get("fragments", []):
        s, e = f["sent_start"], f["sent_end"]
        mi = f.get("medoid_idx")
        if sents:
            text = " ".join(sent_texts[s:e + 1])
            if mi is None:                       # legacy event: derive the medoid
                mi, _ = _fragment_medoid(sent_texts, sent_embs, s, e)
        else:                                    # no sentences → legacy text, no vector
            text = f.get("text", "")
            mi = None
        store.insert_fragment(
            conn, user_id, f["frag_id"], p["episode_id"], text,
            s, e, f["route"], f["z"], f["residual"],
            f["weight"], f["anchor_id"], f["direction"],
            bool(f["is_centre"]), bool(f["is_novel_peak"]), p["ts"],
            strength=f.get("strength", 1.0), cluster=f.get("cluster"), medoid_idx=mi)
    # reinforce AFTER inserting, so a within-note echo's sibling anchor exists
    for anchor_id in p.get("reinforced", []):
        store.bump_fragment_strength(conn, user_id, anchor_id, p["ts"],
                                     config.WRITE_REINFORCE_BUMP)


# ══════════════════════════════════════════════════════════════════════════════
# ORCHESTRATOR — DB-bound. Loads the persisted note, routes, emits + applies.
# ══════════════════════════════════════════════════════════════════════════════
def _existing_fragmented(conn, user_id: str, episode_id: str) -> bool:
    return conn.execute(
        "SELECT 1 FROM episode_fragmentations WHERE episode_id = ? AND user_id = ?",
        (episode_id, user_id)).fetchone() is not None


def refine_episode(conn, user_id: str, episode_id: str, *,
                   classify=None, calibration: dict | None = None) -> dict:
    """W2–W8 for one persisted episode. Idempotent: a note already fragmented is
    skipped (so a retry sweep or a double trigger is a no-op). No embedding round-trip
    happens at all — fragment vectors are medoid SENTENCE vectors already persisted at
    encode. The only network work is the optional LLM resolver for AMBIGUOUS fragments,
    OUTSIDE the write transaction; the event + rows + the done-marker commit atomically,
    so a failure mid-way leaves the episode unfragmented and the next sweep retries it."""
    if _existing_fragmented(conn, user_id, episode_id):
        return {"status": "skip", "reason": "already_fragmented", "episode_id": episode_id}
    ep = store.get_episode(conn, user_id, episode_id)
    if ep is None:
        return {"status": "skip", "reason": "no_episode", "episode_id": episode_id}

    sents = store.episode_sentences_with_vectors(conn, user_id, episode_id)
    if not sents:
        with conn:
            store.mark_fragmented(conn, user_id, episode_id, 0)
        return {"status": "empty", "n_fragments": 0, "episode_id": episode_id}

    calibration = calibration if calibration is not None else _load_calibration(conn, user_id)
    sent_texts = [s["text"] for s in sents]
    sent_embs = np.vstack([s["embedding"] for s in sents])

    # --- CPU work, no transaction held, NO embedding round-trip --------------
    # Fragment vectors are MEDOID sentences selected from the already-persisted
    # sentence vectors (no second HF call); `embed` is kept only for the applier's
    # last-resort fallback when an episode's sentences are unavailable.
    specs = plan_fragments(sent_texts, sent_embs, calibration=calibration)
    frag_embs, medoid_idxs = fragment_representatives(specs, sent_texts, sent_embs)
    memory = _memory_pool(conn, user_id)
    routed = route_fragments(
        specs, frag_embs, memory, classify=classify, calibration=calibration,
        frag_id_fn=lambda s, e: store.fragment_id_for(user_id, episode_id, s, e),
        medoid_idxs=medoid_idxs)
    payload = _build_payload(episode_id, ep["ts"], routed, len(memory))

    # --- commit: claim the episode FIRST (atomic race guard), then event +
    #     rows. If a concurrent trigger/sweep already fragmented it, the claim
    #     fails and we skip — no duplicate FRAGMENTED event, no double reinforce.
    #     apply re-reads the episode's sentences (text + medoid vectors) by
    #     reference — no re-embed, no cached vectors threaded through.
    with conn:
        if not store.mark_fragmented(conn, user_id, episode_id, len(payload["fragments"])):
            return {"status": "skip", "reason": "already_fragmented",
                    "episode_id": episode_id}
        store.append_event(conn, user_id, "FRAGMENTED", payload)
        apply_fragmented(conn, user_id, payload)
    return {"status": "ok", "episode_id": episode_id,
            "n_fragments": len(payload["fragments"]),
            "n_reinforced": len(payload["reinforced"]),
            "n_intra_echo": routed["n_intra_echo"], "memory_size": len(memory)}


def refine_pending(conn, user_id: str | None = None, *, max_episodes: int = 200,
                   classify=None) -> list[dict]:
    """Retry sweep: process every episode the refine pass hasn't reached yet
    (HF/LLM failures, replayed imports, or the async trigger never firing). One
    stubborn note's failure leaves it unfragmented for the NEXT sweep and never
    blocks the others. `user_id=None` sweeps all users (the nightly entry)."""
    targets = store.users_with_unfragmented(conn) if user_id is None else [user_id]
    reports = []
    for uid in targets:
        eps = store.unfragmented_episodes(conn, uid)[:max_episodes]
        refined = skipped = errors = 0
        for ep in eps:
            try:
                r = refine_episode(conn, uid, ep["id"], classify=classify)
                if r["status"] == "ok":
                    refined += 1
                else:
                    skipped += 1
            except Exception as exc:        # leave unfragmented → next sweep retries
                errors += 1
                log.warning("refine failed for %s/%s: %s", uid, ep["id"], exc)
        reports.append({"user_id": uid, "pending": len(eps), "refined": refined,
                        "skipped": skipped, "errors": errors})
    return reports


# ── trigger: best-effort, fire-and-forget off the synchronous save (W1) ───────
def trigger_refine_async(user_id: str, episode_id: str) -> None:
    """Kick off refine for one freshly-saved note in a daemon thread with its own
    connection (sqlite connections aren't shareable across threads). Best-effort:
    any failure just leaves the episode for refine_pending. No-op when
    WRITE_REFINE_ASYNC is off (tests / bulk replay, which sweep instead)."""
    if not config.WRITE_REFINE_ASYNC:
        return

    def _job():
        conn = store.connect()
        try:
            refine_episode(conn, user_id, episode_id)
        except Exception as exc:
            log.warning("async refine failed for %s/%s: %s", user_id, episode_id, exc)
        finally:
            try:
                conn.close()
            except Exception:
                pass

    threading.Thread(target=_job, daemon=True).start()


__all__ = ["plan_fragments", "route_fragments", "apply_fragmented", "refine_episode",
           "refine_pending", "trigger_refine_async", "PREDICTED", "NOVEL", "AMBIGUOUS"]
