"""Evidence lane (E4): the nightly sweep that pairs research-episode sentences with
the user's canonical claims and precomputes a stance for each pair.

See docs/evidence-lane-plan.md. The shape:

    evidence sentence --k-NN (local, free)--> claim
                      --classify_stance(sentence, claim)-->  entail | contradict | neutral

Everything expensive is kept off the read path: recall RENDERS a stance, it never
computes one. The stance call itself is free on 'nli'/'hf', and ECHO_THRESHOLD=0.72
against a MiniLM NN-cosine median ≈0.59 means most candidates never reach one.

Three properties that are load-bearing rather than optimisations:

* **Watermark.** `evidence_attachments` is derived state, truncated and rebuilt like
  `claims`, so there is NO memo row to skip a pair on. Without `evidence_sweeps.last_seq`
  the sweep would re-evaluate every (evidence sentence × claim) pair every night.
  Nightly work is instead
  `(new evidence × all claims) + (all evidence × claims new-or-re-canonicalised since last run)`.
* **Dormancy does not gate the sweep**, only recall. That is the one thing letting a
  2026 source back a 2028 claim.
* **The event carries the claim TEXT**, not just the id. Claim ids are md5 of the
  text (`consolidate.claim_id_for`), so any pinned id dangles the moment a claim is
  re-canonicalised; the text is what survives and lets `rebuild` replay.
"""
import json
import logging

from core import config, store
from core.encode import classify_stance_strict

log = logging.getLogger("slate.evidence")

# Claim ids that appeared or moved since the watermark. CANONICALIZED covers minting
# and re-canonicalisation; VERSIONED/PRUNED/EPISODE_SUPERSEDED change which claims
# exist or which one is current. These name the SECOND term of the nightly work.
CLAIM_CHANGING_EVENTS = ("CANONICALIZED", "VERSIONED", "PRUNED", "EPISODE_SUPERSEDED")


def _stance_budget_guard() -> None:
    """A per-pair-billed provider at sweep volume is the trap that drained credits
    twice before (see the prod-LLM incident notes). Refuse rather than bill.

    'openrouter' is permitted where 'haiku' is not, even though both issue one LLM
    call per pair: the openrouter provider PINS LLM_FALLBACK_ORDER to the free rung,
    so a provider failure fails instead of escalating to the paid Anthropic API.
    'haiku' keeps the full chain, which is exactly the escalation this guard exists
    to stop.
    """
    if config.STANCE_PROVIDER == "haiku":
        raise RuntimeError(
            "STANCE_PROVIDER=haiku bills per pair — the evidence sweep would issue one "
            "LLM call per (evidence sentence, claim) candidate, and its fallback chain "
            "escalates to the paid API when the free rung fails. Set STANCE_PROVIDER=nli "
            "(local), hf (Inference API), or openrouter (free rung, pinned) before sweeping.")


# Stop calling the provider after this many failures in a row. The failure that
# actually happened was OpenRouter's free-tier DAILY cap (1000 requests; observed
# 2026-07-30 with 110 consecutive 429s) — that does not clear within a run, so every
# further call is a guaranteed failure. Bail, defer, log; retry tomorrow.
STANCE_GIVE_UP_AFTER = 5


def _changed_claim_ids(conn, user_id: str, seq: int) -> list[str]:
    """Claims new-or-re-canonicalised since the watermark, that still exist.

    This is the whole point of the watermark. Without it the sweep would re-evaluate
    every (evidence sentence × claim) pair every night — `evidence_attachments` is
    truncated derived state, so there is no memo row to skip a pair on."""
    ids: list[str] = []
    seen: set[str] = set()
    for ev in store.events_since(conn, user_id, seq, types=list(CLAIM_CHANGING_EVENTS),
                                 include_rolled_back=False):
        p = json.loads(ev["payload_json"])
        for key in ("claim_id", "current_id", "other_id"):
            cid = p.get(key)
            if cid and cid not in seen:
                seen.add(cid)
                if store.get_claim(conn, user_id, cid) is not None:
                    ids.append(cid)
    return ids


def _swept_episode_ids(conn, user_id: str) -> set[str]:
    """Research episodes the sweep has already produced an event for. Read off the
    LOG, not the attachments table: an episode can legitimately have zero attachments
    (nothing in the corpus touches it yet), and it must not be re-swept every night."""
    return {json.loads(r["payload_json"])["evidence_episode_id"]
            for r in store.events_since(conn, user_id, 0, types=["EVIDENCE_ATTACHED"],
                                        include_rolled_back=False)}


def sweep(conn, user_id: str, *, k: int = 3, sent_k: int = 5,
          max_pairs: int | None = None) -> dict:
    """One nightly pass for one user. Emits EVIDENCE_ATTACHED (one full snapshot event
    per touched research episode) and returns a report. Caller commits.

    Nightly work is exactly the plan's two terms:

      1. **new evidence × current claims** — forward direction, one `knn_claims` per
         new evidence sentence.
      2. **all evidence × claims changed since the watermark** — reverse direction,
         one `knn_evidence_sentences` per changed claim. Standing evidence is never
         re-scanned wholesale.

    Idempotent: re-run with an unchanged corpus and both terms are empty, so it does
    ZERO stance calls."""
    if not config.EVIDENCE_LANE:
        return {"status": "off"}
    _stance_budget_guard()

    max_pairs = config.EVIDENCE_SWEEP_MAX_PAIRS if max_pairs is None else max_pairs
    watermark = store.evidence_watermark(conn, user_id)
    head = conn.execute("SELECT COALESCE(MAX(seq), 0) AS s FROM events WHERE user_id = ?",
                        (user_id,)).fetchone()["s"]
    ts = store.now_iso()

    all_eps = store.research_episode_ids(conn, user_id)
    if not all_eps:
        store.set_evidence_watermark(conn, user_id, head, ts)
        return {"status": "noop", "episodes": 0, "pairs": 0, "stance_calls": 0,
                "attachments": 0, "deferred": []}

    swept = _swept_episode_ids(conn, user_id)
    new_eps = [e for e in all_eps if e not in swept]
    changed = _changed_claim_ids(conn, user_id, watermark)

    budget = {"pairs": 0, "stance_calls": 0, "stance_failures": 0,
              "consecutive_failures": 0}
    deferred: list[str] = []
    # Episodes with at least one stance call that FAILED. Their snapshot must not be
    # emitted: EVIDENCE_ATTACHED is what marks an episode swept (_swept_episode_ids),
    # and a half-classified snapshot would be permanent — the pairs that failed are
    # never revisited. Deferring costs one night; persisting a wrong verdict is forever.
    poisoned: set[str] = set()
    # {episode_id: {(sentence_idx, claim_id): attachment}} — the delta, merged onto
    # each episode's existing rows before emit so the event stays a full snapshot.
    delta: dict[str, dict[tuple[int, str], dict]] = {}

    def _pair(ep_id: str, sent_idx: int, sent_text: str, claim_id: str,
              claim_text: str, sim: float, own_claims: set[str]) -> None:
        budget["pairs"] += 1
        if sim < config.ECHO_THRESHOLD or not claim_text:
            return
        # A source backing its own restatement is not evidence: claims minted FROM
        # this episode are excluded.
        if claim_id in own_claims:
            return
        if budget["consecutive_failures"] >= STANCE_GIVE_UP_AFTER:
            poisoned.add(ep_id)
            return
        budget["stance_calls"] += 1
        # premise = the source (the warrant), hypothesis = the claim on trial (E2).
        # STRICT: a swallowed failure would be stored as a real "neutral" verdict and
        # the episode marked swept, so genuine backing would read as mere adjacency
        # forever. A failure has to leave the pair unclassified instead.
        try:
            stance = classify_stance_strict(sent_text, claim_text)
        except Exception as e:  # noqa: BLE001 — per-pair; the run decides what to do
            budget["stance_failures"] += 1
            budget["consecutive_failures"] += 1
            poisoned.add(ep_id)
            log.warning("evidence sweep stance failed for %s sent %d × %s (%s: %s) — "
                        "episode deferred, not marked swept",
                        ep_id, sent_idx, claim_id, type(e).__name__, e)
            return
        budget["consecutive_failures"] = 0
        delta.setdefault(ep_id, {})[(sent_idx, claim_id)] = {
            "sentence_idx": sent_idx, "claim_id": claim_id, "claim_text": claim_text,
            "stance": stance, "similarity": round(float(sim), 4)}

    # ── Term 1: new evidence × current claims ──
    for ep_id in new_eps:
        if budget["pairs"] >= max_pairs:
            deferred.append(ep_id)
            continue
        own = set(store.claims_for_episode(conn, user_id, ep_id))
        for r in store.episode_sentences_with_vectors(conn, user_id, ep_id):
            for hit in store.knn_claims(conn, user_id, r["embedding"], k=k):
                _pair(ep_id, r["idx"], r["text"], hit["claim_id"], hit["text"],
                      hit["similarity"], own)
        delta.setdefault(ep_id, {})   # emit even when empty — records "swept, no hits"

    # ── Term 2: standing evidence × claims changed since the watermark ──
    standing = set(all_eps) - set(new_eps)
    own_cache: dict[str, set[str]] = {}
    for claim_id in changed:
        if budget["pairs"] >= max_pairs:
            deferred.append(claim_id)
            continue
        emb = store.claim_embedding(conn, user_id, claim_id)
        if emb is None:
            continue
        claim = store.get_claim(conn, user_id, claim_id)
        for hit in store.knn_evidence_sentences(conn, user_id, emb, k=sent_k):
            if hit["episode_id"] not in standing:
                continue   # a new episode already covered this pair in term 1
            own = own_cache.setdefault(
                hit["episode_id"],
                set(store.claims_for_episode(conn, user_id, hit["episode_id"])))
            _pair(hit["episode_id"], hit["idx"], hit["text"], claim_id,
                  claim["text"] if claim else "", hit["similarity"], own)

    # ── Emit one full per-episode snapshot ──
    from core.consolidate import emit
    attached = 0
    for ep_id, new_rows in delta.items():
        if ep_id in poisoned:
            continue   # emitting would mark it swept with an incomplete verdict set
        merged = {(a["sentence_idx"], a["claim_id"]): a
                  for a in store.evidence_attachments_for_episode(conn, user_id, ep_id)}
        merged.update(new_rows)
        attachments = sorted(merged.values(),
                             key=lambda a: (a["sentence_idx"], a["claim_id"]))
        emit(conn, user_id, None, "EVIDENCE_ATTACHED",
             {"evidence_episode_id": ep_id, "ts": ts, "attachments": attachments})
        attached += len(attachments)

    # A poisoned episode is deferred work, which also holds the watermark back — term 2
    # is keyed off it, so advancing past a failed night would retire those pairs too.
    deferred = deferred + sorted(poisoned - set(deferred))
    if budget["stance_failures"]:
        log.error("evidence sweep: %d of %d stance call(s) FAILED — %d episode(s) held "
                  "back for the next run. If the provider is openrouter this is most "
                  "likely the free tier's daily cap; nothing was recorded as 'neutral' "
                  "on account of it.", budget["stance_failures"], budget["stance_calls"],
                  len(poisoned))
    if deferred:
        # No silent truncation: a consolidation run that re-canonicalises many claims
        # makes that night's sweep proportionally large. What got cut is named, and the
        # watermark is NOT advanced, so the next run picks it up.
        log.warning("evidence sweep hit the %d-pair cap — %d item(s) deferred to the "
                    "next run: %s", max_pairs, len(deferred), ", ".join(deferred[:10]))
    else:
        store.set_evidence_watermark(conn, user_id, head, ts)

    return {"status": "ok", "episodes": len(delta) - len(poisoned),
            "new_evidence": len(new_eps),
            "changed_claims": len(changed), "deferred": deferred,
            "pairs": budget["pairs"], "stance_calls": budget["stance_calls"],
            "stance_failures": budget["stance_failures"], "attachments": attached}


def apply_evidence_attached(conn, user_id: str, payload: dict) -> None:
    """Materialize one EVIDENCE_ATTACHED event (called ONLY by the applier).

    Resolves each attachment by claim id first, then — if that id is gone because the
    claim was re-canonicalised — by re-deriving the id from the stored claim TEXT.
    An attachment whose claim no longer exists at all is dropped: the table is derived
    state, so a dangling row is a bug, not history."""
    from core.consolidate import claim_id_for
    ep_id = payload["evidence_episode_id"]
    ts = payload.get("ts") or store.now_iso()
    store.clear_evidence_attachments(conn, user_id, [ep_id])
    touched: set[str] = set()
    for a in payload.get("attachments", []):
        claim_id = a["claim_id"]
        if store.get_claim(conn, user_id, claim_id) is None:
            claim_id = claim_id_for(user_id, a.get("claim_text") or "")
            if store.get_claim(conn, user_id, claim_id) is None:
                continue
        store.add_evidence_attachment(conn, user_id, ep_id, a["sentence_idx"],
                                      claim_id, a["stance"], a.get("similarity"), ts)
        touched.add(claim_id)
    if touched:
        # Decay refresh: no exemption for evidence, but a new attachment IS usage.
        # last_seen only — never strength (see store.touch_claim).
        for cid in store.claims_for_episode(conn, user_id, ep_id):
            store.touch_claim(conn, user_id, cid, ts)


def sweep_all_users(conn, **kw) -> list[dict]:
    """Nightly entry — one sweep per user, one user's failure never blocking the rest."""
    if not config.EVIDENCE_LANE:
        return []
    reports = []
    for uid in store.all_user_ids(conn):
        try:
            with conn:
                report = sweep(conn, uid, **kw)
        except Exception as e:  # noqa: BLE001 — one user must not kill the sweep
            log.warning("evidence sweep failed for %s: %s", uid, e)
            report = {"status": "failed", "error": str(e)}
        reports.append({"user_id": uid, **report})
    return reports


def member_ratios(conn, user_id: str) -> list[dict]:
    """Evidence:self member ratio per concept — the volume-asymmetry watch (E3 risk).

    Medoid/anchor exclusion protects concept IDENTITY; it does not stop the membership
    MIX from shifting what a concept is made of. Evidence is cheap to add and notes
    are not, so this is logged every run rather than assumed benign."""
    return [{"concept_id": r["concept_id"], "label": r["label"],
             "n_self": r["n_self"], "n_evidence": r["n_evidence"],
             "ratio": round(r["n_evidence"] / r["n_self"], 2) if r["n_self"] else None}
            for r in conn.execute(
                """SELECT c.id AS concept_id, c.label,
                          SUM(CASE WHEN cm.kind = 'evidence' THEN 0 ELSE 1 END) AS n_self,
                          SUM(CASE WHEN cm.kind = 'evidence' THEN 1 ELSE 0 END) AS n_evidence
                   FROM concepts c JOIN concept_members cm
                     ON cm.concept_id = c.id AND cm.user_id = c.user_id
                   WHERE c.user_id = ? GROUP BY c.id
                   HAVING n_evidence > 0 ORDER BY n_evidence DESC""", (user_id,))]


__all__ = ["sweep", "sweep_all_users", "apply_evidence_attached", "member_ratios"]
