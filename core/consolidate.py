"""Nightly sleep phase: blueprint extraction, claim canonicalization, concept merge/split/create, latent bridges, decay/strengthen. Sole writer to the semantic store, only via events. Batch API. See PLAN.md §5.

Every decision is emitted as an event and then applied by the matching
applier in this module; `rebuild()` truncates the semantic tables and
re-applies the whole log through the same appliers, so the two can never
drift. Appliers must stay deterministic: anything non-deterministic
(LLM output, generated ids, timestamps, decayed states) is decided BEFORE
emit and carried inside the event payload. Embeddings are recomputed from
text at apply time — same local model, same vectors.
"""
import hashlib
import json
from datetime import datetime, timezone

from core import config, guard, llm, predict, store
from core.encode import get_embedder, split_sentences

# Strength deltas (spaced-repetition pressure)
SUPPORT_BUMP = 0.5   # claim re-encountered via a new episode at canonicalization
ECHO_BUMP = 0.25     # claim echoed in an encode-time receipt

# Canonicalization similarity bands — COLD-START FALLBACK ONLY (C2). The populated
# path now routes through the predictor spine (measure/decide, spread-relative);
# these absolute-cosine cuts are used only when a new claim's neighbourhood is too
# small to estimate a region spread (< WARMUP_MIN_CORPUS), the same "trust the
# prior/absolute rule when cold" stance as C11.
CANON_AUTO_SAME = 0.92   # >= : same claim, no LLM needed
CANON_LLM_BAND = 0.75    # [band, auto) : ask the LLM; below: new claim

# C2 dedup calibration: the predictor route over a new claim vs its neighbourhood
# of existing canonical claims. PREDICTED→same (echo of an existing claim),
# AMBIGUOUS→uncertain (LLM splits paraphrase from a genuinely distinct/contra
# claim), NOVEL→new. Spread-relative + scopeable, fitted at C12 and pushed down —
# never a baked similarity constant.
#
# DEDUP_Z_ECHO is calibrated to the claim-dedup z scale, which differs sharply
# from the fragment-write scale: a claim's neighbourhood at canonicalization is a
# very TIGHT cluster of near-duplicates, so its σ is tiny and z is near-binary —
# an exact/true duplicate sits at z≈0 while anything genuinely distinct jumps to
# z≳5 (measured on the live corpus, a clean gap in between). So the auto-"same"
# cut is a small POSITIVE z (reconstructs about as tightly as the region's own
# members), not the Write path's strongly-negative Z_ECHO. The wide gap makes the
# exact value robust anywhere in ~[0.5, 4.5]; the rest escalates to the LLM.
DEDUP_Z_ECHO = 1.0
DEDUP_CALIBRATION = {"z_echo": DEDUP_Z_ECHO, "prox_margin": predict.PROX_MARGIN,
                     "per_cluster": {}}

# Bridge candidate band — C5 now a medoid-vs-region RESIDUAL band (predictor
# spine), not a centroid cosine: close enough to relate (residual not too high),
# enough residual that the link is non-obvious (not a near-duplicate concept).
BRIDGE_RES_LOW, BRIDGE_RES_HIGH = 0.40, 0.85
BRIDGE_MAX_VERIFY = 5

CANON_CHUNK = 20         # uncertain pairs per LLM call
CONCEPT_CHUNK = 30       # new claims per concept-pass call (keeps JSON within budget)
CONCEPT_CONTEXT_MEMBERS = 12

# ── Prompts ───────────────────────────────────────────────────────────────────
# Ported from v1 engine/extract.py (best-tested asset — keep wording stable).
PROMPT_BLUEPRINT = """Extract semantic structure. Return ONLY valid JSON, no other text.

{
  "title": "3-6 word title for this note",
  "essence": "one sentence capturing the entire core argument",
  "clusters": [
    {
      "label": "2-4 words",
      "kernel": "one sentence capturing the core idea of this cluster",
      "claims": ["distilled assertion 1", "distilled assertion 2"],
      "representative_sentences": ["verbatim sentence matching each claim"]
    }
  ],
  "assumptions": ["implicit premise the argument relies on"],
  "spine": [
    {"from": "label_of_cluster_A", "to": "label_of_cluster_B", "relation": "short phrase describing how A leads to or connects with B"}
  ]
}

Rules:
- claims and representative_sentences must be parallel arrays (same length)
- minimum clusters needed, no redundancy
- representative_sentences are verbatim from the text
- claims are distilled — strip rhetoric, preserve logic
- spine links must use the exact cluster labels from the clusters array
- spine relation is a short free-form phrase (2-5 words) capturing how A connects to B — e.g. "provides evidence for", "is challenged by", "extends into", "is prerequisite for", "contrasts with", "leads to"
- include all meaningful inter-cluster relationships, not just sequential order

TEXT:
"""

PROMPT_CANON = """You deduplicate a personal knowledge base. For each numbered pair below, decide whether the NEW claim asserts the same thing as the EXISTING canonical claim (same = true) or is a genuinely distinct assertion (same = false). Paraphrase, narrower/broader phrasing of the same point, or restating with different emphasis → same. Different subject, different mechanism, opposite stance, or a new qualification that changes the meaning → not same.

Return ONLY valid JSON: {"verdicts": [{"i": <pair number>, "same": true|false}, ...]}

PAIRS:
"""

PROMPT_CONCEPT = """You maintain the concept layer of a personal knowledge base during nightly consolidation. Below are NEW CLAIMS distilled from recent notes, and the EXISTING CONCEPTS nearest to them (with member claims).

Decide how the concept layer should change. Return ONLY valid JSON:
{"decisions": [
  {"action": "CREATE", "label": "2-4 words", "canonical": "one-sentence description", "claim_ids": ["..."]},
  {"action": "ATTACH", "concept_id": "...", "claim_ids": ["..."]},
  {"action": "MERGE", "winner_id": "...", "loser_id": "...", "label": "optional new label", "canonical": "optional new canonical"},
  {"action": "SPLIT", "concept_id": "...", "into": [{"label": "...", "canonical": "...", "claim_ids": ["..."]}, {"label": "...", "canonical": "...", "claim_ids": ["..."]}]}
]}

Rules:
- Every new claim should land in exactly one concept (CREATE or ATTACH). Orphan claims are allowed only if truly standalone.
- CREATE only when no existing concept fits; prefer ATTACH.
- MERGE two concepts only when their member claims are about the same idea — not merely related ideas.
- SPLIT a concept only when its members clearly cover two distinct ideas that deserve separate concepts; every member claim_id must be assigned to exactly one side.
- Use only claim_ids and concept_ids that appear below. Return an empty decisions list if nothing should change.

NEW CLAIMS:
{new_claims}

EXISTING CONCEPTS (nearest first):
{concepts}
"""

PROMPT_BRIDGE = """Two concepts from a personal knowledge base drifted near each other in meaning, but no note connects them yet. Decide whether there is a real, non-obvious intellectual connection worth surfacing to the author.

Concept A: {a_label} — {a_canonical}
Sample claims: {a_claims}

Concept B: {b_label} — {b_canonical}
Sample claims: {b_claims}

Return ONLY valid JSON: {{"bridge": true|false, "rationale": "one sentence naming the connection (empty if false)"}}
"""


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def claim_id_for(user_id: str, text: str) -> str:
    """User-salted so identical text from two users mints distinct ids — vec0
    PRIMARY KEYs are globally unique across partitions (AUTH.md §1)."""
    raw = f"{user_id}\x00{text.strip().lower()}".encode("utf-8")
    return "clm_" + hashlib.md5(raw).hexdigest()


# ── Event emit + apply (the backbone) ─────────────────────────────────────────
def emit(conn, user_id: str, run_id: str | None, type_: str, payload: dict) -> None:
    """Append the event, then materialize it. Decision → event → row, always."""
    store.append_event(conn, user_id, type_, payload, run_id=run_id)
    apply_event(conn, user_id, type_, payload)


def apply_event(conn, user_id: str, type_: str, payload: dict) -> None:
    """Materialize one event into the semantic tables. Deterministic."""
    p = payload
    if type_ == "CANONICALIZED":
        if p["action"] == "new":
            emb = get_embedder().encode([p["text"]], normalize_embeddings=True,
                                        show_progress_bar=False)[0]
            store.insert_claim(conn, user_id, p["claim_id"], p["text"], emb, p["ts"])
        else:  # support: re-encounter of an existing claim
            store.bump_claim_strength(conn, user_id, p["claim_id"], p["ts"], SUPPORT_BUMP)
        store.add_claim_support(conn, user_id, p["claim_id"], p["episode_id"], p.get("verbatim"))

    elif type_ == "CONCEPT_CREATED":
        store.insert_concept(conn, user_id, p["concept_id"], p["label"], p["canonical"], p["ts"])
        for cid in p["claim_ids"]:
            store.add_concept_member(conn, user_id, p["concept_id"], cid)
        store.recompute_concept_embedding(conn, user_id, p["concept_id"])

    elif type_ == "ATTACHED":
        for cid in p["claim_ids"]:
            store.add_concept_member(conn, user_id, p["concept_id"], cid)
        store.update_concept(conn, user_id, p["concept_id"], last_activity=p["ts"])
        store.recompute_concept_embedding(conn, user_id, p["concept_id"])

    elif type_ == "MERGED":
        # C6 nuance guard: fold only the members the winner reconstructs; members
        # the guard kept (fold/kept partition frozen in the payload for replay)
        # stay in the loser, which therefore survives. Legacy events without the
        # partition fold everything (fold_claim_ids defaults to all loser members).
        all_members = p["loser_snapshot"]["member_claim_ids"]
        fold_ids = p.get("fold_claim_ids", all_members)
        kept_ids = p.get("kept_claim_ids", [])
        for cid in fold_ids:
            store.add_concept_member(conn, user_id, p["winner_id"], cid)
        if kept_ids:  # nuance remains → loser is not emptied, only the folded leave
            for cid in fold_ids:
                store.remove_concept_member(conn, user_id, p["loser_id"], cid)
            store.update_concept(conn, user_id, p["loser_id"], last_activity=p["ts"])
            store.recompute_concept_embedding(conn, user_id, p["loser_id"])
        else:
            store.delete_concept(conn, user_id, p["loser_id"])
        store.update_concept(conn, user_id, p["winner_id"], label=p.get("label"),
                             canonical=p.get("canonical"), last_activity=p["ts"])
        store.recompute_concept_embedding(conn, user_id, p["winner_id"])

    elif type_ == "SPLIT":
        store.delete_concept(conn, user_id, p["concept_id"])
        for child in p["into"]:
            store.insert_concept(conn, user_id, child["concept_id"], child["label"],
                                 child["canonical"], p["ts"])
            for cid in child["claim_ids"]:
                store.add_concept_member(conn, user_id, child["concept_id"], cid)
            store.recompute_concept_embedding(conn, user_id, child["concept_id"])

    elif type_ == "RELATED":
        store.insert_relation(conn, user_id, p["from_id"], p["to_id"], p["relation"],
                              p.get("weight", 1.0), p["ts"],
                              p.get("evidence_episode_id"))

    elif type_ == "BRIDGED":
        store.insert_relation(conn, user_id, p["a"], p["b"], "bridges",
                              p.get("score", 1.0), p["ts"],
                              p.get("evidence_episode_id"))

    elif type_ == "STRENGTHENED":
        if p.get("claim_id"):
            store.bump_claim_strength(conn, user_id, p["claim_id"], p["ts"], p["delta"])
        if p.get("concept_id"):
            c = store.get_concept(conn, user_id, p["concept_id"])
            if c:
                store.update_concept(conn, user_id, p["concept_id"],
                                     strength=c["strength"] + p["delta"],
                                     last_activity=p["ts"])

    elif type_ == "DECAYED":
        store.update_concept(conn, user_id, p["concept_id"], state=p["state_to"])

    elif type_ == "BACKGROUNDED":
        # C9 — the claim has been re-predicted enough to be background; its theme
        # now stands for it, so its standalone retrieval pull is demoted.
        store.set_claim_background(conn, user_id, p["claim_id"], 1)

    elif type_ == "PRUNED":
        # C7 safe-forget: the guard confirmed the surviving members reconstruct
        # this claim, so dropping it loses no nuance. Remove it from the concept;
        # if no concept still holds it, drop it from working memory (re-derivable
        # from raw — that is what makes the prune safe).
        store.remove_concept_member(conn, user_id, p["concept_id"], p["claim_id"])
        if not store.claim_in_any_concept(conn, user_id, p["claim_id"]):
            store.delete_claim(conn, user_id, p["claim_id"])
        if p.get("concept_id"):
            store.recompute_concept_embedding(conn, user_id, p["concept_id"])

    elif type_ == "VERSIONED":
        # C8 — apply a reconciliation decided (with the flip margin) before emit.
        # Both rivals join one version_group; the current view is surfaced as
        # contested at retrieval, the loser is never silently dropped.
        grp = p["version_group"]
        store.set_claim_version(conn, user_id, p["current_id"],
                                status="current", version_group=grp,
                                qualifier=p.get("qualifier_current"))
        if p["mode"] == "supersede":
            store.set_claim_version(conn, user_id, p["other_id"],
                                    status="superseded", superseded_by=p["current_id"],
                                    version_group=grp)
        elif p["mode"] == "scope":  # both true under different conditions
            store.set_claim_version(conn, user_id, p["other_id"],
                                    status="current", version_group=grp,
                                    qualifier=p.get("qualifier_other"))
        else:  # "version": both stand; other is held
            store.set_claim_version(conn, user_id, p["other_id"],
                                    status="version", version_group=grp)

    elif type_ == "FRAGMENTED":
        # Write-side working memory. Decided by the async refine pass (core/write);
        # the applier here is what lets rebuild() re-derive fragments from the log
        # (embeddings recomputed from the fragment text, like CANONICALIZED claims).
        from core import write
        write.apply_fragmented(conn, user_id, p)

    # ENCODED / BLUEPRINTED / INTEGRITY_FLAGGED: episodic-side or log-only —
    # nothing to materialize (INTEGRITY_FLAGGED is a review signal, not state).


def rebuild(conn) -> dict:
    """Truncate the semantic store (all users) and re-apply the event log, each
    event under the user that emitted it. Rolled-back runs are skipped, so this
    is also the re-materialization path after a rollback. Admin path."""
    with conn:
        store.truncate_semantic(conn)
        events = store.events_since(conn, None, 0, include_rolled_back=False)
        for ev in events:
            apply_event(conn, ev["user_id"], ev["type"], json.loads(ev["payload_json"]))
    return {"events_applied": len(events)}


def rollback_run(conn, run_id: str) -> dict:
    """Reverse one consolidation run (PRD §Consolidation: "bad runs must be
    undoable"). The run's events are flagged rolled-back (kept on disk for
    audit), its episodes are made re-eligible, and the semantic store is
    re-materialized from the remaining active log.

    This also defeats a *poisoned* log: a re-run after rollback re-derives the
    freed episodes from their immutable raw text — the rolled-back BLUEPRINTED/
    CANONICALIZED events are invisible to the `_existing_*` guards (they filter
    on the same active-run predicate), so nothing stale is reused."""
    run = store.get_run(conn, run_id)
    if run is None:
        return {"status": "unknown_run", "run_id": run_id}
    with conn:
        store.mark_run_rolled_back(conn, run_id)
        freed = store.unconsolidate_run_episodes(conn, run_id)
    res = rebuild(conn)
    return {"status": "rolled_back", "run_id": run_id,
            "episodes_freed": freed, "events_applied": res["events_applied"]}


# ── Step 2: blueprint extraction ──────────────────────────────────────────────
def _blueprint_local(text: str) -> dict:
    """KMeans fallback (ported from v1 extract_local): clusters + kernels, no spine."""
    import numpy as np
    from sklearn.cluster import KMeans
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.preprocessing import normalize

    sentences = split_sentences(text)
    if len(sentences) <= 1:
        s = sentences[0] if sentences else text[:300]
        title = " ".join(s.split()[:6]).rstrip(".,;:?!")
        return {"title": title, "essence": s,
                "clusters": [{"label": "main", "kernel": s, "claims": [s],
                              "representative_sentences": [s]}],
                "assumptions": [], "spine": [], "_method": "local"}

    emb = normalize(get_embedder().encode(sentences, show_progress_bar=False))
    n = max(2, min(6, int(np.sqrt(len(sentences)))))
    n = min(n, len(sentences))
    km = KMeans(n_clusters=n, random_state=42, n_init=10)
    labels = km.fit_predict(emb)
    centroids = normalize(km.cluster_centers_)

    clusters = []
    for idx in range(n):
        members = [i for i, l in enumerate(labels) if l == idx]
        if not members:
            continue
        sims = cosine_similarity(emb[members], centroids[idx].reshape(1, -1)).flatten()
        kernel = sentences[members[int(np.argmax(sims))]]
        clusters.append({"label": " ".join(kernel.split()[:4]).rstrip(".,;:") + "…",
                         "kernel": kernel, "claims": [kernel],
                         "representative_sentences": [kernel]})

    all_sims = cosine_similarity(emb, emb)
    essence = sentences[int(np.argmax((all_sims.sum(axis=1) - 1) / max(1, len(sentences) - 1)))]
    return {"title": " ".join(essence.split()[:6]).rstrip(".,;:?!"), "essence": essence,
            "clusters": clusters, "assumptions": [], "spine": [], "_method": "local"}


def blueprint(text: str) -> tuple[dict, float]:
    """LLM blueprint; local KMeans fallback only where sklearn exists (dev).
    On the slim server image the LLMError propagates instead — the caller
    skips the episode and the next nightly run retries it."""
    try:
        result = llm.call(PROMPT_BLUEPRINT + text, tier="mechanical", max_tokens=2048)
        bp = result["json"]
        bp["_method"] = result["provider"]
        return bp, result["cost"]
    except llm.LLMError:
        try:
            return _blueprint_local(text), 0.0
        except ImportError:
            raise llm.LLMError("blueprint failed and no local sklearn fallback")


# ── Retry reuse: a failed run's blueprint + canon events are authoritative ────
# Blueprint extraction is nondeterministic, so re-extracting on retry mints
# near-duplicate claims. An episode's BLUEPRINTED event and its CANONICALIZED
# events are written in one transaction, so either both exist fully or neither.
def _existing_blueprint(conn, user_id: str, episode_id: str) -> dict | None:
    # Active runs only — a rolled-back run's blueprint must not short-circuit
    # re-derivation from raw (the poisoned-log guard; see rollback_run).
    row = conn.execute(
        f"""SELECT payload_json FROM events WHERE type = 'BLUEPRINTED'
           AND user_id = ?
           AND json_extract(payload_json, '$.episode_id') = ?
           AND {store.ACTIVE_RUN_PREDICATE}
           ORDER BY seq DESC LIMIT 1""", (user_id, episode_id)).fetchone()
    return json.loads(row["payload_json"])["blueprint"] if row else None


def _existing_canon(conn, user_id: str, episode_id: str) -> list[tuple[str, str]]:
    rows = conn.execute(
        f"""SELECT payload_json FROM events WHERE type = 'CANONICALIZED'
           AND user_id = ?
           AND json_extract(payload_json, '$.episode_id') = ?
           AND {store.ACTIVE_RUN_PREDICATE} ORDER BY seq""",
        (user_id, episode_id)).fetchall()
    out = []
    for r in rows:
        p = json.loads(r["payload_json"])
        out.append((p.get("cluster", ""), p["claim_id"]))
    return out


# ── Step 3: claim canonicalization ────────────────────────────────────────────
def _anchor_concept_prior(conn, user_id: str, episode_id: str | None, claim_emb) -> str | None:
    """Fragment→consolidate PRIOR (plan §3): the concept the claim's Write-time source
    span anchored to. Bridges claim → nearest same-episode fragment (cosine over the
    medoid vectors Write already built — NO new embedding) → that fragment's AMBIGUOUS
    `anchor_id` (a prior memory fragment) → the anchor's episode → its claims → their
    concept.

    Raw stays source-of-truth: this is a HINT the predictor (C2) / LLM (C3) may use, not
    a decision. Returns None whenever fragments are absent (Write async / no refine yet)
    or the source span was NOVEL/unanchored — so a corpus without materialized fragments
    behaves EXACTLY as before. Never a network call (all reads are stored vectors)."""
    if not episode_id:
        return None
    frags = store.episode_fragments(conn, user_id, episode_id)
    if not frags:
        return None
    import numpy as np
    q = np.asarray(claim_emb, dtype=np.float32)
    q = q / (float(np.linalg.norm(q)) or 1.0)
    best = max(frags, key=lambda f: float(q @ f["embedding"]))
    anchor = best.get("anchor_id")
    if not anchor:                                  # NOVEL span → no anchored concept
        return None
    anchor_ep = store.fragment_episode(conn, user_id, anchor)
    if not anchor_ep:
        return None
    from collections import Counter
    counts = Counter(
        cid for cl in store.claims_for_episode(conn, user_id, anchor_ep)
        if (cid := _claim_concept(conn, user_id, cl)))
    return counts.most_common(1)[0][0] if counts else None


def _dedup_route(conn, user_id: str, raw_claims: list[dict], embs,
                 calibration: dict, episode_id: str | None = None) -> tuple[list[tuple], list[tuple]]:
    """C2 — route each new claim same / uncertain / new against its neighbourhood
    of existing canonical claims, via the predictor spine (measure → decide),
    replacing the raw-cosine CANON_AUTO_SAME/CANON_LLM_BAND cuts.

    For each new claim we gather its STAT_K nearest existing claims (their concept
    ids become the `cluster` field, so the z is measured against the right region's
    self-cohesion) and route:
      PREDICTED → same   (echo: reconstructs at least as tightly as the region)
      AMBIGUOUS → LLM    (attached but looser — paraphrase vs distinct/contra)
      NOVEL     → new
    A neighbourhood too small to estimate a spread falls back to the absolute
    cosine bands (the cold-start stance of C11). Returns (decided, uncertain),
    same shape the caller consumed before."""
    import numpy as np
    decided, uncertain = [], []
    for c, emb in zip(raw_claims, embs):
        hits = store.knn_claims(conn, user_id, emb, k=predict.STAT_K)
        # C2 fragment prior: the source span's anchored concept may hold a canonical
        # claim that global knn under-ranked (the LLM paraphrase drifted from the
        # verbatim span). Widen the dedup neighbourhood with that concept's members —
        # the predictor still routes below; an absent prior leaves this loop unchanged.
        prior_cid = _anchor_concept_prior(conn, user_id, episode_id, emb)
        seen = {h["claim_id"] for h in hits}
        extra = ([cid for cid in store.concept_member_ids(conn, user_id, prior_cid)
                  if cid not in seen] if prior_cid else [])
        if not hits and not extra:
            decided.append((c, "new", None))
            continue
        qn = np.asarray(emb, dtype=np.float32)
        qn = qn / (float(np.linalg.norm(qn)) or 1.0)
        cands = [{"claim_id": h["claim_id"], "text": h["text"],
                  "similarity": h["similarity"]} for h in hits]
        corpus = []
        for h in hits:
            he = store.claim_embedding(conn, user_id, h["claim_id"])
            if he is not None:
                corpus.append({"id": h["claim_id"], "text": h["text"],
                               "embedding": he,
                               "cluster": _claim_concept(conn, user_id, h["claim_id"])})
        for cid in extra:                              # prior concept's members
            he = store.claim_embedding(conn, user_id, cid)
            if he is None:
                continue
            row = store.get_claim(conn, user_id, cid) or {"text": ""}
            corpus.append({"id": cid, "text": row["text"], "embedding": he,
                           "cluster": _claim_concept(conn, user_id, cid)})
            cands.append({"claim_id": cid, "text": row["text"],
                          "similarity": float(qn @ np.asarray(he, dtype=np.float32))})
        best = max(cands, key=lambda x: x["similarity"])
        # A spread-relative z is only meaningful over a populated neighbourhood:
        # at canonicalization claims carry no concept yet, so the per-cluster
        # baselines are empty and the region spread comes from `corpus_prior`,
        # which returns a real residual distribution only above SPAN_K members
        # (below that it yields a generic default that makes z meaningless). So
        # route through the spine only with > SPAN_K neighbours; a thinner region
        # is "cold" and trusts the absolute-cosine rule (the C11 stance).
        if len(corpus) <= predict.SPAN_K:              # cold region → absolute rule
            if best["similarity"] >= CANON_AUTO_SAME:
                decided.append((c, "same", best["claim_id"]))
            elif best["similarity"] >= CANON_LLM_BAND:
                uncertain.append((c, best))
            else:
                decided.append((c, "new", None))
            continue
        m = predict.measure([{"text": c["text"], "embedding": emb}], corpus)[0]
        route = predict.decide([m], calibration)["fragments"][0]["route"]
        if route == predict._PREDICTED and m["anchor_id"]:
            decided.append((c, "same", m["anchor_id"]))
        elif route == predict._AMBIGUOUS and m["anchor_id"]:
            uncertain.append((c, {"claim_id": m["anchor_id"],
                                  "text": m["anchor_text"]}))
        else:
            decided.append((c, "new", None))
    return decided, uncertain


def _canonicalize_episode(conn, user_id: str, run_id: str, episode, bp: dict,
                          bp_payload: dict) -> tuple[list[str], float]:
    """Dedupe each blueprint claim against existing canonical claims.

    All reads + LLM judging happen OUTSIDE any transaction (a save_note must
    never wait on a network call); the BLUEPRINTED event and all CANONICALIZED
    events then commit in one short transaction — atomic for retry reuse.
    Returns (claim ids touched by this episode, llm cost)."""
    ts = episode["ts"]
    raw_claims = []
    for cluster in bp.get("clusters", []):
        reps = cluster.get("representative_sentences", [])
        for i, ctext in enumerate(cluster.get("claims", [])):
            if not ctext or not ctext.strip():
                continue
            raw_claims.append({"text": ctext.strip(),
                               "verbatim": reps[i] if i < len(reps) else None,
                               "cluster": cluster.get("label", "")})
    if not raw_claims:
        return [], 0.0

    embs = get_embedder().encode([c["text"] for c in raw_claims],
                                 normalize_embeddings=True, show_progress_bar=False)

    decided, uncertain = _dedup_route(conn, user_id, raw_claims, embs,
                                      DEDUP_CALIBRATION, episode_id=episode["id"])

    cost = 0.0
    for start in range(0, len(uncertain), CANON_CHUNK):
        chunk = uncertain[start:start + CANON_CHUNK]
        pairs = "\n".join(
            f'{i}. NEW: "{c["text"]}"\n   EXISTING: "{best["text"]}"'
            for i, (c, best) in enumerate(chunk))
        try:
            result = llm.call(PROMPT_CANON + pairs, tier="mechanical", max_tokens=1024)
            cost += result["cost"]
            verdicts = {v["i"]: v["same"] for v in result["json"].get("verdicts", [])}
        except llm.LLMError:
            verdicts = {}  # treat all as new — md5 ids keep retries idempotent
        for i, (c, best) in enumerate(chunk):
            decided.append((c, "same", best["claim_id"]) if verdicts.get(i)
                           else (c, "new", None))

    episode_claims = []
    with conn:  # short txn: blueprint + canon events commit atomically
        store.append_event(conn, user_id, "BLUEPRINTED", bp_payload, run_id=run_id)
        for c, action, existing_id in decided:
            if action == "same":
                emit(conn, user_id, run_id, "CANONICALIZED", {
                    "action": "support", "claim_id": existing_id,
                    "episode_id": episode["id"], "verbatim": c["verbatim"],
                    "cluster": c["cluster"], "ts": ts})
                episode_claims.append((c["cluster"], existing_id))
            else:
                cid = claim_id_for(user_id, c["text"])
                emit(conn, user_id, run_id, "CANONICALIZED", {
                    "action": "new", "claim_id": cid, "text": c["text"],
                    "episode_id": episode["id"], "verbatim": c["verbatim"],
                    "cluster": c["cluster"], "ts": ts})
                episode_claims.append((c["cluster"], cid))
    return episode_claims, cost


# ── C4: concept spread test (split candidacy + medoid re-anchor) ──────────────
CONCEPT_BIMODAL_MARGIN = 0.15   # within-blob cohesion must beat the cross-blob
                                # gap by this for a concept to read as bimodal


def _spread_is_bimodal(V, margin: float = CONCEPT_BIMODAL_MARGIN) -> bool:
    """measure()-style spread test: do the member vectors fall into TWO separated
    blobs? Take the least-similar member pair as poles, assign each member to its
    nearer pole, and call it bimodal when each side is internally tighter than the
    cross-side similarity by `margin` (two ideas wearing one concept → SPLIT
    candidate). Pure numpy (no sklearn — prod-safe), deterministic."""
    import numpy as np
    n = V.shape[0]
    if n < 4:                       # too few members to claim two ideas
        return False
    G = V @ V.T
    i, j = divmod(int(np.argmin(G)), n)     # the two most-dissimilar members
    side = G[i] >= G[j]                      # nearer to pole i?
    a, b = V[side], V[~side]
    if a.shape[0] < 2 or b.shape[0] < 2:
        return False
    within = 0.5 * (float((a @ a.T).mean()) + float((b @ b.T).mean()))
    between = float((a @ b.T).mean())
    return (within - between) > margin


def _concept_geometry(conn, user_id: str, concept_id: str) -> dict:
    """C4 geometry of a concept: its medoid member text (the re-anchored core) and
    a coarse shape ('bimodal' → SPLIT candidate, else 'cohesive'). Empty concept →
    neutral defaults."""
    import numpy as np
    rows = _members_with_emb(conn, user_id,
                             store.concept_member_ids(conn, user_id, concept_id))
    if not rows:
        return {"representative": "", "shape": "cohesive"}
    V = np.vstack([r["embedding"] for r in rows])
    med = _medoid_vec(V)
    rep = next((r["text"] for r in rows if np.array_equal(r["embedding"], med)),
               rows[0]["text"])
    return {"representative": rep,
            "shape": "bimodal" if _spread_is_bimodal(V) else "cohesive"}


# ── Step 4: concept pass (the judgment call) ──────────────────────────────────
def _concept_pass(conn, user_id: str, run_id: str, new_claim_ids: list[str],
                  ts: str) -> float:
    """Chunked so each judgment call stays within output budget; later chunks
    see concepts created by earlier ones, so attachment stays incremental."""
    unique_ids = list(dict.fromkeys(new_claim_ids))
    cost = 0.0
    for start in range(0, len(unique_ids), CONCEPT_CHUNK):
        cost += _concept_pass_chunk(conn, user_id, run_id,
                                    unique_ids[start:start + CONCEPT_CHUNK], ts)
    return cost


# C3: a claim is a plausible member of a concept when it reconstructs against that
# concept's members about as well as the members do themselves — i.e. its residual
# z (measured vs the concept as its own cluster) sits within the region's spread.
# Spread-relative, replacing the flat `similarity >= 0.40` floor. A concept too
# thin to estimate a spread (<= SPAN_K members) is "cold" → absolute-cosine rule.
CONCEPT_MEMBERSHIP_Z = 2.0      # within ~2σ of the concept's own cohesion
CONCEPT_MEMBERSHIP_SIM = 0.40   # cold-concept cosine fallback (legacy floor)


def _membership_z(conn, user_id: str, emb, claim_text: str,
                  concept_id: str) -> float | None:
    """C3 — the claim's residual z against `concept_id`'s members as their own
    cluster (measure() nearest-cluster). None when the concept is too thin for a
    spread estimate (caller falls back to cosine)."""
    rows = _members_with_emb(conn, user_id,
                             store.concept_member_ids(conn, user_id, concept_id))
    if len(rows) <= predict.SPAN_K:
        return None
    corpus = [{**r, "cluster": concept_id} for r in rows]
    return predict.measure([{"text": claim_text, "embedding": emb}], corpus)[0]["z"]


def _concept_pass_chunk(conn, user_id: str, run_id: str, new_claim_ids: list[str],
                        ts: str) -> float:
    if not new_claim_ids:
        return 0.0

    new_claims = []
    nearby: dict[str, dict] = {}
    for cid in new_claim_ids:
        row = store.get_claim(conn, user_id, cid)
        if not row:
            continue
        new_claims.append({"id": cid, "text": row["text"]})
        emb = store.claim_embedding(conn, user_id, cid)
        if emb is not None:
            for hit in store.knn_concepts(conn, user_id, emb, k=3):
                z = _membership_z(conn, user_id, emb, row["text"], hit["id"])
                plausible = (z <= CONCEPT_MEMBERSHIP_Z) if z is not None \
                    else (hit["similarity"] >= CONCEPT_MEMBERSHIP_SIM)
                if plausible:
                    nearby[hit["id"]] = hit
            # C3 fragment prior: surface the concept the claim's source span anchored
            # to as an attach candidate, even when global knn under-ranked it — this
            # is what cuts concept fragmentation. The LLM below still attaches/splits;
            # an absent prior (no fragments / NOVEL span) adds nothing.
            for ep in store.claim_source_episodes(conn, user_id, cid):
                pc = _anchor_concept_prior(conn, user_id, ep, emb)
                if pc and pc not in nearby:
                    crow = store.get_concept(conn, user_id, pc)
                    if crow:
                        nearby[pc] = {"id": pc, "label": crow["label"],
                                      "canonical": crow["canonical"], "similarity": 0.0}

    concepts_ctx = []
    for c in nearby.values():
        member_ids = store.concept_member_ids(conn, user_id, c["id"])[:CONCEPT_CONTEXT_MEMBERS]
        members = [{"id": m, "text": (store.get_claim(conn, user_id, m) or {"text": ""})["text"]}
                   for m in member_ids]
        # C4 spread test: a measure()-based geometric read of the concept, surfaced
        # to steer the judgment call — `representative` re-anchors the concept to
        # its medoid core (the most central member, robust to a drifted mean), and
        # `geometry: bimodal` flags a concept whose members fall into two separated
        # blobs (a SPLIT candidate). The LLM still decides; geometry only informs.
        geom = _concept_geometry(conn, user_id, c["id"])
        concepts_ctx.append({"id": c["id"], "label": c["label"],
                             "canonical": c["canonical"],
                             "representative": geom["representative"],
                             "geometry": geom["shape"], "members": members})

    prompt = PROMPT_CONCEPT.replace(
        "{new_claims}", json.dumps(new_claims, ensure_ascii=False, indent=1)).replace(
        "{concepts}", json.dumps(concepts_ctx, ensure_ascii=False, indent=1) or "[]")
    result = llm.call(prompt, tier="judgment", max_tokens=8192)  # outside any txn
    decisions = result["json"].get("decisions", [])

    valid_claims = {c["id"] for c in new_claims}
    for concept in concepts_ctx:
        valid_claims.update(m["id"] for m in concept["members"])
    valid_concepts = set(nearby.keys())

    with conn:
        _apply_concept_decisions(conn, user_id, run_id, decisions, valid_claims,
                                 valid_concepts, ts)
    return result["cost"]


def _apply_concept_decisions(conn, user_id, run_id, decisions, valid_claims,
                             valid_concepts, ts) -> None:
    for d in decisions:
        action = d.get("action", "").upper()
        if action == "CREATE":
            claim_ids = [c for c in d.get("claim_ids", []) if c in valid_claims]
            if not claim_ids:
                continue
            emit(conn, user_id, run_id, "CONCEPT_CREATED", {
                "concept_id": "cpt_" + store.ulid(), "label": d.get("label", ""),
                "canonical": d.get("canonical", ""), "claim_ids": claim_ids, "ts": ts})
        elif action == "ATTACH" and d.get("concept_id") in valid_concepts:
            claim_ids = [c for c in d.get("claim_ids", []) if c in valid_claims]
            if claim_ids:
                emit(conn, user_id, run_id, "ATTACHED", {
                    "concept_id": d["concept_id"], "claim_ids": claim_ids, "ts": ts})
        elif action == "MERGE":
            w, l = d.get("winner_id"), d.get("loser_id")
            if w in valid_concepts and l in valid_concepts and w != l:
                loser_snap = _snapshot(conn, user_id, l)
                fold_ids, kept_ids = _merge_guard_partition(
                    conn, user_id, w, loser_snap["member_claim_ids"])
                emit(conn, user_id, run_id, "MERGED", {
                    "winner_id": w, "loser_id": l,
                    "label": d.get("label"), "canonical": d.get("canonical"),
                    "fold_claim_ids": fold_ids, "kept_claim_ids": kept_ids,
                    "winner_snapshot": _snapshot(conn, user_id, w),
                    "loser_snapshot": loser_snap, "ts": ts})
        elif action == "SPLIT" and d.get("concept_id") in valid_concepts:
            members = set(store.concept_member_ids(conn, user_id, d["concept_id"]))
            into = []
            for child in d.get("into", []):
                kept = [c for c in child.get("claim_ids", []) if c in members]
                if kept:
                    into.append({"concept_id": "cpt_" + store.ulid(),
                                 "label": child.get("label", ""),
                                 "canonical": child.get("canonical", ""),
                                 "claim_ids": kept})
            if len(into) >= 2:
                emit(conn, user_id, run_id, "SPLIT", {
                    "concept_id": d["concept_id"],
                    "snapshot": _snapshot(conn, user_id, d["concept_id"]),
                    "into": into, "ts": ts})


def _snapshot(conn, user_id: str, concept_id: str) -> dict:
    c = store.get_concept(conn, user_id, concept_id)
    return {"concept": dict(c) if c else None,
            "member_claim_ids": store.concept_member_ids(conn, user_id, concept_id)}


def _members_with_emb(conn, user_id: str, claim_ids: list[str]) -> list[dict]:
    """Memory rows {"id","text","embedding"} for the reconstruction guard, read
    from STORED vectors (vec_claims) — no embedding call, so it is safe inside a
    txn even when prod embeds via the HF API (memory: slate-embeddings-hf-only)."""
    rows = []
    for cid in claim_ids:
        emb = store.claim_embedding(conn, user_id, cid)
        claim = store.get_claim(conn, user_id, cid)
        if emb is not None and claim is not None:
            rows.append({"id": cid, "text": claim["text"], "embedding": emb})
    return rows


def _merge_guard_partition(conn, user_id: str, winner_id: str,
                           loser_member_ids: list[str]) -> tuple[list[str], list[str]]:
    """C6 nuance guard. Split the loser's members into those the winner already
    reconstructs (safe to FOLD) vs those carrying a nuance the merge would erase
    (KEEP standalone in the loser). Pure geometry over stored vectors; the LLM
    concept-pass already played resolver in proposing the merge, and C14 makes a
    wrong fold reversible. Empty/uncomparable → legacy behaviour (fold all)."""
    survivors = _members_with_emb(conn, user_id,
                                  store.concept_member_ids(conn, user_id, winner_id))
    losers = _members_with_emb(conn, user_id, loser_member_ids)
    if not survivors or not losers:
        return list(loser_member_ids), []
    verdicts = guard.merge(losers, survivors, scope=winner_id)
    fold = [v["id"] for v in verdicts if v["safe_to_drop"]]
    keep = [v["id"] for v in verdicts if not v["safe_to_drop"]]
    return fold, keep


# ── Step 5: relations (spine promotion + contradiction receipts) ──────────────
def _claim_concept(conn, user_id: str, claim_id: str) -> str | None:
    row = conn.execute(
        "SELECT concept_id FROM concept_members WHERE claim_id = ? AND user_id = ? LIMIT 1",
        (claim_id, user_id)).fetchone()
    return row["concept_id"] if row else None


def _relations(conn, user_id: str, run_id: str, episode, bp: dict,
               episode_claims: list[tuple[str, str]]) -> None:
    ts = episode["ts"]
    by_cluster: dict[str, list[str]] = {}
    for cluster_label, claim_id in episode_claims:
        by_cluster.setdefault(cluster_label, []).append(claim_id)

    spine = bp.get("spine") or []
    if isinstance(spine, list):
        for link in spine:
            from_claims = by_cluster.get(link.get("from", ""), [])
            to_claims = by_cluster.get(link.get("to", ""), [])
            relation = (link.get("relation") or "leads_to").strip().lower().replace(" ", "_")
            from_c = next((c for c in (_claim_concept(conn, user_id, cl) for cl in from_claims) if c), None)
            to_c = next((c for c in (_claim_concept(conn, user_id, cl) for cl in to_claims) if c), None)
            if from_c and to_c and from_c != to_c:
                emit(conn, user_id, run_id, "RELATED", {
                    "from_id": from_c, "to_id": to_c, "relation": relation,
                    "weight": 1.0, "evidence_episode_id": episode["id"], "ts": ts})

    receipt = json.loads(episode["receipt_json"] or "{}")
    episode_claim_ids = [cid for _, cid in episode_claims]
    for contra in receipt.get("contradictions", []):
        old_claim = contra.get("claim_id")
        if old_claim and episode_claim_ids:
            emit(conn, user_id, run_id, "RELATED", {
                "from_id": episode_claim_ids[0], "to_id": old_claim,
                "relation": "contradicts", "weight": 1.0,
                "evidence_episode_id": episode["id"], "ts": ts})


# ── Step 5b: reconcile + version conflicts (C8) ───────────────────────────────
# Challenger must outweigh the incumbent by this much (claim strength — the
# spaced-repetition signal) before the CURRENT view flips, so a belief that
# flip-flops can't oscillate forever (PRD §Consolidation). A scopeable knob
# (usage signals from C13 will refine the metric); a constant for now.
VERSION_FLIP_MARGIN = 1.0

PROMPT_VERSION = """Two of the user's notes contradict each other. Decide how they reconcile. Return ONLY JSON.

NEWER claim: "{newer}"
OLDER claim: "{older}"

Pick "mode":
- "supersede": the newer claim replaces the older on better/newer grounds (a changed mind).
- "scope": both are true under DIFFERENT conditions — give each a short "qualifier_newer"/"qualifier_older" naming its condition.
- "version": both genuinely stand as rival views; neither clearly wins.

{"mode": "supersede|scope|version", "qualifier_newer": null, "qualifier_older": null}"""


def _resolve_conflict(newer_text: str, older_text: str) -> tuple[dict, float]:
    """Reconcile a contradiction into supersede/scope/version (PRD: the predictor
    detects, the resolver — LLM — reconciles). On LLM failure the safe default is
    "version": keep both, flip nothing, drop nothing."""
    try:
        result = llm.call(
            PROMPT_VERSION.replace("{newer}", newer_text).replace("{older}", older_text),
            tier="judgment", max_tokens=512)
        j = result["json"]
        mode = (j.get("mode") or "version").lower()
        if mode not in ("supersede", "scope", "version"):
            mode = "version"
        return {"mode": mode, "qualifier_newer": j.get("qualifier_newer"),
                "qualifier_older": j.get("qualifier_older")}, result["cost"]
    except llm.LLMError:
        return {"mode": "version"}, 0.0


def _reconcile(conn, user_id: str, run_id: str, claim_ids: list[str], ts: str) -> float:
    """Resolve every contradiction touching this run's claims. Reads + LLM run
    OUTSIDE any txn (a concurrent save_note never waits on the network); the
    VERSIONED events then commit in one short txn. The flip margin is applied
    here, so the resulting current/other is frozen in the payload → deterministic
    replay (the applier never re-resolves)."""
    seen, decisions, cost = set(), [], 0.0
    for a_id, b_id in store.contradiction_pairs(conn, user_id, claim_ids):
        key = tuple(sorted((a_id, b_id)))
        if key in seen:
            continue
        seen.add(key)
        ca = store.get_claim(conn, user_id, a_id)   # newer (challenger)
        cb = store.get_claim(conn, user_id, b_id)   # older (incumbent)
        if not ca or not cb:
            continue
        res, c = _resolve_conflict(ca["text"], cb["text"])
        cost += c
        decisions.append((dict(ca), dict(cb), res))
    if not decisions:
        return cost

    with conn:
        for ca, cb, res in decisions:
            grp = cb["version_group"] or ca["version_group"] or cb["id"]
            payload = {"version_group": grp, "ts": ts,
                       "qualifier_current": None, "qualifier_other": None}
            mode = res["mode"]
            if mode == "scope":  # both stand under conditions — no flip
                payload.update(mode="scope", current_id=cb["id"], other_id=ca["id"],
                               qualifier_current=res.get("qualifier_older"),
                               qualifier_other=res.get("qualifier_newer"))
            elif (mode == "supersede"
                  and ca["strength"] >= cb["strength"] + VERSION_FLIP_MARGIN):
                payload.update(mode="supersede", current_id=ca["id"],  # newer wins
                               other_id=cb["id"])
            else:  # sub-margin supersede or "version": incumbent stays, challenger held
                payload.update(mode="version", current_id=cb["id"], other_id=ca["id"])
            emit(conn, user_id, run_id, "VERSIONED", payload)
    return cost


# ── Step 5c: store-integrity check (C10) ──────────────────────────────────────
def _primary_support_episode(conn, user_id: str, claim_id: str) -> str | None:
    row = conn.execute(
        "SELECT episode_id FROM claim_support WHERE claim_id = ? AND user_id = ? "
        "ORDER BY episode_id LIMIT 1", (claim_id, user_id)).fetchone()
    return row["episode_id"] if row else None


def _check_integrity(conn, user_id: str, run_id: str, claim_ids: list[str],
                     ts: str) -> None:
    """C10 — is each derived claim faithful to the raw episode it was distilled
    from? Route the claim against its source sentences (the same predictor, X=the
    claim, Y=the raw episode):
      PREDICTED → grounded in the source, faithful (pass)
      NOVEL     → not grounded in the cited source → FLAG (ungrounded)
      AMBIGUOUS → resolver decides direction; contradicts source → FLAG
    Geometry uses STORED vectors (no embedding call); the resolver runs only on
    AMBIGUOUS claims. Flags are an `INTEGRITY_FLAGGED` event (log-only — PRD: a
    store-integrity check, NOT a headline metric), committed in one short txn."""
    flags = []
    for cid in claim_ids:
        claim = store.get_claim(conn, user_id, cid)
        emb = store.claim_embedding(conn, user_id, cid)
        ep_id = _primary_support_episode(conn, user_id, cid)
        if not claim or emb is None or not ep_id:
            continue
        sents = store.episode_sentences_with_vectors(conn, user_id, ep_id)
        if len(sents) < 2:                       # too thin to route → skip (cold)
            continue
        corpus = [{"id": f"{ep_id}:{s['idx']}", "text": s["text"],
                   "embedding": s["embedding"]} for s in sents]
        v = predict.decide(predict.measure(
            [{"id": cid, "text": claim["text"], "embedding": emb}], corpus))["fragments"][0]
        if v["route"] == predict._PREDICTED:
            continue
        if v["route"] == predict._NOVEL:
            flags.append((cid, ep_id, "ungrounded", None))
        else:  # AMBIGUOUS — geometry can't see a flipped polarity; ask the resolver
            anchor = v.get("anchor_text") or ""
            if predict.resolve_direction(claim["text"], anchor)["direction"] == "contradict":
                flags.append((cid, ep_id, "contradicts_source", anchor))
    if not flags:
        return
    with conn:
        for cid, ep_id, kind, anchor in flags:
            emit(conn, user_id, run_id, "INTEGRITY_FLAGGED",
                 {"claim_id": cid, "episode_id": ep_id, "kind": kind,
                  "anchor_text": anchor, "ts": ts})


# ── Step 6: latent bridges (embedding math + small verify calls) ──────────────
def _medoid_vec(V):
    """Representative core of a member set (W7): the member nearest all others
    (max summed cosine). One member → itself."""
    import numpy as np
    if V.shape[0] == 1:
        return V[0]
    return V[int(np.argmax((V @ V.T).sum(axis=1)))]


def _bridges(conn, user_id: str, run_id: str, ts: str) -> float:
    """C5 — propose bridges between concepts whose MEDOIDS sit in a RESIDUAL band:
    close enough to relate (the one's core is partly reconstructable from the
    other's region), far enough that the link is non-obvious (a real residual
    remains — not a near-duplicate concept). Spread-relative via the predictor
    spine (`residuals_against`), replacing the centroid-cosine band. The LLM still
    confirms each candidate (direction/meaning is its job, not the geometry's)."""
    import numpy as np
    concepts = store.all_concepts(conn, user_id)
    if len(concepts) < 2:
        return 0.0

    existing = {(r["from_id"], r["to_id"]) for r in
                conn.execute("SELECT from_id, to_id FROM relations WHERE user_id = ?",
                             (user_id,))}
    members, medoids = {}, {}
    for c in concepts:
        rows = _members_with_emb(conn, user_id,
                                 store.concept_member_ids(conn, user_id, c["id"]))
        if rows:
            V = np.vstack([r["embedding"] for r in rows])
            members[c["id"]] = V
            medoids[c["id"]] = _medoid_vec(V)

    candidates = []
    ids = sorted(medoids.keys())
    for i, a in enumerate(ids):
        for b in ids[i + 1:]:
            if (a, b) in existing or (b, a) in existing:
                continue
            # symmetric: how much of each core the OTHER region can't reconstruct
            r_ab = float(predict.residuals_against(medoids[a], members[b])[0])
            r_ba = float(predict.residuals_against(medoids[b], members[a])[0])
            res = 0.5 * (r_ab + r_ba)
            if BRIDGE_RES_LOW <= res <= BRIDGE_RES_HIGH:
                candidates.append((res, a, b))
    candidates.sort()                       # most-related (lowest residual) first

    cost = 0.0
    by_id = {c["id"]: c for c in concepts}
    for res, a, b in candidates[:BRIDGE_MAX_VERIFY]:
        ca, cb = by_id[a], by_id[b]
        sample = lambda cid: json.dumps([
            (store.get_claim(conn, user_id, m) or {"text": ""})["text"]
            for m in store.concept_member_ids(conn, user_id, cid)[:4]], ensure_ascii=False)
        prompt = PROMPT_BRIDGE.format(
            a_label=ca["label"], a_canonical=ca["canonical"], a_claims=sample(a),
            b_label=cb["label"], b_canonical=cb["canonical"], b_claims=sample(b))
        try:
            result = llm.call(prompt, tier="mechanical", max_tokens=256)  # outside txn
            cost += result["cost"]
            if result["json"].get("bridge"):
                with conn:
                    emit(conn, user_id, run_id, "BRIDGED", {
                        "a": a, "b": b, "score": round(1.0 - res, 3),
                        "rationale": result["json"].get("rationale", ""), "ts": ts})
        except llm.LLMError:
            break  # bridges are best-effort; never fail the run over them
    return cost


# ── Step 7: decay / strengthen (ported v1 health state model) ─────────────────
def _days_since(iso_ts: str | None, now: datetime) -> int:
    if not iso_ts:
        return 999
    try:
        dt = datetime.fromisoformat(iso_ts)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return (now - dt).days
    except (ValueError, TypeError):
        return 999


def _decay_strengthen(conn, user_id: str, run_id: str, episodes, ts: str) -> None:
    # Strengthen: encode-time echo receipts bump the echoed claims.
    for ep in episodes:
        receipt = json.loads(ep["receipt_json"] or "{}")
        for echo in receipt.get("echoes", []):
            if echo.get("claim_id") and store.get_claim(conn, user_id, echo["claim_id"]):
                emit(conn, user_id, run_id, "STRENGTHENED", {
                    "claim_id": echo["claim_id"], "delta": ECHO_BUMP, "ts": ts})

    # Decay: state transitions decided here (with dates), applied from payload.
    now = datetime.now(timezone.utc)
    for c in store.all_concepts(conn, user_id):
        days = _days_since(c["last_activity"], now)
        if days <= config.HEALTH_ACTIVE_DAYS:
            new_state = "active"
        elif days <= config.HEALTH_STALE_DAYS:
            new_state = "stale"
        else:
            new_state = "dormant"
        if new_state != c["state"]:
            emit(conn, user_id, run_id, "DECAYED", {
                "concept_id": c["id"], "state_from": c["state"],
                "state_to": new_state, "days_inactive": days, "ts": ts})


# C9 — background. Times a claim must be RE-predicted (support beyond the first
# episode that minted it) before it folds into its theme. The trigger is high
# recurrence — the opposite of rare — so it never suppresses a rare-but-correct
# claim (PRD's one caution). Folding requires a theme to fold INTO (concept member).
BACKGROUND_MIN_REPEATS = 2


def _demote_background(conn, user_id: str, run_id: str, claim_ids: list[str],
                       ts: str) -> None:
    """Fold claims that have become background into their theme (PRD §Consolidation:
    "things that have become pure background are folded into the theme"). Surprise
    trending to PREDICTED is read as recurrence: a claim re-encountered across
    ≥ BACKGROUND_MIN_REPEATS further episodes, that belongs to a concept, is
    demoted. Scoped to this run's touched claims (where recurrence just grew)."""
    for cid in dict.fromkeys(claim_ids):           # de-dup, keep order
        claim = store.get_claim(conn, user_id, cid)
        if not claim or claim["background"]:
            continue
        if not store.claim_in_any_concept(conn, user_id, cid):
            continue                                # no theme to fold into
        if store.claim_support_count(conn, user_id, cid) - 1 >= BACKGROUND_MIN_REPEATS:
            emit(conn, user_id, run_id, "BACKGROUNDED",
                 {"claim_id": cid, "reason": "recurrent", "ts": ts})


# C13 — retrieval-signal consumption. Times a claim's source episode appeared as a
# retrieval CANDIDATE before a never-fetched claim is demoted. The floor protects
# the rare-but-correct claim that is merely quiet (PRD's one caution on usage).
RETRIEVAL_EXPOSURE_MIN = 3


def _consume_retrieval_signals(conn, user_id: str, run_id: str, ts: str) -> None:
    """Fold logged retrieval usage back into salience — the loop closes here (PRD
    §Consolidation). Bridge: a signal's fragment ids → their source episode → the
    claims derived from it. A claim whose episodes were candidates ≥
    RETRIEVAL_EXPOSURE_MIN times yet NEVER fetched is demoted to background (its
    theme carries it). Idempotent: recomputed from ALL signals; BACKGROUNDED is a
    SET, guarded so it fires once. Promote / un-demote is deferred — "needed"
    cannot be known without the gold/SR@B signal, and usage is logged, never the
    predictor (rare-but-correct must not be suppressed for being quiet)."""
    from collections import Counter
    sigs = store.events_since(conn, user_id, 0, types=["RETRIEVAL_SIGNAL"])
    if not sigs:
        return
    ep_of: dict[str, str | None] = {}

    def episode(fid: str):
        if fid not in ep_of:
            ep_of[fid] = store.fragment_episode(conn, user_id, fid)
        return ep_of[fid]

    seed_ep, fetched_ep = Counter(), Counter()
    for s in sigs:
        p = json.loads(s["payload_json"])
        fetched = p.get("fetched", [])
        for fid in fetched:
            if (ep := episode(fid)):
                fetched_ep[ep] += 1
        for fid in list(fetched) + p.get("dropped", []):   # seed = fetched ∪ dropped
            if (ep := episode(fid)):
                seed_ep[ep] += 1

    claim_seed, claim_fetched = Counter(), Counter()
    for ep, n in seed_ep.items():
        for cid in store.claims_for_episode(conn, user_id, ep):
            claim_seed[cid] += n
            claim_fetched[cid] += fetched_ep.get(ep, 0)

    for cid, seen in claim_seed.items():
        if seen >= RETRIEVAL_EXPOSURE_MIN and claim_fetched[cid] == 0:
            claim = store.get_claim(conn, user_id, cid)
            if claim and not claim["background"]:
                emit(conn, user_id, run_id, "BACKGROUNDED",
                     {"claim_id": cid, "reason": "exposed_never_fetched", "ts": ts})


# C7 — safe forget. Floor below which the guard returns cold_start anyway; an
# explicit gate so prune never touches a thin concept (PRD: protect the rare).
PRUNE_MIN_MEMBERS = 3


def _prune_safely(conn, user_id: str, run_id: str, ts: str) -> None:
    """Prune what the remaining structure can reconstruct (PRD §Consolidation:
    "only let go of what the remaining structure can reconstruct"). Restricted to
    DORMANT concepts (old, quiet); the reconstruction guard's leave-one-out z then
    drops only the reconstructable members and PROTECTS the irreplaceable ones —
    however quiet. Never empties a concept: at least one representative stays."""
    for c in store.all_concepts(conn, user_id):
        if c["state"] != "dormant":
            continue
        member_ids = store.concept_member_ids(conn, user_id, c["id"])
        if len(member_ids) < PRUNE_MIN_MEMBERS:
            continue
        verdicts = guard.forget(_members_with_emb(conn, user_id, member_ids),
                                scope=c["id"])
        droppable = [v["id"] for v in verdicts if v["safe_to_drop"]]
        if len(droppable) >= len(member_ids):   # never forget an entire concept
            droppable = droppable[1:]
        for cid in droppable:
            emit(conn, user_id, run_id, "PRUNED", {
                "concept_id": c["id"], "claim_id": cid,
                "reason": "reconstructable_dormant", "ts": ts})


# ── C1: revisit order — spend effort where surprise was highest ───────────────
def _revisit_order(episodes: list) -> list:
    """Order a batch most-surprising first (PRD §How C1: AMBIGUOUS / high-residual
    revisited first). Surprise is read from the encode-time receipt: a
    contradiction (a residual ON an anchor — the AMBIGUOUS case) outranks raw
    novelty count. Pure WORK-ORDERING — the batch membership (oldest N, fair) is
    unchanged, so it only steers which items the resolver budget hits first."""
    def surprise(ep):
        r = json.loads(ep["receipt_json"] or "{}")
        return (len(r.get("contradictions", [])), r.get("n_novelties", 0))
    return sorted(episodes, key=surprise, reverse=True)


# ── Entry point ───────────────────────────────────────────────────────────────
def consolidate(conn, user_id: str, max_episodes: int = 50) -> dict:
    """One sleep cycle over one user's oldest unconsolidated episodes (sync mode).

    A failed run leaves its episodes unmarked, so the next run retries them;
    md5 claim ids + ON CONFLICT appliers make replayed decisions idempotent.
    """
    episodes = _revisit_order(
        store.unconsolidated_episodes(conn, user_id)[:max_episodes])
    if not episodes:
        return {"status": "noop", "episodes": 0}

    run_id = "run_" + store.ulid()
    ts = _now_iso()
    with conn:
        store.start_run(conn, user_id, run_id, len(episodes))

    cost = 0.0
    new_claim_ids: list[str] = []
    per_episode: list[tuple] = []
    try:
        # Helpers manage their own SHORT transactions; LLM calls never run
        # inside one, so a concurrent save_note never waits on the network.
        skipped: list[str] = []
        for ep in episodes:
            bp = _existing_blueprint(conn, user_id, ep["id"])
            if bp is not None:  # retry of a failed run — reuse, don't re-extract
                episode_claims = _existing_canon(conn, user_id, ep["id"])
            else:
                try:
                    bp, bp_cost = blueprint(ep["raw_text"])  # LLM, no txn
                except llm.LLMError:
                    # One stubborn note must not kill the night: leave it
                    # unconsolidated; the next run retries it.
                    skipped.append(ep["id"])
                    continue
                cost += bp_cost
                episode_claims, canon_cost = _canonicalize_episode(
                    conn, user_id, run_id, ep, bp,
                    {"episode_id": ep["id"], "blueprint": bp,
                     "method": bp.get("_method"), "ts": ts})
                cost += canon_cost
            new_claim_ids.extend(cid for _, cid in episode_claims)
            per_episode.append((ep, bp, episode_claims))

        cost += _concept_pass(conn, user_id, run_id, new_claim_ids, ts)
        with conn:
            for ep, bp, episode_claims in per_episode:
                _relations(conn, user_id, run_id, ep, bp, episode_claims)

        # C8: reconcile contradictions emitted above (LLM, so its own short txn).
        cost += _reconcile(conn, user_id, run_id, new_claim_ids, ts)
        # C10: flag derived claims that aren't faithful to their raw source.
        _check_integrity(conn, user_id, run_id, new_claim_ids, ts)
        # C9: fold recurrent (now-background) claims into their theme.
        # C13: fold logged retrieval usage back into salience (close the loop).
        with conn:
            _demote_background(conn, user_id, run_id, new_claim_ids, ts)
            _consume_retrieval_signals(conn, user_id, run_id, ts)
        cost += _bridges(conn, user_id, run_id, ts)
        with conn:
            _decay_strengthen(conn, user_id, run_id, episodes, ts)
            _prune_safely(conn, user_id, run_id, ts)

        with conn:
            for ep, _, _ in per_episode:  # skipped episodes stay unconsolidated
                store.mark_consolidated(conn, user_id, ep["id"], run_id)
            store.finish_run(conn, run_id, "ok", round(cost, 4))
        return {"status": "ok", "run_id": run_id, "episodes": len(per_episode),
                "skipped": skipped, "claims_touched": len(set(new_claim_ids)),
                "cost": round(cost, 4)}
    except Exception:
        with conn:
            store.finish_run(conn, run_id, "failed", round(cost, 4))
        raise


def consolidate_all_users(conn, max_episodes: int = 50) -> list[dict]:
    """One cycle per user with pending episodes — the nightly cron entry
    (AUTH.md §4). Each user gets their own run rows and LLM cost; one user's
    failure must not block the others."""
    reports = []
    for uid in store.users_with_unconsolidated(conn):
        try:
            report = consolidate(conn, uid, max_episodes=max_episodes)
        except Exception as e:
            report = {"status": "failed", "error": str(e)}
        reports.append({"user_id": uid, **report})
    return reports
