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

from core import config, llm, store
from core.encode import get_embedder, split_sentences

# Strength deltas (spaced-repetition pressure)
SUPPORT_BUMP = 0.5   # claim re-encountered via a new episode at canonicalization
ECHO_BUMP = 0.25     # claim echoed in an encode-time receipt

# Canonicalization similarity bands
CANON_AUTO_SAME = 0.92   # >= : same claim, no LLM needed
CANON_LLM_BAND = 0.75    # [band, auto) : ask the LLM; below: new claim

# Bridge candidate band (centroid cosine): close enough to relate,
# far enough that the connection is non-obvious
BRIDGE_LOW, BRIDGE_HIGH = 0.45, 0.80
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


def claim_id_for(text: str) -> str:
    return "clm_" + hashlib.md5(text.strip().lower().encode("utf-8")).hexdigest()


# ── Event emit + apply (the backbone) ─────────────────────────────────────────
def emit(conn, run_id: str | None, type_: str, payload: dict) -> None:
    """Append the event, then materialize it. Decision → event → row, always."""
    store.append_event(conn, type_, payload, run_id=run_id)
    apply_event(conn, type_, payload)


def apply_event(conn, type_: str, payload: dict) -> None:
    """Materialize one event into the semantic tables. Deterministic."""
    p = payload
    if type_ == "CANONICALIZED":
        if p["action"] == "new":
            emb = get_embedder().encode([p["text"]], normalize_embeddings=True,
                                        show_progress_bar=False)[0]
            store.insert_claim(conn, p["claim_id"], p["text"], emb, p["ts"])
        else:  # support: re-encounter of an existing claim
            store.bump_claim_strength(conn, p["claim_id"], p["ts"], SUPPORT_BUMP)
        store.add_claim_support(conn, p["claim_id"], p["episode_id"], p.get("verbatim"))

    elif type_ == "CONCEPT_CREATED":
        store.insert_concept(conn, p["concept_id"], p["label"], p["canonical"], p["ts"])
        for cid in p["claim_ids"]:
            store.add_concept_member(conn, p["concept_id"], cid)
        store.recompute_concept_embedding(conn, p["concept_id"])

    elif type_ == "ATTACHED":
        for cid in p["claim_ids"]:
            store.add_concept_member(conn, p["concept_id"], cid)
        store.update_concept(conn, p["concept_id"], last_activity=p["ts"])
        store.recompute_concept_embedding(conn, p["concept_id"])

    elif type_ == "MERGED":
        # Snapshots of both concepts ride in the payload; the loser's history
        # lives in the event log, so removing its row is non-destructive.
        for cid in p["loser_snapshot"]["member_claim_ids"]:
            store.add_concept_member(conn, p["winner_id"], cid)
        store.delete_concept(conn, p["loser_id"])
        store.update_concept(conn, p["winner_id"], label=p.get("label"),
                             canonical=p.get("canonical"), last_activity=p["ts"])
        store.recompute_concept_embedding(conn, p["winner_id"])

    elif type_ == "SPLIT":
        store.delete_concept(conn, p["concept_id"])
        for child in p["into"]:
            store.insert_concept(conn, child["concept_id"], child["label"],
                                 child["canonical"], p["ts"])
            for cid in child["claim_ids"]:
                store.add_concept_member(conn, child["concept_id"], cid)
            store.recompute_concept_embedding(conn, child["concept_id"])

    elif type_ == "RELATED":
        store.insert_relation(conn, p["from_id"], p["to_id"], p["relation"],
                              p.get("weight", 1.0), p["ts"],
                              p.get("evidence_episode_id"))

    elif type_ == "BRIDGED":
        store.insert_relation(conn, p["a"], p["b"], "bridges",
                              p.get("score", 1.0), p["ts"],
                              p.get("evidence_episode_id"))

    elif type_ == "STRENGTHENED":
        if p.get("claim_id"):
            store.bump_claim_strength(conn, p["claim_id"], p["ts"], p["delta"])
        if p.get("concept_id"):
            c = store.get_concept(conn, p["concept_id"])
            if c:
                store.update_concept(conn, p["concept_id"],
                                     strength=c["strength"] + p["delta"],
                                     last_activity=p["ts"])

    elif type_ == "DECAYED":
        store.update_concept(conn, p["concept_id"], state=p["state_to"])

    # ENCODED / BLUEPRINTED: episodic-side or log-only — nothing to materialize.


def rebuild(conn) -> dict:
    """Truncate the semantic store and re-apply the entire event log."""
    with conn:
        store.truncate_semantic(conn)
        events = store.events_since(conn, 0)
        for ev in events:
            apply_event(conn, ev["type"], json.loads(ev["payload_json"]))
    return {"events_applied": len(events)}


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
    """LLM blueprint with local KMeans fallback. Returns (blueprint, cost)."""
    try:
        result = llm.call(PROMPT_BLUEPRINT + text, tier="mechanical", max_tokens=2048)
        bp = result["json"]
        bp["_method"] = result["provider"]
        return bp, result["cost"]
    except llm.LLMError:
        return _blueprint_local(text), 0.0


# ── Retry reuse: a failed run's blueprint + canon events are authoritative ────
# Blueprint extraction is nondeterministic, so re-extracting on retry mints
# near-duplicate claims. An episode's BLUEPRINTED event and its CANONICALIZED
# events are written in one transaction, so either both exist fully or neither.
def _existing_blueprint(conn, episode_id: str) -> dict | None:
    row = conn.execute(
        """SELECT payload_json FROM events WHERE type = 'BLUEPRINTED'
           AND json_extract(payload_json, '$.episode_id') = ?
           ORDER BY seq DESC LIMIT 1""", (episode_id,)).fetchone()
    return json.loads(row["payload_json"])["blueprint"] if row else None


def _existing_canon(conn, episode_id: str) -> list[tuple[str, str]]:
    rows = conn.execute(
        """SELECT payload_json FROM events WHERE type = 'CANONICALIZED'
           AND json_extract(payload_json, '$.episode_id') = ? ORDER BY seq""",
        (episode_id,)).fetchall()
    out = []
    for r in rows:
        p = json.loads(r["payload_json"])
        out.append((p.get("cluster", ""), p["claim_id"]))
    return out


# ── Step 3: claim canonicalization ────────────────────────────────────────────
def _canonicalize_episode(conn, run_id: str, episode, bp: dict,
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

    decided, uncertain = [], []
    for c, emb in zip(raw_claims, embs):
        hits = store.knn_claims(conn, emb, k=3)
        best = hits[0] if hits else None
        if best and best["similarity"] >= CANON_AUTO_SAME:
            decided.append((c, "same", best["claim_id"]))
        elif best and best["similarity"] >= CANON_LLM_BAND:
            uncertain.append((c, best))
        else:
            decided.append((c, "new", None))

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
        store.append_event(conn, "BLUEPRINTED", bp_payload, run_id=run_id)
        for c, action, existing_id in decided:
            if action == "same":
                emit(conn, run_id, "CANONICALIZED", {
                    "action": "support", "claim_id": existing_id,
                    "episode_id": episode["id"], "verbatim": c["verbatim"],
                    "cluster": c["cluster"], "ts": ts})
                episode_claims.append((c["cluster"], existing_id))
            else:
                cid = claim_id_for(c["text"])
                emit(conn, run_id, "CANONICALIZED", {
                    "action": "new", "claim_id": cid, "text": c["text"],
                    "episode_id": episode["id"], "verbatim": c["verbatim"],
                    "cluster": c["cluster"], "ts": ts})
                episode_claims.append((c["cluster"], cid))
    return episode_claims, cost


# ── Step 4: concept pass (the judgment call) ──────────────────────────────────
def _concept_pass(conn, run_id: str, new_claim_ids: list[str], ts: str) -> float:
    """Chunked so each judgment call stays within output budget; later chunks
    see concepts created by earlier ones, so attachment stays incremental."""
    unique_ids = list(dict.fromkeys(new_claim_ids))
    cost = 0.0
    for start in range(0, len(unique_ids), CONCEPT_CHUNK):
        cost += _concept_pass_chunk(conn, run_id,
                                    unique_ids[start:start + CONCEPT_CHUNK], ts)
    return cost


def _concept_pass_chunk(conn, run_id: str, new_claim_ids: list[str], ts: str) -> float:
    if not new_claim_ids:
        return 0.0

    new_claims = []
    nearby: dict[str, dict] = {}
    for cid in new_claim_ids:
        row = store.get_claim(conn, cid)
        if not row:
            continue
        new_claims.append({"id": cid, "text": row["text"]})
        emb = store.claim_embedding(conn, cid)
        if emb is not None:
            for hit in store.knn_concepts(conn, emb, k=3):
                if hit["similarity"] >= 0.40:
                    nearby[hit["id"]] = hit

    concepts_ctx = []
    for c in nearby.values():
        member_ids = store.concept_member_ids(conn, c["id"])[:CONCEPT_CONTEXT_MEMBERS]
        members = [{"id": m, "text": (store.get_claim(conn, m) or {"text": ""})["text"]}
                   for m in member_ids]
        concepts_ctx.append({"id": c["id"], "label": c["label"],
                             "canonical": c["canonical"], "members": members})

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
        _apply_concept_decisions(conn, run_id, decisions, valid_claims,
                                 valid_concepts, ts)
    return result["cost"]


def _apply_concept_decisions(conn, run_id, decisions, valid_claims,
                             valid_concepts, ts) -> None:
    for d in decisions:
        action = d.get("action", "").upper()
        if action == "CREATE":
            claim_ids = [c for c in d.get("claim_ids", []) if c in valid_claims]
            if not claim_ids:
                continue
            emit(conn, run_id, "CONCEPT_CREATED", {
                "concept_id": "cpt_" + store.ulid(), "label": d.get("label", ""),
                "canonical": d.get("canonical", ""), "claim_ids": claim_ids, "ts": ts})
        elif action == "ATTACH" and d.get("concept_id") in valid_concepts:
            claim_ids = [c for c in d.get("claim_ids", []) if c in valid_claims]
            if claim_ids:
                emit(conn, run_id, "ATTACHED", {
                    "concept_id": d["concept_id"], "claim_ids": claim_ids, "ts": ts})
        elif action == "MERGE":
            w, l = d.get("winner_id"), d.get("loser_id")
            if w in valid_concepts and l in valid_concepts and w != l:
                emit(conn, run_id, "MERGED", {
                    "winner_id": w, "loser_id": l,
                    "label": d.get("label"), "canonical": d.get("canonical"),
                    "winner_snapshot": _snapshot(conn, w),
                    "loser_snapshot": _snapshot(conn, l), "ts": ts})
        elif action == "SPLIT" and d.get("concept_id") in valid_concepts:
            members = set(store.concept_member_ids(conn, d["concept_id"]))
            into = []
            for child in d.get("into", []):
                kept = [c for c in child.get("claim_ids", []) if c in members]
                if kept:
                    into.append({"concept_id": "cpt_" + store.ulid(),
                                 "label": child.get("label", ""),
                                 "canonical": child.get("canonical", ""),
                                 "claim_ids": kept})
            if len(into) >= 2:
                emit(conn, run_id, "SPLIT", {
                    "concept_id": d["concept_id"],
                    "snapshot": _snapshot(conn, d["concept_id"]),
                    "into": into, "ts": ts})


def _snapshot(conn, concept_id: str) -> dict:
    c = store.get_concept(conn, concept_id)
    return {"concept": dict(c) if c else None,
            "member_claim_ids": store.concept_member_ids(conn, concept_id)}


# ── Step 5: relations (spine promotion + contradiction receipts) ──────────────
def _claim_concept(conn, claim_id: str) -> str | None:
    row = conn.execute(
        "SELECT concept_id FROM concept_members WHERE claim_id = ? LIMIT 1",
        (claim_id,)).fetchone()
    return row["concept_id"] if row else None


def _relations(conn, run_id: str, episode, bp: dict,
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
            from_c = next((c for c in (_claim_concept(conn, cl) for cl in from_claims) if c), None)
            to_c = next((c for c in (_claim_concept(conn, cl) for cl in to_claims) if c), None)
            if from_c and to_c and from_c != to_c:
                emit(conn, run_id, "RELATED", {
                    "from_id": from_c, "to_id": to_c, "relation": relation,
                    "weight": 1.0, "evidence_episode_id": episode["id"], "ts": ts})

    receipt = json.loads(episode["receipt_json"] or "{}")
    episode_claim_ids = [cid for _, cid in episode_claims]
    for contra in receipt.get("contradictions", []):
        old_claim = contra.get("claim_id")
        if old_claim and episode_claim_ids:
            emit(conn, run_id, "RELATED", {
                "from_id": episode_claim_ids[0], "to_id": old_claim,
                "relation": "contradicts", "weight": 1.0,
                "evidence_episode_id": episode["id"], "ts": ts})


# ── Step 6: latent bridges (embedding math + small verify calls) ──────────────
def _bridges(conn, run_id: str, ts: str) -> float:
    import numpy as np
    concepts = store.all_concepts(conn)
    if len(concepts) < 2:
        return 0.0

    existing = {(r["from_id"], r["to_id"]) for r in
                conn.execute("SELECT from_id, to_id FROM relations")}
    vecs = {}
    for c in concepts:
        row = conn.execute("SELECT embedding FROM vec_concepts WHERE concept_id = ?",
                           (c["id"],)).fetchone()
        if row:
            vecs[c["id"]] = store._deserialize(row["embedding"])

    candidates = []
    ids = sorted(vecs.keys())
    for i, a in enumerate(ids):
        for b in ids[i + 1:]:
            if (a, b) in existing or (b, a) in existing:
                continue
            sim = float(np.dot(vecs[a], vecs[b]))
            if BRIDGE_LOW <= sim <= BRIDGE_HIGH:
                candidates.append((sim, a, b))
    candidates.sort(reverse=True)

    cost = 0.0
    by_id = {c["id"]: c for c in concepts}
    for sim, a, b in candidates[:BRIDGE_MAX_VERIFY]:
        ca, cb = by_id[a], by_id[b]
        sample = lambda cid: json.dumps([
            (store.get_claim(conn, m) or {"text": ""})["text"]
            for m in store.concept_member_ids(conn, cid)[:4]], ensure_ascii=False)
        prompt = PROMPT_BRIDGE.format(
            a_label=ca["label"], a_canonical=ca["canonical"], a_claims=sample(a),
            b_label=cb["label"], b_canonical=cb["canonical"], b_claims=sample(b))
        try:
            result = llm.call(prompt, tier="mechanical", max_tokens=256)  # outside txn
            cost += result["cost"]
            if result["json"].get("bridge"):
                with conn:
                    emit(conn, run_id, "BRIDGED", {
                        "a": a, "b": b, "score": round(sim, 3),
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


def _decay_strengthen(conn, run_id: str, episodes, ts: str) -> None:
    # Strengthen: encode-time echo receipts bump the echoed claims.
    for ep in episodes:
        receipt = json.loads(ep["receipt_json"] or "{}")
        for echo in receipt.get("echoes", []):
            if echo.get("claim_id") and store.get_claim(conn, echo["claim_id"]):
                emit(conn, run_id, "STRENGTHENED", {
                    "claim_id": echo["claim_id"], "delta": ECHO_BUMP, "ts": ts})

    # Decay: state transitions decided here (with dates), applied from payload.
    now = datetime.now(timezone.utc)
    for c in store.all_concepts(conn):
        days = _days_since(c["last_activity"], now)
        if days <= config.HEALTH_ACTIVE_DAYS:
            new_state = "active"
        elif days <= config.HEALTH_STALE_DAYS:
            new_state = "stale"
        else:
            new_state = "dormant"
        if new_state != c["state"]:
            emit(conn, run_id, "DECAYED", {
                "concept_id": c["id"], "state_from": c["state"],
                "state_to": new_state, "days_inactive": days, "ts": ts})


# ── Entry point ───────────────────────────────────────────────────────────────
def consolidate(conn, max_episodes: int = 50) -> dict:
    """One sleep cycle over the oldest unconsolidated episodes (sync mode).

    A failed run leaves its episodes unmarked, so the next run retries them;
    md5 claim ids + ON CONFLICT appliers make replayed decisions idempotent.
    """
    episodes = store.unconsolidated_episodes(conn)[:max_episodes]
    if not episodes:
        return {"status": "noop", "episodes": 0}

    run_id = "run_" + store.ulid()
    ts = _now_iso()
    with conn:
        store.start_run(conn, run_id, len(episodes))

    cost = 0.0
    new_claim_ids: list[str] = []
    per_episode: list[tuple] = []
    try:
        # Helpers manage their own SHORT transactions; LLM calls never run
        # inside one, so a concurrent save_note never waits on the network.
        for ep in episodes:
            bp = _existing_blueprint(conn, ep["id"])
            if bp is not None:  # retry of a failed run — reuse, don't re-extract
                episode_claims = _existing_canon(conn, ep["id"])
            else:
                bp, bp_cost = blueprint(ep["raw_text"])  # LLM, no txn
                cost += bp_cost
                episode_claims, canon_cost = _canonicalize_episode(
                    conn, run_id, ep, bp,
                    {"episode_id": ep["id"], "blueprint": bp,
                     "method": bp.get("_method"), "ts": ts})
                cost += canon_cost
            new_claim_ids.extend(cid for _, cid in episode_claims)
            per_episode.append((ep, bp, episode_claims))

        cost += _concept_pass(conn, run_id, new_claim_ids, ts)
        with conn:
            for ep, bp, episode_claims in per_episode:
                _relations(conn, run_id, ep, bp, episode_claims)

        cost += _bridges(conn, run_id, ts)
        with conn:
            _decay_strengthen(conn, run_id, episodes, ts)

        with conn:
            for ep in episodes:
                store.mark_consolidated(conn, ep["id"], run_id)
            store.finish_run(conn, run_id, "ok", round(cost, 4))
        return {"status": "ok", "run_id": run_id, "episodes": len(episodes),
                "claims_touched": len(set(new_claim_ids)), "cost": round(cost, 4)}
    except Exception:
        with conn:
            store.finish_run(conn, run_id, "failed", round(cost, 4))
        raise
