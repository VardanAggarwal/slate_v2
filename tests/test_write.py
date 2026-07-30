"""Write pipeline (W2–W8) tests — execution plan §1 + §5.

Two layers, mirroring test_predict / test_wrappers:
  - PURE CORE (route_fragments / plan_fragments): synthetic near-orthogonal
    vectors with planted structure (anchors, near-dups, intra-note repeats), so
    routing is checkable without an LLM or the live DB. POLICY (the route) is
    asserted under a calibration FITTED to the fixture — never a baked constant.
  - INTEGRATION (refine_episode): the real local embedder + a temp DB, asserting
    the plumbing — fragments persisted, FRAGMENTED event emitted, idempotency,
    rebuild fidelity, retry-after-failure — without asserting model-specific routes.
Plus the W6 prod fix (HFStance, no torch) and the store helpers.
"""
import json

import numpy as np
import pytest

from core import config, predict, scan, store, write
from core.encode import encode
from tests.conftest import UID, UID_B

DIM = 48
SEED = 7


# ── synthetic-geometry helpers (shared shape with test_predict) ───────────────
def _topic_dirs(n, rng):
    Q, _ = np.linalg.qr(rng.standard_normal((n, DIM)).T)
    return Q.T[:n]


def _frag(direction, rng, noise=0.04):
    v = direction + noise * rng.standard_normal(DIM)
    return v / np.linalg.norm(v)


def _rows(vecs, cluster, start=0):
    return [{"id": start + i, "text": f"{cluster}{start + i}",
             "embedding": v, "cluster": cluster} for i, v in enumerate(vecs)]


def _z(probe, Y, baselines):
    return predict.measure([{"text": "p", "embedding": probe}], Y,
                           baselines=baselines)[0]["z"]


# ══════════════════════════════════════════════════════════════════════════════
# PURE CORE — route_fragments (W4+W5+W6+W7), no DB / no network
# ══════════════════════════════════════════════════════════════════════════════
def test_route_matrix_predicted_ambiguous_novel():
    """Near-dup-of-memory → PREDICTED (reinforced, not stored); same-topic-looser
    → AMBIGUOUS (stored + resolver); off-topic → NOVEL (stored, no anchor). The
    z-ordering is the instrument; the routes fall out under a fitted z_echo."""
    rng = np.random.default_rng(SEED)
    dirs = _topic_dirs(3, rng)
    memory = _rows([_frag(dirs[0], rng) for _ in range(8)], "A")
    bl = predict.compute_baselines(memory)
    near = _frag(dirs[0], rng, noise=0.005)
    fresh = _frag(dirs[0], rng, noise=0.03)
    off = _frag(dirs[2], rng)
    cut = (_z(near, memory, bl) + _z(fresh, memory, bl)) / 2   # between the two
    calib = {"z_echo": cut, "prox_margin": predict.PROX_MARGIN}

    calls = []

    def classify(premise, hypothesis):
        calls.append((premise, hypothesis))
        return "neutral"

    out = write.route_fragments(
        [("near", 0, 0), ("fresh", 1, 1), ("off", 2, 2)],
        np.vstack([near, fresh, off]), memory, classify=classify, calibration=calib)

    routes = [p["route"] for p in out["planned"]]
    assert routes == ["AMBIGUOUS", "NOVEL"]                 # PREDICTED never stored
    assert out["reinforced"] and out["reinforced"][0] < 8  # near reinforced an A member
    fresh_p = next(p for p in out["planned"] if p["text"] == "fresh")
    off_p = next(p for p in out["planned"] if p["text"] == "off")
    assert fresh_p["anchor_id"] is not None and off_p["anchor_id"] is None
    assert len(calls) == 1                                  # ONLY the AMBIGUOUS one hit the resolver
    # W7: centre = lowest-residual stored; peak = highest-z stored
    assert out["centre_id"] == fresh_p["frag_id"]
    assert out["novel_peak_id"] == off_p["frag_id"]
    # weights are a within-note residual share over the stored set
    assert abs(sum(p["weight"] for p in out["planned"]) - 1.0) < 1e-2


def test_route_intra_note_dedup():
    """W4 — a within-note repeat collapses against its sibling, not memory: the
    second of two near-identical fragments routes PREDICTED with the FIRST as its
    anchor, is counted as an intra-note echo, and is not stored twice."""
    rng = np.random.default_rng(SEED)
    dirs = _topic_dirs(2, rng)
    memory = _rows([_frag(dirs[1], rng) for _ in range(8)], "B")  # off-topic to topic 0
    bl = predict.compute_baselines(memory)
    f1 = _frag(dirs[0], rng, noise=0.01)
    f2 = _frag(dirs[0], rng, noise=0.01)                          # ≈ f1
    sib = [{"id": "frg_0", "text": "f1", "embedding": f1, "cluster": None}]
    cut = (_z(f2, memory + sib, bl) + _z(f1, memory, bl)) / 2     # f2 below, f1 above
    calib = {"z_echo": cut, "prox_margin": predict.PROX_MARGIN}

    out = write.route_fragments([("f1", 0, 0), ("f2", 1, 1)], np.vstack([f1, f2]),
                                memory, calibration=calib,
                                frag_id_fn=lambda s, e: f"frg_{s}")
    assert [p["route"] for p in out["planned"]] == ["NOVEL"]      # only f1 stored
    assert out["n_intra_echo"] == 1
    assert out["reinforced"] == ["frg_0"]                         # f2 reinforced its sibling


def test_per_cluster_threshold_decides_predicted_vs_ambiguous():
    """The per-cluster knob is LIVE: PREDICTED↔AMBIGUOUS is decided by a PER-REGION
    z_echo, not one global bar. Two equally-novel probes — one on region A, one on
    region B — both route AMBIGUOUS under a global bar; raising ONLY A's z_echo via
    `per_cluster` flips the A probe to PREDICTED while B is untouched."""
    rng = np.random.default_rng(SEED)
    dirs = _topic_dirs(2, rng)
    memory = _rows([_frag(dirs[0], rng) for _ in range(8)], "A", start=0) + \
             _rows([_frag(dirs[1], rng) for _ in range(8)], "B", start=8)
    bl = predict.compute_baselines(memory)
    pa = _frag(dirs[0], rng, 0.03)              # attached to region A
    pb = _frag(dirs[1], rng, 0.03)              # attached to region B
    za, zb = _z(pa, memory, bl), _z(pb, memory, bl)
    base = {"z_echo": min(za, zb) - 1.0, "prox_margin": predict.PROX_MARGIN,
            "per_cluster": {}}
    quiet = lambda p, h: "neutral"

    out0 = write.route_fragments([("pa", 0, 0), ("pb", 1, 1)], np.vstack([pa, pb]),
                                 memory, classify=quiet, calibration=base)
    assert {p["text"]: p["route"] for p in out0["planned"]} == \
        {"pa": "AMBIGUOUS", "pb": "AMBIGUOUS"}      # one global bar → both stored

    # raise ONLY region A's bar above pa's z → A's probe now reads as PREDICTED
    perc = {**base, "per_cluster": {"A": {"z_echo": za + 0.5}}}
    out1 = write.route_fragments([("pa", 0, 0), ("pb", 1, 1)], np.vstack([pa, pb]),
                                 memory, classify=quiet, calibration=perc)
    routes = {p["text"]: p["route"] for p in out1["planned"]}
    assert "pa" not in routes                   # dropped by region A's stricter bar
    assert routes.get("pb") == "AMBIGUOUS"      # region B unchanged
    assert out1["reinforced"]                   # pa reinforced an A member


def test_route_empty_memory_all_novel():
    """Cold start: with no memory and < 2 peers, nothing can be predicted, so the
    first fragments are all NOVEL (never crashes on an empty corpus)."""
    rng = np.random.default_rng(SEED)
    d = _topic_dirs(1, rng)[0]
    embs = np.vstack([_frag(d, rng) for _ in range(2)])
    out = write.route_fragments([("a", 0, 0), ("b", 1, 1)], embs, [])
    assert [p["route"] for p in out["planned"]] == ["NOVEL", "NOVEL"]
    assert out["reinforced"] == []


def test_plan_fragments_partitions_the_note():
    """W2+W3 — fragments contiguously partition every sentence, with joined text
    and ordered spans (boundary placement itself is scan's tested concern)."""
    rng = np.random.default_rng(SEED)
    dirs = _topic_dirs(2, rng)
    E = np.vstack([_frag(dirs[0], rng) for _ in range(3)] +
                  [_frag(dirs[1], rng) for _ in range(3)])
    texts = [f"sentence number {i} here" for i in range(6)]
    specs = write.plan_fragments(texts, E, calibration=scan.DEFAULT_CALIBRATION)
    covered = sorted(i for _, s, e in specs for i in range(s, e + 1))
    assert covered == list(range(6))                       # exact partition, no gaps/overlap
    assert all(s <= e for _, s, e in specs)
    assert specs[0][0].startswith("sentence number 0")     # joined verbatim in order


def test_route_all_predicted_note_stores_nothing():
    """The router floor: a note whose every fragment merely restates memory routes
    ALL PREDICTED — nothing stored, every anchor reinforced, and the W7 readouts
    degrade cleanly (centre/peak None, no weight div-by-zero on an empty stored
    set). This is the 'write is a redundancy filter, not a compressor' case."""
    rng = np.random.default_rng(SEED)
    d = _topic_dirs(1, rng)[0]
    memory = _rows([_frag(d, rng, noise=0.01) for _ in range(8)], "A")
    bl = predict.compute_baselines(memory)
    near = [_frag(d, rng, noise=0.004), _frag(d, rng, noise=0.004)]
    zs = [_z(e, memory, bl) for e in near]
    calib = {"z_echo": max(zs) + 0.5, "prox_margin": predict.PROX_MARGIN}  # both below → PREDICTED

    out = write.route_fragments([("a", 0, 0), ("b", 1, 1)], np.vstack(near),
                                memory, calibration=calib)
    assert out["planned"] == []                       # nothing crossed into the store
    assert len(out["reinforced"]) == 2                # both confirmed a memory anchor
    assert out["centre_id"] is None and out["novel_peak_id"] is None
    assert out["n_intra_echo"] == 0                   # collapsed against memory, not siblings


def test_z_echo_is_the_compression_dial():
    """z_echo IS the write-side compression knob: over a FIXED measurement set,
    raising it monotonically moves fragments from stored → PREDICTED (dropped) and
    never the reverse. NOVEL-vs-AMBIGUOUS (the anchor-present split) is independent
    of it. This is the invariant the compression-ratio sweep rides on."""
    rng = np.random.default_rng(SEED)
    dirs = _topic_dirs(3, rng)
    memory = _rows([_frag(dirs[0], rng) for _ in range(8)], "A")
    bl = predict.compute_baselines(memory)
    probes = [_frag(dirs[0], rng, 0.005), _frag(dirs[0], rng, 0.02),
              _frag(dirs[0], rng, 0.05), _frag(dirs[2], rng)]
    ms = predict.measure([{"text": f"p{i}", "embedding": e} for i, e in enumerate(probes)],
                         memory, baselines=bl)

    dropped = []
    for ze in (-5.0, -2.0, -1.0, 0.0, 1.0, 2.0):
        out = predict.decide(ms, {"z_echo": ze, "prox_margin": predict.PROX_MARGIN})
        dropped.append(sum(1 for v in out["fragments"] if v["route"] == write.PREDICTED))
    assert dropped == sorted(dropped)                 # monotonic non-decreasing in z_echo
    assert dropped[0] == 0 and dropped[-1] > dropped[0]  # the dial actually turns


def test_fragment_medoid_selects_a_real_sentence():
    """A fragment's vector is a SELECTED sentence (its medoid) — never a synthesized
    average: identical to one of the span's sentence vectors, and the central
    (lowest-residual) one, not the outlier. A 1-sentence span is its own medoid."""
    rng = np.random.default_rng(SEED)
    d = _topic_dirs(2, rng)
    E = np.vstack([_frag(d[0], rng, 0.01), _frag(d[0], rng, 0.01), _frag(d[1], rng)])
    texts = ["central a", "central b", "outlier c"]
    idx, vec = write._fragment_medoid(texts, E, 0, 2)
    assert idx in (0, 1)                       # a central topic-0 sentence, not the outlier
    assert np.allclose(vec, E[idx])            # selection, not generation: a real vector
    j, v = write._fragment_medoid(texts, E, 2, 2)   # single-sentence span
    assert j == 2 and np.allclose(v, E[2])


def test_refine_makes_no_embedding_call(conn, monkeypatch):
    """Point 4: fragment vectors are medoid sentences reused from encode, so refine
    performs NO embedding round-trip — patching the embedder to raise still completes —
    and every fragment REFERENCES a real episode sentence (medoid_idx) rather than
    storing its own vector. The referenced vector is that sentence's encode-time vector."""
    ep = encode(conn, UID, NOTE, source="test")["episode_id"]

    from core import encode as encode_mod
    monkeypatch.setattr(encode_mod, "get_embedder",
                        lambda: (_ for _ in ()).throw(AssertionError("refine must not embed")))

    r = write.refine_episode(conn, UID, ep)
    assert r["status"] == "ok" and r["n_fragments"] >= 1
    sents = store.episode_sentences_with_vectors(conn, UID, ep)
    rows = conn.execute("SELECT id, medoid_idx FROM fragments WHERE episode_id=?",
                        (ep,)).fetchall()
    assert rows and all(0 <= f["medoid_idx"] < len(sents) for f in rows)  # real references
    # the pool resolves each fragment's vector to its medoid sentence vector (no copy)
    pool = {p["id"]: p["embedding"] for p in store.fragment_pool(conn, UID)}
    for f in rows:
        assert np.allclose(pool[f["id"]], sents[f["medoid_idx"]]["embedding"], atol=1e-5)


def test_route_contradiction_held_strongest():
    """W6 — a fragment the resolver flags as a contradiction is born held STRONGER
    than a refine (PRD: 'store, held strongest'); the resolver's sign is carried,
    not discarded."""
    rng = np.random.default_rng(SEED)
    dirs = _topic_dirs(3, rng)
    memory = _rows([_frag(dirs[0], rng) for _ in range(8)], "A")
    bl = predict.compute_baselines(memory)
    near = _frag(dirs[0], rng, noise=0.005)
    fresh = _frag(dirs[0], rng, noise=0.03)         # AMBIGUOUS → resolver runs
    cut = (_z(near, memory, bl) + _z(fresh, memory, bl)) / 2
    calib = {"z_echo": cut, "prox_margin": predict.PROX_MARGIN}
    out = write.route_fragments([("fresh", 0, 0)], np.atleast_2d(fresh), memory,
                                classify=lambda p, h: "contradiction", calibration=calib)
    frag = out["planned"][0]
    assert frag["route"] == "AMBIGUOUS" and frag["direction"] == "contradict"
    assert frag["strength"] == config.WRITE_CONTRADICT_HOLD > 1.0


# ══════════════════════════════════════════════════════════════════════════════
# APPLIER — reinforcement + idempotency at the store layer
# ══════════════════════════════════════════════════════════════════════════════
def _payload(episode_id, frags, reinforced=(), n_intra=0):
    return {"episode_id": episode_id, "ts": store.now_iso(), "memory_size": 0,
            "fragments": frags, "reinforced": list(reinforced), "n_intra_echo": n_intra}


def test_apply_fragmented_inserts_and_reinforces(conn):
    """The applier reconstructs text from the span and references the medoid sentence
    vector (no fragment-vector copy), then reinforces + stays idempotent."""
    ep = encode(conn, UID, NOTE, source="test")["episode_id"]
    fid = store.fragment_id_for(UID, ep, 0, 0)
    f = {"frag_id": fid, "sent_start": 0, "sent_end": 0, "medoid_idx": 0,
         "route": "NOVEL", "z": 1.2, "residual": 0.5, "weight": 1.0,
         "anchor_id": None, "direction": None, "is_centre": True, "is_novel_peak": True}
    p1 = _payload(ep, [f])
    with conn:
        write.apply_fragmented(conn, UID, p1)
    row = conn.execute("SELECT * FROM fragments WHERE id=?", (fid,)).fetchone()
    assert row["route"] == "NOVEL" and row["is_centre"] == 1 and row["strength"] == 1.0
    # text reconstructed from the span; vector referenced from the medoid sentence
    sents = store.episode_sentences_with_vectors(conn, UID, ep)
    assert row["text"] == sents[0]["text"] and row["medoid_idx"] == 0
    pool = {p["id"]: p["embedding"] for p in store.fragment_pool(conn, UID)}
    assert np.allclose(pool[fid], sents[0]["embedding"], atol=1e-5)

    # reinforce it (PREDICTED echo elsewhere) — strength + count bump, no new row
    with conn:
        write.apply_fragmented(conn, UID, _payload(ep, [], reinforced=[fid]))
    row = conn.execute("SELECT strength, reinforced FROM fragments WHERE id=?", (fid,)).fetchone()
    assert row["strength"] > 1.0 and row["reinforced"] == 1

    # re-applying the same insert is idempotent (deterministic id, ON CONFLICT)
    with conn:
        write.apply_fragmented(conn, UID, p1)
    assert conn.execute("SELECT COUNT(*) FROM fragments").fetchone()[0] == 1


# ══════════════════════════════════════════════════════════════════════════════
# INTEGRATION — refine_episode end-to-end (real local embedder + temp DB)
# ══════════════════════════════════════════════════════════════════════════════
NOTE = ("Spaced repetition is the most reliable way to retain knowledge over years. "
        "The brain consolidates memories during sleep, replaying the day's experiences. "
        "Without periodic review, even important insights decay into vague impressions.")


def test_refine_episode_persists_fragments_and_event(conn):
    receipt = encode(conn, UID, NOTE, title="memory", source="test")
    ep = receipt["episode_id"]
    r = write.refine_episode(conn, UID, ep)
    assert r["status"] == "ok" and r["n_fragments"] >= 1

    frags = conn.execute("SELECT * FROM fragments WHERE episode_id=? ORDER BY sent_start",
                         (ep,)).fetchall()
    assert len(frags) == r["n_fragments"]
    # every stored fragment is a real route, vectorised, and spans real sentences
    assert all(f["route"] in ("NOVEL", "AMBIGUOUS") for f in frags)
    # every fragment references a real medoid sentence (no separate fragment vector)
    n_sents = conn.execute("SELECT COUNT(*) FROM episode_sentences WHERE episode_id=?",
                           (ep,)).fetchone()[0]
    assert all(0 <= f["medoid_idx"] < n_sents for f in frags)
    assert len(store.fragment_pool(conn, UID)) == len(frags)
    assert sum(f["is_centre"] for f in frags) == 1
    assert sum(f["is_novel_peak"] for f in frags) == 1
    # FRAGMENTED event emitted + episode marked done
    evs = store.events_since(conn, UID, 0, types=["FRAGMENTED"])
    assert len(evs) == 1
    assert json.loads(evs[0]["payload_json"])["episode_id"] == ep
    assert store.unfragmented_episodes(conn, UID) == []


def test_refine_episode_is_idempotent(conn):
    ep = encode(conn, UID, NOTE, source="test")["episode_id"]
    write.refine_episode(conn, UID, ep)
    n_before = store.fragment_count(conn, UID)
    r2 = write.refine_episode(conn, UID, ep)
    assert r2["status"] == "skip"
    assert store.fragment_count(conn, UID) == n_before
    assert len(store.events_since(conn, UID, 0, types=["FRAGMENTED"])) == 1


def test_rebuild_reproduces_fragments(conn):
    from core.consolidate import rebuild
    ep = encode(conn, UID, NOTE, source="test")["episode_id"]
    write.refine_episode(conn, UID, ep)
    before = {r["id"]: (r["text"], r["route"], r["weight"], r["medoid_idx"])
              for r in conn.execute("SELECT * FROM fragments").fetchall()}

    rebuild(conn)   # truncates fragments + replays the FRAGMENTED event

    after = {r["id"]: (r["text"], r["route"], r["weight"], r["medoid_idx"])
             for r in conn.execute("SELECT * FROM fragments").fetchall()}
    assert after == before              # text + medoid reference reproduced exactly
    # each fragment's referenced vector still resolves it as its own nearest neighbour
    pool = {p["id"]: p["embedding"] for p in store.fragment_pool(conn, UID)}
    assert set(pool) == set(after)
    for fid, emb in pool.items():
        assert store.knn_fragments(conn, UID, emb, k=1)[0]["frag_id"] == fid


def test_refine_concurrent_claim_no_duplicate(conn, monkeypatch):
    """The TOCTOU guard: even if two refiners both pass the cheap pre-check (the
    window where neither has committed), the in-transaction mark_fragmented claim
    lets only ONE emit a FRAGMENTED event — no duplicate event, no double apply."""
    ep = encode(conn, UID, NOTE, source="test")["episode_id"]
    # Force both callers past the pre-filter, simulating the concurrent window.
    monkeypatch.setattr(write, "_existing_fragmented", lambda *a, **k: False)
    r1 = write.refine_episode(conn, UID, ep)
    n_after_first = store.fragment_count(conn, UID)
    r2 = write.refine_episode(conn, UID, ep)
    assert r1["status"] == "ok" and r2["status"] == "skip"
    assert r2["reason"] == "already_fragmented"
    assert store.fragment_count(conn, UID) == n_after_first          # no second apply
    assert len(store.events_since(conn, UID, 0, types=["FRAGMENTED"])) == 1


def test_refine_pending_retries_after_failure(conn, monkeypatch):
    """A transient failure mid-refine leaves the episode unfragmented (no partial
    write) and the next sweep completes it. Medoid selection removed the fragment
    re-embed, so the failure is injected at the routing step that still runs."""
    ep = encode(conn, UID, NOTE, source="test")["episode_id"]

    real, calls = write.route_fragments, {"n": 0}

    def flaky(*a, **k):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("transient")
        return real(*a, **k)

    monkeypatch.setattr(write, "route_fragments", flaky)

    reps = write.refine_pending(conn, UID)
    assert reps[0]["errors"] == 1
    assert store.unfragmented_episodes(conn, UID)            # still pending — no partial write
    assert store.fragment_count(conn, UID) == 0
    assert store.events_since(conn, UID, 0, types=["FRAGMENTED"]) == []

    reps2 = write.refine_pending(conn, UID)                  # retry now succeeds
    assert reps2[0]["refined"] == 1
    assert store.unfragmented_episodes(conn, UID) == []
    assert store.fragment_count(conn, UID) >= 1


def test_refine_pending_all_users_isolated(conn):
    a = encode(conn, UID, NOTE, source="test")["episode_id"]
    b = encode(conn, UID_B, NOTE, source="test")["episode_id"]
    assert set(store.users_with_unfragmented(conn)) == {UID, UID_B}
    write.refine_pending(conn, None)                         # sweep every user
    assert store.unfragmented_episodes(conn, UID) == []
    assert store.unfragmented_episodes(conn, UID_B) == []
    # each user's fragments belong only to their own episode
    assert all(r["episode_id"] == a for r in
               conn.execute("SELECT episode_id FROM fragments WHERE user_id=?", (UID,)))
    assert all(r["episode_id"] == b for r in
               conn.execute("SELECT episode_id FROM fragments WHERE user_id=?", (UID_B,)))


# ══════════════════════════════════════════════════════════════════════════════
# W6 PROD FIX — HFStance: contradictions survive with NO torch
# ══════════════════════════════════════════════════════════════════════════════
def test_hf_stance_buckets_entailment_prob(monkeypatch):
    from core.encode import HFStance

    def stub(shape):
        """Return a _post replacement emitting one of the router's live shapes."""
        def _post(self, premise, hypothesis):
            return shape
        return _post

    def bucket(shape):
        s = HFStance.__new__(HFStance)
        s._token, s._model = "t", "m"
        monkeypatch.setattr(HFStance, "_post", stub(shape))
        return s.classify("The sky is blue.", "The sky is not blue.")

    # DeBERTa MNLI shape: bare dict with parallel labels/scores. This is the one
    # huggingface_hub's typed helper rejects — the bug that made 'hf' a no-op.
    assert bucket({"sequence": "x", "labels": ["y"], "scores": [0.05]}) == "contradiction"
    assert bucket({"sequence": "x", "labels": ["y"], "scores": [0.92]}) == "entailment"
    assert bucket({"sequence": "x", "labels": ["y"], "scores": [0.40]}) == "neutral"
    # bart-large-mnli shape: list of {label, score}
    assert bucket([{"label": "y", "score": 0.05}]) == "contradiction"
    assert bucket([{"label": "y", "score": 0.92}]) == "entailment"


def test_hf_stance_raises_on_error_payload(monkeypatch):
    """An error body must raise, not be read as a 0.0 score (= false contradiction)."""
    from core.encode import HFStance

    s = HFStance.__new__(HFStance)
    s._token, s._model = "t", "m"
    monkeypatch.setattr(HFStance, "_post",
                        lambda self, p, h: {"error": "Model not supported by provider"})
    with pytest.raises(RuntimeError):
        s.classify("a", "b")


def test_hf_stance_posts_passthrough_template(monkeypatch):
    """hypothesis_template must stay '{}' — the hypothesis IS the candidate label,
    not a slot-filled 'This example is {}.' sentence."""
    from core import encode

    sent = {}

    class FakeResp:
        def raise_for_status(self): pass
        def json(self): return {"scores": [0.5]}

    def fake_post(url, headers=None, json=None, timeout=None):
        sent.update(url=url, headers=headers, body=json)
        return FakeResp()

    monkeypatch.setattr("requests.post", fake_post)
    s = encode.HFStance.__new__(encode.HFStance)
    s._token, s._model = "tok", "some/model"
    s._entail_prob("premise text", "hypothesis text")

    assert sent["body"]["parameters"]["hypothesis_template"] == "{}"
    assert sent["body"]["parameters"]["candidate_labels"] == ["hypothesis text"]
    assert sent["body"]["parameters"]["multi_label"] is True
    assert sent["body"]["inputs"] == "premise text"
    assert sent["headers"]["Authorization"] == "Bearer tok"
    assert "some/model" in sent["url"]


def test_hf_stance_retries_transient_then_succeeds(monkeypatch):
    """A router 503 must be retried, not degraded to neutral — an intermittent
    silent-neutral drops contradictions just as surely as a total outage."""
    from core import encode

    calls = []

    class Resp:
        def __init__(self, status):
            self.status_code = status

        def raise_for_status(self):
            if self.status_code >= 400:
                err = Exception(f"{self.status_code} Server Error")
                err.response = self
                raise err

        def json(self):
            return {"scores": [0.02]}

    def fake_post(url, **kw):
        calls.append(1)
        return Resp(503 if len(calls) < 3 else 200)

    monkeypatch.setattr("requests.post", fake_post)
    monkeypatch.setattr(config, "STANCE_HF_BACKOFF", 0.0)   # no real sleeping
    monkeypatch.setattr(config, "STANCE_HF_RETRIES", 3)
    s = encode.HFStance.__new__(encode.HFStance)
    s._token, s._model = "t", "m"
    assert s.classify("a", "b") == "contradiction"
    assert len(calls) == 3                                   # two 503s, then success


def test_hf_stance_does_not_retry_permanent_4xx(monkeypatch):
    """A 400 ('model not supported by provider') is permanent — fail fast."""
    from core import encode

    calls = []

    class Resp:
        status_code = 400

        def raise_for_status(self):
            err = Exception("400 Bad Request")
            err.response = self
            raise err

    def fake_post(url, **kw):
        calls.append(1)
        return Resp()

    monkeypatch.setattr("requests.post", fake_post)
    monkeypatch.setattr(config, "STANCE_HF_BACKOFF", 0.0)
    s = encode.HFStance.__new__(encode.HFStance)
    s._token, s._model = "t", "m"
    with pytest.raises(Exception):
        s.classify("a", "b")
    assert len(calls) == 1                                   # no retries burned


def test_classify_stance_hf_provider_signs_contradiction(monkeypatch):
    """The prod path: STANCE_PROVIDER='hf' detects a contradiction with no torch,
    and resolve_direction signs it −1 (held strongest)."""
    from core import encode

    class FakeStance:
        def classify(self, premise, hypothesis):
            return "contradiction"

    monkeypatch.setattr(config, "STANCE_PROVIDER", "hf")
    monkeypatch.setattr(encode, "_get_hf_stance", lambda: FakeStance())
    assert encode.classify_stance("anchor", "fragment") == "contradiction"
    assert predict.resolve_direction("fragment", "anchor")["sign"] == -1.0


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
