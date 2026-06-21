"""Spine tests for the predictor (execution plan §5.1, P1 exit): the direct
`measure()`/`decide()` paths the wrappers only exercise indirectly.

Synthetic embeddings (near-orthogonal topic directions + small noise) make the
structure — anchors, near-duplicates, fresh-but-attached, off-topic — known
exactly, so routing is checkable without an LLM or the live DB. As in the wrapper
tests, the INSTRUMENT (residual/z ordering) is asserted directly, and POLICY
(the route) is asserted under a calibration fitted to the fixture — never a baked
constant — mirroring how consolidation fits z_echo per corpus.
"""
import numpy as np
import pytest

from core import predict

DIM = 48
SEED = 7


def _topic_dirs(n, rng):
    Q, _ = np.linalg.qr(rng.standard_normal((n, DIM)).T)
    return Q.T[:n]


def _frag(direction, rng, noise=0.04):
    v = direction + noise * rng.standard_normal(DIM)
    return v / np.linalg.norm(v)


def _rows(vecs, cluster, start=0):
    return [{"id": start + i, "text": f"{cluster}{start + i}",
             "embedding": v, "cluster": cluster} for i, v in enumerate(vecs)]


def _probe(vec, text="p"):
    return [{"text": text, "embedding": vec}]


def _corpus_AB(rng):
    """Two cohesive clusters A (topic0) and B (topic1), 8 members each."""
    dirs = _topic_dirs(3, rng)              # 0,1 used for corpus; 2 reserved off-topic
    A = _rows([_frag(dirs[0], rng) for _ in range(8)], "A")
    B = _rows([_frag(dirs[1], rng) for _ in range(8)], "B", start=8)
    return dirs, A + B


# ── measure(): pure sensor ──────────────────────────────────────────────────────
def test_measure_anchor_and_residual():
    rng = np.random.default_rng(SEED)
    dirs, corpus = _corpus_AB(rng)
    near = _frag(dirs[0], rng, noise=0.005)          # near-dup of an A member
    off = _frag(dirs[2], rng)                        # off-topic, not in corpus
    m_near = predict.measure(_probe(near), corpus)[0]
    m_off = predict.measure(_probe(off), corpus)[0]
    assert m_near["anchor_id"] < 8                   # anchored into cluster A
    assert m_near["residual"] < m_off["residual"]    # near-dup reconstructs better
    assert m_near["nearest_sim"] > m_off["nearest_sim"]
    assert not m_near["cold_start"] and not m_off["cold_start"]


def test_measure_cold_start_below_warmup():
    rng = np.random.default_rng(SEED)
    d = _topic_dirs(1, rng)[0]
    corpus = _rows([_frag(d, rng)], "A")             # |Y| = 1 < WARMUP_MIN_CORPUS
    m = predict.measure(_probe(_frag(d, rng)), corpus)[0]
    assert m["cold_start"] and m["anchor_id"] is None and m["residual"] == 1.0


def test_measure_z_orders_by_novelty():
    """The instrument, before any policy: near-dup ≪ fresh-same-topic ≪ off-topic."""
    rng = np.random.default_rng(SEED)
    dirs, corpus = _corpus_AB(rng)
    z_near = predict.measure(_probe(_frag(dirs[0], rng, noise=0.005)), corpus)[0]["z"]
    z_fresh = predict.measure(_probe(_frag(dirs[0], rng, noise=0.03)), corpus)[0]["z"]
    z_off = predict.measure(_probe(_frag(dirs[2], rng)), corpus)[0]["z"]
    assert z_near < z_fresh < z_off, (z_near, z_fresh, z_off)


# ── decide(): the route matrix (W6) ──────────────────────────────────────────────
def test_decide_route_matrix():
    """PREDICTED / AMBIGUOUS / NOVEL over labeled fixtures. z-ordering is the
    instrument; the route falls out under a z_echo fitted BETWEEN the near-dup and
    the fresh-attached fragment — exactly what consolidation fits per corpus."""
    rng = np.random.default_rng(SEED)
    dirs, corpus = _corpus_AB(rng)
    near = _frag(dirs[0], rng, noise=0.005)          # PREDICTED: near-identical, attached
    fresh = _frag(dirs[0], rng, noise=0.03)          # AMBIGUOUS: same topic, looser, attached
    off = _frag(dirs[2], rng)                         # NOVEL: off-topic, no anchor present
    ms = predict.measure([{"text": "near", "embedding": near},
                          {"text": "fresh", "embedding": fresh},
                          {"text": "off", "embedding": off}], corpus)
    z_near, z_fresh = ms[0]["z"], ms[1]["z"]
    assert z_near < z_fresh                           # instrument separates them
    cut = (z_near + z_fresh) / 2                      # fitted gate
    out = predict.decide(ms, {"z_echo": cut, "prox_margin": predict.PROX_MARGIN})
    routes = [f["route"] for f in out["fragments"]]
    assert routes == ["PREDICTED", "AMBIGUOUS", "NOVEL"], routes
    # and the decision aggregates line up with the routes
    assert out["reinforced"] == [ms[0]["anchor_id"]]   # PREDICTED → reinforce anchor
    assert out["new"] == ["off"]                        # NOVEL → encode
    assert [r["fragment"] for r in out["to_resolve"]] == ["fresh"]  # AMBIGUOUS only


def test_decide_only_ambiguous_routes_to_resolver():
    """Invariant: ONLY AMBIGUOUS reaches the LLM. `to_resolve` is the resolver's
    work queue; assert exactly one entry per AMBIGUOUS verdict and none else."""
    rng = np.random.default_rng(SEED)
    dirs, corpus = _corpus_AB(rng)
    ms = predict.measure([{"text": "near", "embedding": _frag(dirs[0], rng, 0.005)},
                          {"text": "fresh", "embedding": _frag(dirs[0], rng, 0.03)},
                          {"text": "off", "embedding": _frag(dirs[2], rng)}], corpus)
    cut = (ms[0]["z"] + ms[1]["z"]) / 2
    out = predict.decide(ms, {"z_echo": cut})
    n_ambiguous = sum(f["route"] == "AMBIGUOUS" for f in out["fragments"])
    assert len(out["to_resolve"]) == n_ambiguous == 1
    assert all(f["resolve"] == (f["route"] == "AMBIGUOUS") for f in out["fragments"])


def test_decide_weights_are_a_within_batch_share():
    """Stored fragments carry a residual share summing to 1; reinforced (stored
    nothing) carry weight 0 and never dilute the share."""
    rng = np.random.default_rng(SEED)
    dirs, corpus = _corpus_AB(rng)
    ms = predict.measure([{"text": "near", "embedding": _frag(dirs[0], rng, 0.005)},
                          {"text": "fresh", "embedding": _frag(dirs[0], rng, 0.03)},
                          {"text": "off", "embedding": _frag(dirs[2], rng)}], corpus)
    cut = (ms[0]["z"] + ms[1]["z"]) / 2
    out = predict.decide(ms, {"z_echo": cut})
    stored = [f for f in out["fragments"] if f["store"]]
    assert abs(sum(f["weight"] for f in stored) - 1.0) < 1e-2
    assert all(f["weight"] == 0.0 for f in out["fragments"] if not f["store"])


def test_decide_cold_start_stores_as_novel():
    rng = np.random.default_rng(SEED)
    d = _topic_dirs(1, rng)[0]
    corpus = _rows([_frag(d, rng)], "A")             # below warmup → cold_start
    out = predict.decide(predict.measure(_probe(_frag(d, rng)), corpus))
    f = out["fragments"][0]
    assert f["route"] == "NOVEL" and f["store"] and not f["resolve"]


def test_decide_ranking_most_surprising_first():
    rng = np.random.default_rng(SEED)
    dirs, corpus = _corpus_AB(rng)
    ms = predict.measure([{"text": "near", "embedding": _frag(dirs[0], rng, 0.005)},
                          {"text": "off", "embedding": _frag(dirs[2], rng)}], corpus)
    out = predict.decide(ms, {"z_echo": -99})         # routing irrelevant to ranking
    # off-topic (idx 1) is the most surprising → ranked first
    assert out["ranking"][0] == 1


# ── resolve_direction(): the LLM layer, direction only ───────────────────────────
def test_resolve_direction_signs():
    """Direction is the resolver's, not the predictor's. Inject a stub classifier
    so the test is deterministic and LLM-free."""
    contra = predict.resolve_direction("x", "y", classify=lambda p, h: "contradiction")
    entail = predict.resolve_direction("x", "y", classify=lambda p, h: "entailment")
    neutral = predict.resolve_direction("x", "y", classify=lambda p, h: "neutral")
    assert contra == {"direction": "contradict", "sign": -1.0}
    assert entail["sign"] == +1.0 and entail["direction"] == "reinforce"
    assert neutral["sign"] == +1.0 and neutral["direction"] == "refine"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
