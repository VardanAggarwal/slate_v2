"""Query-conditioned consolidation — "encode retrieval queries in memory".

HYPOTHESIS (user): a single retrieval lights up only the geometry-nearest region. If we
ENCODE the queries themselves as a usage signal during consolidation, concepts that keep
co-firing under the same questions — even when far apart in embedding space — can be wired
together. Next retrieval then spreads across those usage-bridges and lights up things a
single geometric shot would miss. This is the lever for `broad` (cross-note synthesis),
which is the documented hard wall.

WHY IT MIGHT BEAT THE INERT GEOMETRIC BRIDGES: geometric bridges connect already-near
nodes (resonance already reaches those by spreading). Usage-bridges connect STRUCTURAL
HOLES — pairs that are geometrically distant (cos < GEO_MAX) yet answer the same question.
That signal cannot come from geometry; it only comes from queries.

HONESTY: gold is eval-only. The query stream is SYNTHETIC, generated from concept labels
(blind to gold) and cached to qbridge_qstream.json. A/B is on the frozen golds.

Run:  PYTHONPATH=.:scratchpad .venv/bin/python scratchpad/query_bridges.py
"""
from __future__ import annotations

import json
import random
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np

import os
from core import config, store, llm, resonance, calibration as calib, retrieve

# Use the LOCAL Claude (subscription, keychain login) for query generation — skip gemini.
# Gate (llm.call) needs config.CLAUDE_CODE_OAUTH_TOKEN truthy to *attempt* claude-cli, but
# the subprocess must NOT inherit a token or it overrides the keychain — so pop it from env.
config.CLAUDE_CODE_OAUTH_TOKEN = "use-keychain"
os.environ.pop("CLAUDE_CODE_OAUTH_TOKEN", None)
config.LLM_FALLBACK_ORDER = ["claude-cli"]
from eval.coverage import coverage_eval, _load_gold, HERE
from eval_both import GOLD, TAU

U = "usr_01KTXAYR20J4R6F7PT3DP10W3W"
DB_SRC = "/tmp/slate_eval.db"
QSTREAM = Path(__file__).parent / "qbridge_qstream.json"

# bridge-build knobs
N_QUERIES = 60          # synthetic query stream size
TOPK = 8                # concepts considered "co-activated" per query (by salience)
GEO_MAX = 0.45          # only bridge pairs FARTHER than this (structural holes)
MIN_SUPPORT = 2         # a pair must co-fire under >= this many distinct queries
BRIDGE_W = 0.7          # edge weight written (resonance caps at 1.0)
MAX_BRIDGES = 200       # cap total bridges added (avoid hubifying the graph)


def fresh_conn(tag="qb"):
    import os, shutil, tempfile
    fd, path = tempfile.mkstemp(prefix=f"slate_{tag}_", suffix=".db", dir=tempfile.gettempdir())
    os.close(fd)
    shutil.copyfile(DB_SRC, path)
    return store.connect(path), path


def concept_rows(conn):
    return conn.execute("SELECT id, label FROM concepts WHERE user_id=?", (U,)).fetchall()


# ── synthetic query stream (blind to gold, cached) ──────────────────────────────
def build_qstream(conn) -> list[str]:
    if QSTREAM.exists():
        return json.loads(QSTREAM.read_text())
    labels = [r["label"] for r in concept_rows(conn) if r["label"] and r["label"] != "(seed)"]
    rng = random.Random(42)
    # sample concept-label PAIRS → ask for a natural cross-topic question a reader might ask.
    # cross-topic on purpose: that's the broad-synthesis demand we want to encode.
    pairs = []
    for _ in range(N_QUERIES):
        a, b = rng.sample(labels, 2)
        pairs.append((a, b))
    queries: list[str] = []
    B = 6
    for i in range(0, len(pairs), B):
        chunk = pairs[i:i + B]
        listing = "\n".join(f"{j+1}. {a}  +  {b}" for j, (a, b) in enumerate(chunk))
        prompt = (
            "These are pairs of topics from a person's notebook. For each pair, write ONE "
            "natural question that person might later ask which would require BOTH topics to "
            "answer well. Keep each question under 20 words, concrete.\n\n"
            f"{listing}\n\n"
            "Output ONLY the questions, one per line, no numbering, no preamble.")
        res = llm.call(prompt, tier="judgment", max_tokens=1500, json_out=False)
        # robust to JSON-array / numbered / line formats: grab any ?-terminated clause
        import re
        for m in re.findall(r'([A-Z][^"\n?]{8,}\?)', res["text"]):
            queries.append(m.strip())
    QSTREAM.write_text(json.dumps(queries, indent=2))
    return queries


# ── co-activation harvest → usage-bridges ───────────────────────────────────────
def harvest_bridges(conn, queries: list[str]):
    C = calib.merged(conn, {**retrieve.DEFAULT_CALIBRATION, **resonance.DEFAULT_CALIBRATION}, U)
    emb_cache: dict[str, np.ndarray] = {}

    def cemb(cid):
        if cid not in emb_cache:
            v = store.concept_embedding(conn, U, cid)
            emb_cache[cid] = None if v is None else np.asarray(v, float) / (np.linalg.norm(v) + 1e-9)
        return emb_cache[cid]

    pair_support = defaultdict(int)     # distinct queries a pair co-fired in
    pair_weight = defaultdict(float)    # accumulated min-salience
    for q in queries:
        field = resonance.activate(conn, U, q, calibration=C)
        cps = sorted(((n, d["salience"]) for n, d in field["nodes"].items() if n.startswith("cpt_")),
                     key=lambda kv: -kv[1])[:TOPK]
        for (a, sa), (b, sb) in combinations(cps, 2):
            ea, eb = cemb(a), cemb(b)
            if ea is None or eb is None:
                continue
            if float(ea @ eb) >= GEO_MAX:      # too near — geometry already reaches it
                continue
            key = tuple(sorted((a, b)))
            pair_support[key] += 1
            pair_weight[key] += min(sa, sb)
    cand = [(k, pair_weight[k]) for k in pair_support if pair_support[k] >= MIN_SUPPORT]
    cand.sort(key=lambda kv: -kv[1])
    return cand[:MAX_BRIDGES]


def write_bridges(conn, bridges):
    with conn:
        for (a, b), _w in bridges:
            store.insert_relation(conn, U, a, b, "query_bridge", BRIDGE_W, "2026-06-27")


# ── eval ────────────────────────────────────────────────────────────────────────
def evalrun(conn, tag):
    golds = {g: _load_gold(HERE / f) for g, f in GOLD.items()}
    cov = {g: round(coverage_eval(gd, conn, U, resonance.resonance_context, TAU)["sr_at_b"], 4)
           for g, gd in golds.items()}
    cov["mean"] = round(sum(cov.values()) / 3, 4)
    print(f"{tag:24} narrow={cov['narrow']:.3f}  broad={cov['broad']:.3f}  "
          f"paragraph={cov['paragraph']:.3f}  mean={cov['mean']:.3f}")
    return cov


def main():
    conn, path = fresh_conn()
    print(f"work db: {path}")
    queries = build_qstream(conn)
    print(f"qstream: {len(queries)} synthetic queries (cached at {QSTREAM.name})")
    base = evalrun(conn, "BASELINE (as-is)")
    bridges = harvest_bridges(conn, queries)
    print(f"harvested {len(bridges)} usage-bridges "
          f"(geo<{GEO_MAX}, support>={MIN_SUPPORT}); sample weights: "
          f"{[round(w,2) for _,w in bridges[:5]]}")
    write_bridges(conn, bridges)
    treat = evalrun(conn, "TREATMENT (+q-bridges)")
    print("\nΔ  narrow={:+.3f}  broad={:+.3f}  paragraph={:+.3f}  mean={:+.3f}".format(
        treat["narrow"] - base["narrow"], treat["broad"] - base["broad"],
        treat["paragraph"] - base["paragraph"], treat["mean"] - base["mean"]))


if __name__ == "__main__":
    main()
