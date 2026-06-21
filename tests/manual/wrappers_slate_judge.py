"""Behavioural probe for the three wrappers on the REAL Slate corpus (engine.db),
for an LLM judge. No synthetic vectors, no hand-written sentences — actual stored
claims, concepts and episodes, with their stored MiniLM embeddings.

  SCAN     — segment a real multi-topic episode; judge cuts vs real topic shifts.
  ASSEMBLY — greedy non-redundant selection over a real concept cluster's members
             (dedupe) + a query-seeded retrieval over the whole corpus.
  GUARD    — per-member reconstruction-z within a real cluster: which claims are
             safe to forget (redundant) vs irreplaceable, vs the cluster baseline.

Run: PYTHONPATH=. .venv/bin/python tests/manual/wrappers_slate_judge.py
"""
import sqlite3
import numpy as np
import sqlite_vec
from sentence_transformers import SentenceTransformer

from core import scan, assembly, guard, predict

# ── load the real corpus ────────────────────────────────────────────────────────
c = sqlite3.connect("data/engine.db"); c.row_factory = sqlite3.Row
c.enable_load_extension(True); sqlite_vec.load(c); c.enable_load_extension(False)
cl, lbl = {}, {}
for r in c.execute("SELECT claim_id, concept_id FROM concept_members"):
    cl.setdefault(r["claim_id"], r["concept_id"])
for r in c.execute("SELECT id,label FROM concepts"):
    lbl[r["id"]] = r["label"]
corpus = []
for r in c.execute("SELECT c.id,c.text,v.embedding FROM claims c "
                   "JOIN vec_claims v ON v.claim_id=c.id"):
    corpus.append({"id": r["id"], "text": r["text"], "cluster": cl.get(r["id"]),
                   "embedding": np.frombuffer(r["embedding"], dtype=np.float32).astype(float)})
C = np.vstack([x["embedding"] for x in corpus])
by_cluster = {}
for i, x in enumerate(corpus):
    if x["cluster"]:
        by_cluster.setdefault(x["cluster"], []).append(i)
baselines = predict.compute_baselines(corpus, C)

_m = SentenceTransformer("all-MiniLM-L6-v2")
def embed(t):
    return np.atleast_2d(np.asarray(_m.encode(list(t), normalize_embeddings=True,
                                               show_progress_bar=False), dtype=float))
def section(s): print("\n" + "=" * 80 + "\n" + s + "\n" + "=" * 80)
print(f"corpus: {len(corpus)} claims / {len(by_cluster)} clusters")


# ════════════════════════════════════════════════════════════════════════════
# SCAN — segment a real multi-topic episode
# ════════════════════════════════════════════════════════════════════════════
section("SCAN — segment a real episode at its topic shifts")
# A mid-size synthesis note (multiple sources/sections) → real topic boundaries.
ep = c.execute("SELECT id,title,raw_text FROM episodes "
               "WHERE length(raw_text) BETWEEN 1500 AND 6000 "
               "ORDER BY length(raw_text) DESC LIMIT 1").fetchone()
sents = predict._split(ep["raw_text"])[:26]   # cap for readable judging
E = embed(sents)
cuts = scan.boundaries(E)
curve = scan.residual_curve(E)
print(f"episode {ep['id']}  title={ep['title']!r}  ({len(sents)} sentences shown)")
print(f"cut indices: {cuts}\n")
for gi, seg in enumerate(scan.segment(E)):
    print(f"  ── segment {gi} ──")
    for i in seg:
        print(f"     [{i:2}] sim={curve[i]:.2f} | {sents[i][:88]}")


# ════════════════════════════════════════════════════════════════════════════
# ASSEMBLY — non-redundant selection over a real cluster + query retrieval
# ════════════════════════════════════════════════════════════════════════════
section("ASSEMBLY — greedy non-redundant pick over a real cluster's members")
big = max(by_cluster, key=lambda k: len(by_cluster[k]))
idxs = by_cluster[big]
cand = C[idxs]
out = assembly.assemble(cand, k=8, calibration={"gain_floor": 0.45})
print(f"cluster {lbl.get(big,big)!r} — {len(idxs)} members; "
      f"assemble picked {len(out['chosen'])}, stopped_on_gain={out['stopped']}\n")
print("  CHOSEN (most marginal info first):")
for rank, (j, g, s) in enumerate(zip(out["chosen"], out["gains"], out["shares"])):
    print(f"   #{rank+1} gain={g:.2f} share={s:.2f} | {corpus[idxs[j]]['text'][:84]}")
print("\n  A FEW SKIPPED (should read as redundant with the chosen set):")
for j in [k for k in range(len(idxs)) if k not in out["chosen"]][:6]:
    print(f"      | {corpus[idxs[j]]['text'][:84]}")

section("ASSEMBLY — query-seeded retrieval over the whole corpus")
for q in ["Why do guests book goSTOPS on impulse?",
          "What is wrong with our membership and how do we fix referrals?"]:
    qe = embed([q])
    rel = (C @ qe.T).max(axis=1)
    pool = [i for i in np.argsort(-rel)[:40] if rel[i] >= 0.3]
    out = assembly.assemble(C[pool], seed=qe, k=6, calibration={"gain_floor": 0.4})
    print(f"\n  Q: {q}\n     pool={len(pool)} relevant claims; "
          f"assembled {len(out['chosen'])}, stopped={out['stopped']}")
    for rank, (j, g) in enumerate(zip(out["chosen"], out["gains"])):
        i = pool[j]
        print(f"     #{rank+1} gain={g:.2f} rel={rel[i]:.2f} | {corpus[i]['text'][:80]}")


# ════════════════════════════════════════════════════════════════════════════
# GUARD — safe-forget / protect within real clusters (reconstruction-z)
# ════════════════════════════════════════════════════════════════════════════
section("GUARD — guard.forget() over real clusters (thin iterator over measure())")
# guard.forget measures each member leave-one-out against its own cluster (one
# measure() call), reading the z field; the cluster's own spread sets the bar.
for cid in sorted(by_cluster, key=lambda k: -len(by_cluster[k]))[:3]:
    idxs = by_cluster[cid]
    if len(idxs) < 6:
        continue
    members = [corpus[j] for j in idxs]
    res = sorted(guard.forget(members), key=lambda m: m["z"])
    print(f"\n  cluster {lbl.get(cid,cid)!r}  ({len(idxs)} members)")
    print("   SAFE TO FORGET (rebuildable from the rest, z≪0):")
    for m in res[:3]:
        print(f"      z={m['z']:+.2f} res={m['residual']:.2f} drop={m['safe_to_drop']} | {m['text'][:78]}")
    print("   PROTECT (irreplaceable, z≫0):")
    for m in res[-3:]:
        print(f"      z={m['z']:+.2f} res={m['residual']:.2f} drop={m['safe_to_drop']} | {m['text'][:78]}")
