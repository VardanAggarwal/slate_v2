"""Focus: the AMBIGUOUS band is the linchpin (store + flag for resolve_direction).
resolve_direction (contradiction detection) ONLY runs on to_resolve, which only
gets AMBIGUOUS verdicts. If contradictions don't land AMBIGUOUS, the whole
contradiction/versioning path is dead. Quantify where the band actually fires.
"""
import sqlite3, sqlite_vec, numpy as np
from core import predict
from sentence_transformers import SentenceTransformer

c = sqlite3.connect("data/engine.db"); c.row_factory = sqlite3.Row
c.enable_load_extension(True); sqlite_vec.load(c); c.enable_load_extension(False)
cl = {}
for r in c.execute("SELECT claim_id, concept_id FROM concept_members"):
    cl.setdefault(r["claim_id"], r["concept_id"])
corpus = []
for r in c.execute("SELECT c.id,c.text,v.embedding FROM claims c JOIN vec_claims v ON v.claim_id=c.id"):
    corpus.append({"id": r["id"], "text": r["text"], "cluster": cl.get(r["id"]),
                   "embedding": np.frombuffer(r["embedding"], dtype=np.float32).astype(float)})
_m = SentenceTransformer("all-MiniLM-L6-v2")
def embed(t): return np.atleast_2d(np.asarray(_m.encode(list(t), normalize_embeddings=True, show_progress_bar=False), dtype=float))
base = predict.compute_baselines(corpus)

# 1) Route distribution when we push EVERY real claim back in (leave-it-in dup).
from collections import Counter
ms = predict.measure([c["text"] for c in corpus][0], corpus, embed=embed, baselines=base)  # warm
routes = Counter(); zs = []
texts = [c["text"] for c in corpus]
embs = embed(texts)
C = np.vstack([c["embedding"] for c in corpus])
for t, e in zip(texts, embs):
    m = predict._measure_one(t, e, corpus, C, base)
    d = predict.decide([m])["fragments"][0]
    routes[d["route"]] += 1; zs.append(m["z"])
print("Re-pushing all 568 real claims verbatim → route counts:", dict(routes))
print(f"  z range {min(zs):+.2f}..{max(zs):+.2f}  median {np.median(zs):+.2f}")

# 2) Sweep: blend a claim's own vector with its nearest-different neighbour to
#    walk similarity down from 1.0, and record the route at each step. Shows the
#    sim-window in which AMBIGUOUS *can* fire at all.
print("\nSim-sweep on one claim (blend toward a far vector), z_reinforce=0.5 z_new=2.0:")
probe = next(c for c in corpus if c["cluster"])
own = probe["embedding"]
far = corpus[int(np.argmin(C @ own))]["embedding"]
print(f"  probe: {probe['text'][:60]!r}")
for a in [1.0,0.95,0.9,0.8,0.7,0.6,0.5,0.4,0.3,0.2,0.1,0.0]:
    v = a*own + (1-a)*far; v = v/np.linalg.norm(v)
    m = predict._measure_one("x", v, corpus, C, base)
    d = predict.decide([m])["fragments"][0]
    print(f"   blend={a:.2f} sim={m['nearest_sim']:.3f} z={m['z']:+.2f} "
          f"attached={d.get('resolve') or d['route']=='PREDICTED'} -> {d['route']}")

# 3) Many manual contradiction pairs against real claims: how many reach resolve?
pairs = [
 ("Democracy's equality principle is fundamentally false",
  "Democracy's equality principle is fundamentally true and just."),
 ("Modern humans represent negatively evolved beings",
  "Modern humans represent positively evolved, advanced beings."),
 ("Citizens blindly follow systems without questioning their origins",
  "Citizens rigorously question and challenge the systems they live under."),
 ("Spacetime is curved by the presence of mass",
  "Spacetime is perfectly flat and unaffected by mass."),
 ("System-creators receive no societal recognition",
  "System-creators are richly rewarded and celebrated by society."),
]
print("\nContradiction pairs — does each reach to_resolve (AMBIGUOUS)?")
n_resolve = 0
for anchor, contra in pairs:
    r = predict.prediction_error(contra, corpus, embed=embed, baselines=base)
    f = r["fragments"][0]
    n_resolve += len(r["to_resolve"])
    print(f"   z={f['z']:+.2f} sim={f['nearest_sim']:.3f} route={f['route']:9} "
          f"resolve={f['resolve']} | {contra[:45]!r}")
print(f"  => {n_resolve}/{len(pairs)} contradictions routed to resolve_direction.")
print("  If 0, contradiction detection via the predictor path NEVER triggers.")
