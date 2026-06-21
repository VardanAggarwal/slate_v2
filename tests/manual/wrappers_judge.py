"""Real-text behavioural probe for the three predictor wrappers, for an LLM judge.

Unlike tests/test_wrappers.py (synthetic orthogonal vectors → exact geometry),
this runs the wrappers on REAL English sentences embedded by the real encoder
(all-MiniLM-L6-v2), and prints structured output so a human/LLM can judge whether
each wrapper did its JOB — not just its math. No assertions; judgement is external.

Run: .venv/bin/python tests/manual/wrappers_judge.py
"""
import json
import numpy as np
from sentence_transformers import SentenceTransformer

from core import scan, assembly, guard

_m = SentenceTransformer("all-MiniLM-L6-v2")
def embed(texts):
    return np.atleast_2d(np.asarray(
        _m.encode(list(texts), normalize_embeddings=True, show_progress_bar=False),
        dtype=float))

def section(s): print("\n" + "=" * 78 + "\n" + s + "\n" + "=" * 78)
def dump(obj): print(json.dumps(obj, indent=2, ensure_ascii=False))


# ════════════════════════════════════════════════════════════════════════════
# SCAN — does it cut a multi-topic note where the topic actually shifts?
# ════════════════════════════════════════════════════════════════════════════
section("SCAN  (core/scan.py) — segment real notes at topic shifts")

scan_cases = [
    {"name": "3 clean topics (cooking / finance / astronomy)",
     "expect": "boundaries before sentence 3 and before sentence 6",
     "sents": [
        "I sauteed the garlic in olive oil until it turned golden.",
        "Then I added the chopped tomatoes and let the sauce simmer.",
        "A diversified index fund usually beats picking individual stocks.",
        "Compound interest rewards you for starting to invest early.",
        "The Andromeda galaxy is on a collision course with the Milky Way.",
        "Light from distant stars can take billions of years to reach us."]},
    {"name": "gradual drift within one topic (no shift)",
     "expect": "no cuts — all sentences are about the same morning routine",
     "sents": [
        "I wake up at six and drink a glass of water first thing.",
        "After that I stretch for ten minutes to loosen up.",
        "Then I make a strong cup of coffee and sit by the window.",
        "I usually read the news while the coffee cools down.",
        "By seven I am at my desk ready to start work."]},
    {"name": "two topics, abrupt switch mid-note",
     "expect": "one boundary before the sentence that switches to the dog",
     "sents": [
        "The quarterly report shows revenue grew twelve percent.",
        "Most of that growth came from the enterprise segment.",
        "Margins held steady despite rising cloud costs.",
        "My dog learned to fetch the newspaper this week.",
        "He drops it by the door and waits for a treat."]},
]
for c in scan_cases:
    E = embed(c["sents"])
    cuts = scan.boundaries(E)
    curve = scan.residual_curve(E)
    segs = scan.segment(E)
    print(f"\n— {c['name']}")
    print(f"  EXPECT: {c['expect']}")
    print(f"  residual_curve: {[round(x,2) for x in curve]}")
    print(f"  cut indices (segment starts): {cuts}")
    for gi, seg in enumerate(segs):
        print(f"  segment {gi}: " + " | ".join(c["sents"][i][:38] for i in seg))


# ════════════════════════════════════════════════════════════════════════════
# ASSEMBLY — does it pick the most informative, non-redundant set and stop?
# ════════════════════════════════════════════════════════════════════════════
section("ASSEMBLY  (core/assembly.py) — greedy max-info selection + STOP")

assembly_cases = [
    {"name": "3 distinct facts + 3 paraphrase-duplicates of the first",
     "expect": "picks the 3 distinct facts; skips the paraphrases; stops early",
     "items": [
        "The Eiffel Tower is in Paris.",                       # A
        "Photosynthesis converts sunlight into chemical energy.",  # B
        "The human heart has four chambers.",                  # C
        "Paris is home to the Eiffel Tower.",                  # A'
        "You can find the Eiffel Tower in the French capital.",# A''
        "The famous iron tower of Paris is the Eiffel Tower."],# A'''
     "seed": None},
    {"name": "query-seeded: assembly already covers part of the question",
     "expect": "skips items the seed already covers; pulls genuinely new angles",
     "items": [
        "Exercise improves cardiovascular health.",
        "Regular workouts strengthen the heart and lungs.",   # ~ seed-covered
        "A balanced diet is also key to staying healthy.",    # new angle
        "Sleep is essential for muscle recovery and focus."], # new angle
     "seed": ["How does physical exercise benefit the heart?"]},
]
for c in assembly_cases:
    cand = embed(c["items"])
    seed = embed(c["seed"]) if c["seed"] else None
    out = assembly.assemble(cand, seed=seed, k=6, calibration={"gain_floor": 0.35})
    print(f"\n— {c['name']}")
    print(f"  EXPECT: {c['expect']}")
    if c["seed"]: print(f"  seed: {c['seed'][0]!r}")
    print(f"  stopped_on_gain={out['stopped']}  chose {len(out['chosen'])}/{len(c['items'])}")
    for rank, (i, g, s) in enumerate(zip(out["chosen"], out["gains"], out["shares"])):
        print(f"   #{rank+1} gain={g:.2f} share={s:.2f} | {c['items'][i]!r}")
    skipped = [i for i in range(len(c["items"])) if i not in out["chosen"]]
    for i in skipped:
        print(f"   skipped | {c['items'][i]!r}")


# ════════════════════════════════════════════════════════════════════════════
# GUARD — does it call redundant claims safe-to-drop and protect unique nuance?
# ════════════════════════════════════════════════════════════════════════════
section("GUARD  (core/guard.py) — safe-forget / safe-merge on real claims")

guard_cases = [
    {"name": "forget: redundant restatement vs a unique fact",
     "expect": "the restatement is safe_to_drop=True; the unique fact False",
     "remaining": [
        "Water boils at 100 degrees Celsius at sea level.",
        "At sea level, water reaches its boiling point at 100 C.",
        "Pure water boils at one hundred degrees Celsius under normal pressure.",
        "Atmospheric pressure affects the temperature at which water boils."],
     "victims": [
        ("redundant", "Water's boiling point is 100 degrees Celsius at sea level."),
        ("unique",    "Salt raises the boiling point of water slightly.")]},
    {"name": "merge: which absorbed members carry nuance the survivors lack",
     "expect": "the plain paraphrase folds (True); the one adding a caveat keeps (False)",
     "remaining": [
        "Remote work increases employee productivity.",
        "Working from home boosts how much employees get done.",
        "Studies show productivity rises when people work remotely."],
     "victims": [
        ("paraphrase", "Employees are more productive when they work remotely."),
        ("nuance",     "Remote work raises productivity only for self-directed roles.")]},
]
for c in guard_cases:
    print(f"\n— {c['name']}")
    print(f"  EXPECT: {c['expect']}")
    print(f"  survivor set ({len(c['remaining'])} claims): "
          + " | ".join(s[:34] for s in c["remaining"]))
    survivors = [{"id": f"s{i}", "text": s, "embedding": embed([s])[0], "cluster": "c"}
                 for i, s in enumerate(c["remaining"])]
    losers = [{"id": kind, "text": t, "embedding": embed([t])[0]}
              for kind, t in c["victims"]]
    for (kind, t), m in zip(c["victims"], guard.merge(losers, survivors)):
        print(f"   [{kind:9}] residual={m['residual']:.3f} z={m['z']:+.2f} "
              f"safe_to_drop={m['safe_to_drop']} | {t!r}")
