"""Deterministic Coverage@B — replace the LLM judge (and the answerer) with
embedding cosine of the PRE-REGISTERED key_facts against retrieved context.

The gold already saves what must be covered (`key_facts`); the only thing the LLM
adds today is the *matching* step ("is fact #i covered in this text?"). Swap that
for cosine over the MiniLM vectors we already use everywhere. No answerer, no judge.

Two entry points:

  calibrate — fit τ against the logged judge runs (eval/baseline_*.json). Uses the
              EXACT judge inputs: for each (key_fact, answer) the judge labeled,
              score = max cosine(fact, answer_chunk); sweep τ; report the τ whose
              cover@τ best agrees with the judge (fact-level + query-level). This is
              the "report the τ that best matches the current judge" deliverable.

  eval      — Coverage@B over a gold file using a context fn (NO LLM): score each
              key_fact vs the ASSEMBLED CONTEXT (not a generated answer); a query
              passes iff every fact is covered. Same report shape as harness.run_eval
              so it drops into the existing comparison tooling.

Note on the answer→context shift: calibration matches facts against the *answer*
(that's what the judge scored, so the labels are valid). The shipped metric matches
facts against the *context* — a superset of what the answer expressed, so τ transfers
as a slightly conservative bound. Re-validate τ periodically against a fresh judge run.
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
from pathlib import Path

import numpy as np

from core import store
from core.encode import get_embedder

CHARS_PER_TOKEN = 4
DEFAULT_BUDGET_TOKENS = 2000
HERE = Path(__file__).parent

# Context-only assemblers (mirror harness._ANSWERERS, minus the answerer LLM).
from core.recall import assemble_context as _slate_ctx
from core.retrieve import assemble_context as _frag_ctx
from core.hybrid import hybrid_context as _hybrid_ctx
from core.hierarchical import hierarchical_context as _hier_ctx
from core.resonance import resonance_context as _resonance_ctx


def _grep_ctx(conn, user_id, query, max_chars):
    """FTS context (mirrors harness.grep_answer's retrieval leg, no answerer)."""
    terms = re.findall(r"[A-Za-z0-9]+", query)
    fts = " OR ".join(f'"{t}"' for t in terms) or '""'
    rows = conn.execute(
        "SELECT raw_text FROM episodes_fts WHERE user_id=? AND episodes_fts MATCH ? "
        "ORDER BY rank LIMIT 20", (user_id, fts)).fetchall()
    ctx, total = [], 0
    for r in rows:
        t = r["raw_text"]
        if total + len(t) > max_chars:
            ctx.append(t[: max(0, max_chars - total)]); break
        ctx.append(t); total += len(t)
    return "\n\n".join(ctx) or "(no matches)"


CONTEXT_FNS = {
    "slate": _slate_ctx, "frag": _frag_ctx, "hybrid": _hybrid_ctx,
    "hier": _hier_ctx, "resonance": _resonance_ctx, "grep": _grep_ctx,
}


# ── embedding + chunking ──────────────────────────────────────────────────────
def _embed(texts: list[str]) -> np.ndarray:
    """384-dim vectors, L2-normalized so cosine == dot (SentenceTransformer doesn't
    normalize by default; HFEmbedder does — normalize here to be path-agnostic)."""
    if not texts:
        return np.zeros((0, 384), dtype=np.float32)
    V = np.asarray(get_embedder().encode(texts), dtype=np.float32)
    if V.ndim == 1:
        V = V[None, :]
    norms = np.linalg.norm(V, axis=1, keepdims=True)
    return V / np.clip(norms, 1e-9, None)


def _chunks(text: str) -> list[str]:
    """Split into sentence units + adjacent pairs, so a fact spanning two sentences
    can still match (max-over-chunk). Cheap; no model."""
    sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", text or "") if s.strip()]
    pairs = [f"{sents[i]} {sents[i+1]}" for i in range(len(sents) - 1)]
    return sents + pairs or [text.strip() or "(empty)"]


def _fact_scores(facts: list[str], text: str) -> list[float]:
    """For each fact, max cosine to any chunk of `text`."""
    chunks = _chunks(text)
    F, C = _embed(facts), _embed(chunks)
    if F.shape[0] == 0 or C.shape[0] == 0:
        return [0.0] * len(facts)
    return (F @ C.T).max(axis=1).tolist()  # max cosine per fact


# ── calibrate τ against logged judge runs ─────────────────────────────────────
def _load_labeled(paths: list[Path]) -> list[tuple[float, bool]]:
    """(score, judge_label) per (fact, answer) from baseline_*.json. The gold gives
    the fact text; the baseline gives the judged answer + per-fact covered[]."""
    gold = {g["id"]: g for g in _load_gold(HERE / "gold.jsonl")}
    pairs = []
    for p in paths:
        data = json.loads(p.read_text())
        for q in data.get("per_query", []):
            facts = gold.get(q["id"], {}).get("key_facts")
            cov = q.get("covered")
            if not facts or not cov or len(cov) != len(facts):
                continue
            scores = _fact_scores(facts, q.get("answer", ""))
            pairs.extend(zip(scores, (bool(c) for c in cov)))
    return pairs


def calibrate(paths: list[Path], lo=0.30, hi=0.80, step=0.02) -> dict:
    pairs = _load_labeled(paths)
    if not pairs:
        raise SystemExit("no labeled (fact, answer) pairs — check baseline_*.json")
    scores = np.array([s for s, _ in pairs])
    labels = np.array([l for _, l in pairs], dtype=bool)
    pos, neg = scores[labels], scores[~labels]

    best = None
    for tau in np.arange(lo, hi + 1e-9, step):
        pred = scores >= tau
        tp = int((pred & labels).sum()); tn = int((~pred & ~labels).sum())
        fp = int((pred & ~labels).sum()); fn = int((~pred & labels).sum())
        acc = (tp + tn) / len(pairs)
        # balanced acc — covered/uncovered are imbalanced; don't let the majority win
        tpr = tp / max(tp + fn, 1); tnr = tn / max(tn + fp, 1)
        bal = 0.5 * (tpr + tnr)
        row = {"tau": round(float(tau), 3), "acc": round(acc, 3),
               "bal_acc": round(bal, 3), "tp": tp, "tn": tn, "fp": fp, "fn": fn}
        if best is None or bal > best["bal_acc"]:
            best = row
    return {
        "n_pairs": len(pairs), "n_covered": int(labels.sum()),
        "score_covered_mean": round(float(pos.mean()), 3) if len(pos) else None,
        "score_uncovered_mean": round(float(neg.mean()), 3) if len(neg) else None,
        "best": best,
    }


# ── Coverage@B eval (no LLM) ──────────────────────────────────────────────────
def _load_gold(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        g = json.loads(line)
        assert g.get("query") and g.get("key_facts"), f"bad gold row: {g}"
        rows.append(g)
    return rows


def coverage_eval(gold, conn, user_id, context_fn, tau, budget_tok=DEFAULT_BUDGET_TOKENS):
    per = []
    for g in gold:
        ctx = context_fn(conn, user_id, g["query"], max_chars=budget_tok * CHARS_PER_TOKEN)
        scores = _fact_scores(g["key_facts"], ctx)
        covered = [s >= tau for s in scores]
        per.append({
            "id": g.get("id", g["query"][:40]), "query": g["query"],
            "hard": bool(g.get("hard")), "pass": len(covered) > 0 and all(covered),
            "covered": covered, "scores": [round(s, 3) for s in scores],
            "context_tokens": round(len(ctx) / CHARS_PER_TOKEN),
        })
    passes = [p["pass"] for p in per]
    hard = [p["pass"] for p in per if p["hard"]]
    return {
        "metric": "coverage@B", "tau": tau, "budget_tokens": budget_tok, "n": len(per),
        "sr_at_b": round(statistics.fmean(passes), 4) if passes else 0.0,
        "n_hard": len(hard),
        "sr_tail": round(statistics.fmean(hard), 4) if hard else None,
        "mean_context_tokens": round(statistics.fmean(
            [p["context_tokens"] for p in per]), 1) if per else 0,
        "per_query": per,
    }


# ── CLI ───────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("calibrate", help="fit τ vs logged judge runs")
    c.add_argument("--baselines", nargs="+",
                   default=["baseline_slate.json", "baseline_grep.json"])

    e = sub.add_parser("eval", help="Coverage@B over a gold file (no LLM)")
    e.add_argument("--db", required=True)
    e.add_argument("--user", required=True)
    e.add_argument("--system", default="resonance", choices=list(CONTEXT_FNS))
    e.add_argument("--gold", default="gold.jsonl")
    e.add_argument("--tau", type=float, required=True)
    e.add_argument("--budget", type=int, default=DEFAULT_BUDGET_TOKENS)

    args = ap.parse_args()
    if args.cmd == "calibrate":
        paths = [HERE / b if not Path(b).is_absolute() else Path(b) for b in args.baselines]
        rep = calibrate([p for p in paths if p.exists()])
        print(json.dumps(rep, indent=2))
        b = rep["best"]
        print(f"\n→ τ* = {b['tau']}  (bal_acc={b['bal_acc']}, acc={b['acc']}; "
              f"covered≈{rep['score_covered_mean']} vs uncovered≈{rep['score_uncovered_mean']})")
    else:
        conn = store.connect(args.db)
        gold = _load_gold(HERE / args.gold if not Path(args.gold).is_absolute() else Path(args.gold))
        rep = coverage_eval(gold, conn, args.user, CONTEXT_FNS[args.system], args.tau, args.budget)
        print(json.dumps(rep, indent=2))
        print(f"\nCoverage@B [{args.system}] = {rep['sr_at_b']:.1%}  "
              f"tail={('%.0f%%' % (100*rep['sr_tail'])) if rep['sr_tail'] is not None else '—'}  "
              f"(τ={args.tau}, n={rep['n']})")


if __name__ == "__main__":
    main()
