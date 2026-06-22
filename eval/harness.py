"""SR@B eval harness — the North Star metric (PRD v2 §North Star). SIMPLEST viable form.

What it does: for each gold query, build an answer within a token budget B, then a
single judge checks the answer against a PRE-REGISTERED key-facts checklist. A query
passes only if EVERY key fact is covered. SR@B = pass rate.

    SR@B = (# gold queries whose budgeted answer covers all key facts) / (# gold queries)

Deliberately omitted until needed (PRD v1 of the metric; add when a phase requires it):
  - two-judge + κ inter-rater agreement (here: one judge),
  - the RAG/grep @ 3B retrieval *oracle* as reference (here: the hand-authored
    key-facts checklist IS the reference / gold),
  - adaptive-probe-set machinery (here: a gold file is a gold file; keep frozen and
    probe in SEPARATE files and never trend the probe — that separation is policy,
    not code),
  - the build-cost gate (write+consolidate tokens/query) — that lives at Write/Consolidate,
    not in the eval of an answer.

The answerer is PLUGGABLE: `run_eval` scores any `answer_fn(conn, user_id, query, budget)`.
Slate is one (`slate_answer`); a grep/FTS baseline is another (`grep_answer`) — same
budget, same judge, so they are directly comparable (PRD: competitors at budget B).
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from core import config, llm, store
from core.hybrid import hybrid_context
from core.recall import assemble_context, list_episodes
from core.retrieve import assemble_context as fragment_context

# Budget is a TOKEN slice (PRD). No tokenizer on the host, so approximate with a
# chars/token ratio — good enough to size assembly; report it as an estimate.
CHARS_PER_TOKEN = 4
DEFAULT_BUDGET_TOKENS = 2000

# Stable markers so tests can route a fake llm.call by prompt kind. Harmless in prod.
_ANSWER_MARK = "[SLATE-EVAL-ANSWER]"
_JUDGE_MARK = "[SLATE-EVAL-JUDGE]"

_ANSWER_SYS = (
    "Answer the question using ONLY the provided context. "
    "If the context does not contain the answer, reply exactly: I don't know. "
    "Be concise and factual; do not add information beyond the context."
)


def _tok(s: str) -> int:
    return len(s) // CHARS_PER_TOKEN


# ── answerers (pluggable; same signature, same return shape) ───────────────────
def _answer_from_context(query: str, context: str) -> dict:
    # Answerer on the JUDGMENT tier (sonnet): a weak (haiku) reader fails to
    # synthesise the answer from sufficient context and DEFLATES SR@B (observed:
    # slate@2000 g01 passed 3/3 on sonnet, failed on haiku). SR@B must measure
    # context sufficiency, not answerer weakness, so the host LLM is sonnet-class.
    prompt = f"{_ANSWER_MARK}\nContext:\n{context}\n\nQuestion: {query}\nAnswer:"
    res = llm.call(prompt, tier="judgment", max_tokens=512,
                   system=_ANSWER_SYS, json_out=False)
    return {"answer": res["text"].strip(), "context": context,
            "context_tokens": _tok(context), "cost": res.get("cost", 0.0)}


def slate_answer(conn, user_id: str, query: str, budget_tok: int) -> dict:
    """Slate: assemble budgeted context via spreading-activation recall, then answer."""
    ctx = assemble_context(conn, user_id, query,
                           max_chars=budget_tok * CHARS_PER_TOKEN)
    return {"system": "slate", **_answer_from_context(query, ctx)}


def grep_answer(conn, user_id: str, query: str, budget_tok: int) -> dict:
    """Competitor: FTS over raw episodes, take top matches up to budget, then answer.
    The cheap retrieval baseline Slate must beat at the same budget (PRD: grep@B)."""
    budget_chars = budget_tok * CHARS_PER_TOKEN
    # Build a safe FTS5 expression: raw query punctuation (?, /) is FTS5 syntax and
    # errors. Keep word tokens, quote each, OR them.
    import re
    terms = re.findall(r"[A-Za-z0-9]+", query)
    fts = " OR ".join(f'"{t}"' for t in terms) or '""'
    rows = conn.execute(
        """SELECT raw_text FROM episodes_fts
           WHERE user_id = ? AND episodes_fts MATCH ?
           ORDER BY rank LIMIT 20""", (user_id, fts)).fetchall()
    ctx, total = [], 0
    for r in rows:
        t = r["raw_text"]
        if total + len(t) > budget_chars:
            ctx.append(t[: max(0, budget_chars - total)])
            break
        ctx.append(t)
        total += len(t)
    context = "\n\n".join(ctx) or "(no matches)"
    return {"system": "grep", **_answer_from_context(query, context)}


def frag_answer(conn, user_id: str, query: str, budget_tok: int) -> dict:
    """Slate, fragment-backed: assemble verbatim fragment spans via the predictor's
    assembly wrapper (core/retrieve.py), then answer. The P2.5 path — reads the
    Write fragment layer the v2 `slate_answer` (claims/concepts) never touched."""
    ctx = fragment_context(conn, user_id, query,
                           max_chars=budget_tok * CHARS_PER_TOKEN)
    return {"system": "frag", **_answer_from_context(query, ctx)}


def hybrid_answer(conn, user_id: str, query: str, budget_tok: int) -> dict:
    """Slate, frag+concept HYBRID: budget-split blend of the concept path (the tail)
    and the fragment path (specificity) — core/hybrid.py. The §6 P4 'path to the
    R-gate': the two single paths are complementary, neither dominates, so a blend
    aims to keep frag's factual wins AND claims' conceptual-tail wins."""
    ctx = hybrid_context(conn, user_id, query,
                         max_chars=budget_tok * CHARS_PER_TOKEN)
    return {"system": "hybrid", **_answer_from_context(query, ctx)}


_ANSWERERS = {"slate": slate_answer, "grep": grep_answer, "frag": frag_answer,
              "hybrid": hybrid_answer}


# ── the judge (single; binary; against the pre-registered checklist) ───────────
def judge(query: str, answer: str, key_facts: list[str]) -> dict:
    """Does `answer` cover every key fact? Returns per-fact booleans + a binary pass.
    pass == every key fact covered (PRD: '"same answer" is binary')."""
    facts = "\n".join(f"{i+1}. {f}" for i, f in enumerate(key_facts))
    prompt = (
        f"{_JUDGE_MARK}\n"
        f"Question: {query}\n\nCandidate answer:\n{answer}\n\n"
        f"Key facts that a correct answer must contain:\n{facts}\n\n"
        "For each numbered key fact, decide whether the candidate answer states it "
        "(or clearly entails it). Ignore extra information. Respond with JSON only:\n"
        '{"covered": [true/false per key fact, in order], "notes": "<one line>"}'
    )
    res = llm.call(prompt, tier="judgment", max_tokens=512, json_out=True)
    covered = [bool(x) for x in (res.get("json", {}).get("covered") or [])]
    # Length mismatch ⇒ judge misbehaved; treat unjudged facts as not covered.
    while len(covered) < len(key_facts):
        covered.append(False)
    covered = covered[: len(key_facts)]
    passed = len(key_facts) > 0 and all(covered)
    return {"pass": passed, "covered": covered,
            "notes": res.get("json", {}).get("notes", ""), "cost": res.get("cost", 0.0)}


# ── gold I/O ───────────────────────────────────────────────────────────────────
def load_gold(path: str | Path) -> list[dict]:
    """JSONL, one query per line: {id, query, key_facts:[...], hard?:bool}."""
    out = []
    for ln in Path(path).read_text().splitlines():
        ln = ln.strip()
        if not ln or ln.startswith("#"):
            continue
        g = json.loads(ln)
        assert g.get("query") and g.get("key_facts"), f"bad gold row: {g}"
        out.append(g)
    return out


# ── the run ────────────────────────────────────────────────────────────────────
def run_eval(gold: list[dict], conn, user_id: str, *,
             answer_fn=slate_answer, budget_tok: int = DEFAULT_BUDGET_TOKENS) -> dict:
    """Score one answerer over the gold set at budget B. Returns SR@B + per-query.

    Tail slice (PRD: 'the tail is the gate'): SR@B over rows tagged "hard": true.
    """
    per = []
    for g in gold:
        a = answer_fn(conn, user_id, g["query"], budget_tok)
        j = judge(g["query"], a["answer"], g["key_facts"])
        per.append({
            "id": g.get("id", g["query"][:40]), "query": g["query"],
            "hard": bool(g.get("hard")), "pass": j["pass"], "covered": j["covered"],
            "answer": a["answer"], "context_tokens": a["context_tokens"],
            "notes": j["notes"], "cost": round(a["cost"] + j["cost"], 6),
        })

    passes = [p["pass"] for p in per]
    hard = [p["pass"] for p in per if p["hard"]]
    return {
        "system": getattr(answer_fn, "__name__", "custom").replace("_answer", ""),
        "user_id": user_id, "budget_tokens": budget_tok, "n": len(per),
        "sr_at_b": round(statistics.fmean(passes), 4) if passes else 0.0,
        "n_hard": len(hard),
        "sr_tail": round(statistics.fmean(hard), 4) if hard else None,
        "mean_context_tokens": round(statistics.fmean(
            [p["context_tokens"] for p in per]), 1) if per else 0,
        "total_cost": round(sum(p["cost"] for p in per), 4),
        "per_query": per,
    }


def compare(before: dict, after: dict) -> dict:
    """ΔSR@B between two runs on the SAME gold (PRD: computed only on the frozen set).
    Catastrophic-forgetting = queries that passed before and now fail (the hard floor)."""
    b = {p["id"]: p["pass"] for p in before["per_query"]}
    a = {p["id"]: p["pass"] for p in after["per_query"]}
    ids = b.keys() & a.keys()
    regressions = sorted(i for i in ids if b[i] and not a[i])
    gains = sorted(i for i in ids if a[i] and not b[i])
    return {
        "delta_sr_at_b": round(after["sr_at_b"] - before["sr_at_b"], 4),
        "forgetting_rate": round(len(regressions) / len(ids), 4) if ids else 0.0,
        "regressions": regressions, "gains": gains,
    }


def format_report(rep: dict) -> str:
    lines = [
        f"SR@B  system={rep['system']}  B={rep['budget_tokens']}tok  n={rep['n']}",
        f"  SR@B        : {rep['sr_at_b']:.1%}",
        f"  SR@B (tail) : " + (f"{rep['sr_tail']:.1%} (n={rep['n_hard']})"
                               if rep["sr_tail"] is not None else "— (no hard rows)"),
        f"  mean ctx    : {rep['mean_context_tokens']:.0f} tok"
        f"   judge+answer cost: ${rep['total_cost']:.4f}",
        "  ── per query ──",
    ]
    for p in rep["per_query"]:
        mark = "✓" if p["pass"] else "✗"
        tag = " [hard]" if p["hard"] else ""
        lines.append(f"  {mark} {p['id']}{tag}  ({sum(p['covered'])}/{len(p['covered'])} facts)")
    return "\n".join(lines)


# ── CLI ────────────────────────────────────────────────────────────────────────
def _main() -> None:
    ap = argparse.ArgumentParser(description="Slate SR@B eval (simplest form)")
    ap.add_argument("--gold", default=str(Path(__file__).parent / "gold.jsonl"))
    ap.add_argument("--user", help="user_id (corpus owner)")
    ap.add_argument("--budget", type=int, default=DEFAULT_BUDGET_TOKENS, help="budget B in tokens")
    ap.add_argument("--system", choices=list(_ANSWERERS), default="slate")
    ap.add_argument("--half", action="store_true", help="also run at B/2 (parsimony check)")
    ap.add_argument("--out", help="write full JSON report here")
    ap.add_argument("--list-notes", type=int, metavar="N",
                    help="print N recent notes to help author gold, then exit")
    args = ap.parse_args()

    conn = store.connect()
    user_id = args.user or config.DEFAULT_USER_ID

    if args.list_notes:
        for e in list_episodes(conn, user_id, limit=args.list_notes):
            print(f"{e['ts'][:10]}  {e['id']}  {e['title'] or '(untitled)'}")
            print(f"    {e['essence'][:160]}")
        return

    gold = load_gold(args.gold)
    fn = _ANSWERERS[args.system]
    for b in ([args.budget, args.budget // 2] if args.half else [args.budget]):
        rep = run_eval(gold, conn, user_id, answer_fn=fn, budget_tok=b)
        print(format_report(rep))
        print()
        if args.out:
            Path(args.out).write_text(json.dumps(rep, indent=2))


if __name__ == "__main__":
    _main()
