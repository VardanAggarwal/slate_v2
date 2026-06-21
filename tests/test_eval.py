"""Offline, deterministic tests for the SR@B harness. No network, no embeddings:
llm.call is faked and routed by the prompt markers; the answerer is a stub."""
import json

import pytest

from eval import harness


# ── fake LLM: route by the markers the harness embeds in each prompt ───────────
class _FakeLLM:
    """answers[query] -> answer text; coverage[query] -> list[bool] for the judge."""
    def __init__(self, answers, coverage):
        self.answers, self.coverage = answers, coverage
        self.calls = {"answer": 0, "judge": 0}

    def __call__(self, prompt, **kw):
        if harness._ANSWER_MARK in prompt:
            self.calls["answer"] += 1
            q = prompt.split("Question: ")[1].split("\nAnswer:")[0]
            return {"text": self.answers.get(q, "I don't know"), "cost": 0.001}
        if harness._JUDGE_MARK in prompt:
            self.calls["judge"] += 1
            q = prompt.split("Question: ")[1].split("\n\n")[0]
            return {"json": {"covered": self.coverage.get(q, []), "notes": "ok"},
                    "cost": 0.001, "text": ""}
        raise AssertionError("unexpected prompt")


@pytest.fixture
def gold():
    return [
        {"id": "q1", "query": "easy one", "key_facts": ["a", "b"], "hard": False},
        {"id": "q2", "query": "hard one", "key_facts": ["c", "d"], "hard": True},
    ]


def _stub_answerer(conn, user_id, query, budget_tok):
    # context is irrelevant here; the fake LLM keys off the query
    return harness._answer_from_context(query, context=f"ctx for {query}")


def test_pass_requires_all_facts(monkeypatch, gold):
    fake = _FakeLLM(
        answers={"easy one": "a and b", "hard one": "only c"},
        coverage={"easy one": [True, True], "hard one": [True, False]},  # q2 misses d
    )
    monkeypatch.setattr(harness.llm, "call", fake)
    rep = harness.run_eval(gold, conn=None, user_id="u", answer_fn=_stub_answerer)

    assert rep["n"] == 2
    assert rep["sr_at_b"] == 0.5          # only q1 passes
    assert rep["sr_tail"] == 0.0          # q2 is the only hard row, and it failed
    assert rep["n_hard"] == 1
    by = {p["id"]: p for p in rep["per_query"]}
    assert by["q1"]["pass"] is True and by["q2"]["pass"] is False
    # one answer + one judge call per query, nothing more
    assert fake.calls == {"answer": 2, "judge": 2}


def test_all_pass(monkeypatch, gold):
    fake = _FakeLLM(
        answers={"easy one": "a b", "hard one": "c d"},
        coverage={"easy one": [True, True], "hard one": [True, True]},
    )
    monkeypatch.setattr(harness.llm, "call", fake)
    rep = harness.run_eval(gold, conn=None, user_id="u", answer_fn=_stub_answerer)
    assert rep["sr_at_b"] == 1.0 and rep["sr_tail"] == 1.0


def test_judge_length_mismatch_is_a_miss(monkeypatch, gold):
    # judge returns too few booleans → unjudged facts count as not covered
    fake = _FakeLLM(answers={"easy one": "a"}, coverage={"easy one": [True]})
    monkeypatch.setattr(harness.llm, "call", fake)
    j = harness.judge("easy one", "a", ["a", "b"])
    assert j["covered"] == [True, False] and j["pass"] is False


def test_compare_delta_and_forgetting():
    before = {"sr_at_b": 1.0, "per_query": [
        {"id": "q1", "pass": True}, {"id": "q2", "pass": True}]}
    after = {"sr_at_b": 0.5, "per_query": [
        {"id": "q1", "pass": True}, {"id": "q2", "pass": False}]}
    d = harness.compare(before, after)
    assert d["delta_sr_at_b"] == -0.5
    assert d["forgetting_rate"] == 0.5 and d["regressions"] == ["q2"]


def test_load_gold_skips_comments(tmp_path):
    p = tmp_path / "g.jsonl"
    p.write_text("# a comment\n"
                 + json.dumps({"id": "x", "query": "q", "key_facts": ["f"]}) + "\n\n")
    gold = harness.load_gold(p)
    assert len(gold) == 1 and gold[0]["id"] == "x"
