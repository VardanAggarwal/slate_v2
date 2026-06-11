import pytest

from core.llm import estimate_cost, parse_json


def test_parse_json_plain():
    assert parse_json('{"a": 1}') == {"a": 1}


def test_parse_json_fenced():
    assert parse_json('```json\n{"a": 1}\n```') == {"a": 1}


def test_parse_json_with_prose():
    assert parse_json('Here you go:\n{"a": {"b": 2}}\nDone.') == {"a": {"b": 2}}


def test_parse_json_garbage_raises():
    with pytest.raises(Exception):
        parse_json("not json at all")


def test_claude_cli_provider_parses_print_mode_json(monkeypatch):
    import json
    import core.llm as llm_mod

    class FakeProc:
        returncode = 0
        stderr = ""
        stdout = json.dumps({"result": '{"a": 1}', "is_error": False,
                             "usage": {"input_tokens": 10, "output_tokens": 5}})

    monkeypatch.setattr("shutil.which", lambda _: "/usr/bin/claude")
    monkeypatch.setattr("subprocess.run", lambda *a, **kw: FakeProc())
    out = llm_mod._call_claude_cli("p", "claude-haiku-4-5", 100, None)
    assert out["provider"] == "claude-cli"
    assert out["cost"] == 0.0
    assert out["input_tokens"] == 10


def test_chain_falls_from_cli_to_api(monkeypatch):
    import core.llm as llm_mod

    def cli_limited(*a, **kw):
        raise llm_mod.LLMError("session limit reached")

    def api_ok(prompt, model, max_tokens, system):
        return {"text": '{"ok": true}', "provider": "claude", "model": model,
                "input_tokens": 1, "output_tokens": 1, "cost": 0.001}

    monkeypatch.setattr(llm_mod, "_call_claude_cli", cli_limited)
    monkeypatch.setattr(llm_mod, "_call_claude", api_ok)
    monkeypatch.setattr(llm_mod.config, "CLAUDE_CODE_OAUTH_TOKEN", "tok")
    monkeypatch.setattr(llm_mod.config, "ANTHROPIC_KEY", "key")
    monkeypatch.setattr(llm_mod.config, "LLM_FALLBACK_ORDER", ["claude-cli", "claude"])
    result = llm_mod.call("anything")
    assert result["provider"] == "claude"  # fell through per-call
    assert result["json"] == {"ok": True}


def test_estimate_cost_haiku_batch_half_price():
    full = estimate_cost("claude-haiku-4-5", 1_000_000, 0)
    half = estimate_cost("claude-haiku-4-5", 1_000_000, 0, batch=True)
    assert full == 1.0
    assert half == 0.5
