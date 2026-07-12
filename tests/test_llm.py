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


def test_claude_cli_error_detail_from_stdout(monkeypatch):
    """Newer CLIs exit 1 with the error JSON on stdout and empty stderr — the
    raised LLMError must surface that result text, not an empty string."""
    import json
    import core.llm as llm_mod
    import pytest

    class FakeProc:
        returncode = 1
        stderr = ""
        stdout = json.dumps({"is_error": True,
                             "result": "You've hit your session limit · resets 8:10am (UTC)"})

    monkeypatch.setattr("shutil.which", lambda _: "/usr/bin/claude")
    monkeypatch.setattr("subprocess.run", lambda *a, **kw: FakeProc())
    with pytest.raises(llm_mod.LLMError, match="session limit"):
        llm_mod._call_claude_cli("p", "claude-haiku-4-5", 100, None)


def test_claude_cli_strips_api_key_from_env(monkeypatch):
    """The CLI prefers ANTHROPIC_API_KEY over the subscription token — if it
    leaks into the subprocess, the 'subscription' rung silently bills the API
    (and inherits its usage caps). The rung must be subscription-only."""
    import json
    import core.llm as llm_mod

    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-leaked")
    monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", "subscription-token")
    seen = {}

    class FakeProc:
        returncode = 0
        stderr = ""
        stdout = json.dumps({"result": "{}", "is_error": False, "usage": {}})

    def fake_run(cmd, **kw):
        seen["env"] = kw.get("env")
        return FakeProc()

    monkeypatch.setattr("shutil.which", lambda _: "/usr/bin/claude")
    monkeypatch.setattr("subprocess.run", fake_run)
    llm_mod._call_claude_cli("p", "claude-haiku-4-5", 100, None)
    assert seen["env"] is not None, "env must be passed explicitly"
    assert "ANTHROPIC_API_KEY" not in seen["env"]
    assert seen["env"].get("CLAUDE_CODE_OAUTH_TOKEN") == "subscription-token"


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
    monkeypatch.setattr(llm_mod.time, "sleep", lambda *_: None)  # no real backoff
    result = llm_mod.call("anything")
    assert result["provider"] == "claude"  # fell through per-call
    assert result["json"] == {"ok": True}


def test_retries_within_provider_then_succeeds(monkeypatch):
    import core.llm as llm_mod

    calls = {"n": 0}

    def cli_flaky(prompt, model, max_tokens, system):
        calls["n"] += 1
        if calls["n"] < 3:  # fail first two attempts, succeed on the third
            raise llm_mod.LLMError("503 overloaded")
        return {"text": '{"ok": true}', "provider": "claude-cli", "model": model,
                "input_tokens": 1, "output_tokens": 1, "cost": 0.0}

    sleeps = []
    monkeypatch.setattr(llm_mod, "_call_claude_cli", cli_flaky)
    monkeypatch.setattr(llm_mod.time, "sleep", lambda s: sleeps.append(s))
    monkeypatch.setattr(llm_mod.config, "CLAUDE_CODE_OAUTH_TOKEN", "tok")
    monkeypatch.setattr(llm_mod.config, "LLM_FALLBACK_ORDER", ["claude-cli"])
    monkeypatch.setattr(llm_mod.config, "LLM_MAX_ATTEMPTS", 3)
    monkeypatch.setattr(llm_mod.config, "LLM_BACKOFF_BASE", 2.0)

    result = llm_mod.call("anything")
    assert result["provider"] == "claude-cli"
    assert calls["n"] == 3                  # retried twice before success
    assert sleeps == [2.0, 4.0]             # exponential backoff between tries


def test_unconfigured_provider_is_skipped(monkeypatch):
    import core.llm as llm_mod

    def api_ok(prompt, model, max_tokens, system):
        return {"text": '{"ok": true}', "provider": "claude", "model": model,
                "input_tokens": 1, "output_tokens": 1, "cost": 0.0}

    monkeypatch.setattr(llm_mod, "_call_claude", api_ok)
    # cli first in order but no token → must be skipped, not attempted
    monkeypatch.setattr(llm_mod, "_call_claude_cli",
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError("cli ran")))
    monkeypatch.setattr(llm_mod.config, "CLAUDE_CODE_OAUTH_TOKEN", "")
    monkeypatch.setattr(llm_mod.config, "ANTHROPIC_KEY", "key")
    monkeypatch.setattr(llm_mod.config, "LLM_FALLBACK_ORDER", ["claude-cli", "claude"])
    result = llm_mod.call("anything")
    assert result["provider"] == "claude"


def test_all_providers_fail_raises(monkeypatch):
    import core.llm as llm_mod

    monkeypatch.setattr(llm_mod, "_call_claude_cli",
                        lambda *a, **k: (_ for _ in ()).throw(llm_mod.LLMError("down")))
    monkeypatch.setattr(llm_mod.time, "sleep", lambda *_: None)
    monkeypatch.setattr(llm_mod.config, "CLAUDE_CODE_OAUTH_TOKEN", "tok")
    monkeypatch.setattr(llm_mod.config, "LLM_FALLBACK_ORDER", ["claude-cli"])
    monkeypatch.setattr(llm_mod.config, "LLM_MAX_ATTEMPTS", 2)
    with pytest.raises(llm_mod.LLMError):
        llm_mod.call("anything")


def test_openrouter_is_configured_and_tried_first(monkeypatch):
    import core.llm as llm_mod

    def openrouter_ok(prompt, model, max_tokens, system):
        return {"text": '{"ok": true}', "provider": "openrouter", "model": model,
                "input_tokens": 1, "output_tokens": 1, "cost": 0.0}

    monkeypatch.setattr(llm_mod, "_call_openrouter", openrouter_ok)
    monkeypatch.setattr(llm_mod, "_call_claude_cli",
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError("cli ran")))
    monkeypatch.setattr(llm_mod.config, "OPENROUTER_KEY", "key")
    monkeypatch.setattr(llm_mod.config, "CLAUDE_CODE_OAUTH_TOKEN", "tok")
    monkeypatch.setattr(llm_mod.config, "LLM_FALLBACK_ORDER", ["openrouter", "claude-cli"])
    result = llm_mod.call("anything")
    assert result["provider"] == "openrouter"


def test_openrouter_unconfigured_falls_through(monkeypatch):
    import core.llm as llm_mod

    def cli_ok(prompt, model, max_tokens, system):
        return {"text": '{"ok": true}', "provider": "claude-cli", "model": model,
                "input_tokens": 1, "output_tokens": 1, "cost": 0.0}

    monkeypatch.setattr(llm_mod, "_call_openrouter",
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError("openrouter ran")))
    monkeypatch.setattr(llm_mod, "_call_claude_cli", cli_ok)
    monkeypatch.setattr(llm_mod.config, "OPENROUTER_KEY", "")
    monkeypatch.setattr(llm_mod.config, "CLAUDE_CODE_OAUTH_TOKEN", "tok")
    monkeypatch.setattr(llm_mod.config, "LLM_FALLBACK_ORDER", ["openrouter", "claude-cli"])
    result = llm_mod.call("anything")
    assert result["provider"] == "claude-cli"


def test_truncated_response_retries_with_bigger_budget(monkeypatch):
    """A reasoning model that burns hidden CoT tokens can hit max_tokens before
    the visible answer — call() must escalate the budget and retry, not return
    a chopped-off response."""
    import core.llm as llm_mod

    seen_budgets = []

    def openrouter_flaky(prompt, model, max_tokens, system):
        seen_budgets.append(max_tokens)
        if max_tokens < 100:
            return {"text": "", "provider": "openrouter", "model": model,
                    "input_tokens": 1, "output_tokens": 1, "cost": 0.0, "truncated": True}
        return {"text": '{"ok": true}', "provider": "openrouter", "model": model,
                "input_tokens": 1, "output_tokens": 1, "cost": 0.0, "truncated": False}

    monkeypatch.setattr(llm_mod, "_call_openrouter", openrouter_flaky)
    monkeypatch.setattr(llm_mod.config, "OPENROUTER_KEY", "key")
    monkeypatch.setattr(llm_mod.config, "LLM_FALLBACK_ORDER", ["openrouter"])
    monkeypatch.setattr(llm_mod.config, "LLM_MAX_ATTEMPTS", 3)
    monkeypatch.setattr(llm_mod.time, "sleep", lambda *_: None)

    result = llm_mod.call("anything", max_tokens=32)
    assert result["json"] == {"ok": True}
    assert seen_budgets == [32, 64, 128]  # escalated until it stopped truncating


def test_estimate_cost_haiku_batch_half_price():
    full = estimate_cost("claude-haiku-4-5", 1_000_000, 0)
    half = estimate_cost("claude-haiku-4-5", 1_000_000, 0, batch=True)
    assert full == 1.0
    assert half == 0.5
