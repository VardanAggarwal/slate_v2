"""Provider fallback chain (claude→gemini→local) + Batch API helpers. Port from /Users/vardanaggarwal/slate engine/extract.py + engine/concepts.py. See PLAN.md §6.

call() is the sync path (Phase 2 dev iteration); submit_batch()/poll_batch()
are the nightly Batch API path (50% off). Callers pick the tier; this module
picks the provider/model and survives rate limits by walking the chain.
"""
import json
import re
import time

from core import config

# $ per MTok (input, output) — PLAN.md §8 pricing refs
_PRICES = {
    "claude-haiku-4-5": (1.0, 5.0),
    "claude-sonnet-4-6": (3.0, 15.0),
}
BATCH_DISCOUNT = 0.5


class LLMError(Exception):
    pass


def _model_for_tier(tier: str) -> str:
    return config.CLAUDE_MODEL_JUDGMENT if tier == "judgment" else config.CLAUDE_MODEL_MECHANICAL


def parse_json(raw: str) -> dict:
    """Strip code fences / stray prose and parse the first JSON object."""
    clean = re.sub(r"^```(?:json)?\s*", "", raw.strip())
    clean = re.sub(r"\s*```$", "", clean.strip())
    try:
        return json.loads(clean)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", clean, re.DOTALL)
        if m:
            return json.loads(m.group(0))
        raise


def estimate_cost(model: str, input_tokens: int, output_tokens: int,
                  batch: bool = False) -> float:
    inp, out = _PRICES.get(model, (3.0, 15.0))
    cost = (input_tokens * inp + output_tokens * out) / 1_000_000
    return cost * (BATCH_DISCOUNT if batch else 1.0)


# ── Sync path ─────────────────────────────────────────────────────────────────
def _call_claude_cli(prompt: str, model: str, max_tokens: int, system: str | None) -> dict:
    """Claude Code CLI in print mode — bills the Max/Pro subscription via
    CLAUDE_CODE_OAUTH_TOKEN instead of API keys. Raises LLMError on any
    failure (missing CLI, session limit, timeout) so the chain falls through
    to the API per-call."""
    import shutil
    import subprocess
    if not shutil.which("claude"):
        raise LLMError("claude CLI not installed")
    cmd = ["claude", "-p", "--model", model, "--output-format", "json"]
    if system:
        cmd += ["--system-prompt", system]
    try:
        proc = subprocess.run(cmd, input=prompt, capture_output=True,
                              text=True, timeout=600)
    except subprocess.TimeoutExpired:
        raise LLMError("claude CLI timed out")
    if proc.returncode != 0:
        raise LLMError(f"claude CLI exit {proc.returncode}: {proc.stderr[:200]}")
    data = json.loads(proc.stdout)
    if data.get("is_error"):
        raise LLMError(f"claude CLI error: {str(data.get('result'))[:200]}")
    usage = data.get("usage") or {}
    return {
        "text": data.get("result", ""),
        "provider": "claude-cli",
        "model": model,
        "input_tokens": usage.get("input_tokens", 0),
        "output_tokens": usage.get("output_tokens", 0),
        "cost": 0.0,  # subscription — no marginal spend
    }


def _call_claude(prompt: str, model: str, max_tokens: int, system: str | None) -> dict:
    import anthropic
    client = anthropic.Anthropic(api_key=config.ANTHROPIC_KEY)
    kwargs = {"model": model, "max_tokens": max_tokens,
              "messages": [{"role": "user", "content": prompt}]}
    if system:
        kwargs["system"] = system
    resp = client.messages.create(**kwargs)
    return {
        "text": resp.content[0].text,
        "provider": "claude",
        "model": model,
        "input_tokens": resp.usage.input_tokens,
        "output_tokens": resp.usage.output_tokens,
        "cost": estimate_cost(model, resp.usage.input_tokens, resp.usage.output_tokens),
    }


def _call_gemini(prompt: str, model: str, max_tokens: int, system: str | None) -> dict:
    from google import genai
    from google.genai import types
    client = genai.Client(api_key=config.GEMINI_KEY)
    cfg = types.GenerateContentConfig(
        temperature=0.1,
        max_output_tokens=max_tokens,
        response_mime_type="application/json",
        **({"system_instruction": system} if system else {}),
    )
    resp = client.models.generate_content(model=model, contents=prompt, config=cfg)
    usage = getattr(resp, "usage_metadata", None)
    return {
        "text": resp.text,
        "provider": "gemini",
        "model": model,
        "input_tokens": getattr(usage, "prompt_token_count", 0) or 0,
        "output_tokens": getattr(usage, "candidates_token_count", 0) or 0,
        "cost": 0.0,  # gemini free tier; only claude costs are tracked
    }


def call(prompt: str, tier: str = "mechanical", max_tokens: int = 2048,
         system: str | None = None, json_out: bool = True) -> dict:
    """Walk LLM_FALLBACK_ORDER; within claude, use the tier's model.

    Returns {json?, text, provider, model, input_tokens, output_tokens, cost}.
    Raises LLMError when every remote provider fails — callers that have a
    local fallback (e.g. KMeans blueprint) catch it; others let it bubble so
    the nightly run records a failed status and retries next night.
    """
    last_err: Exception | None = None
    for provider in config.LLM_FALLBACK_ORDER:
        # Two attempts per provider: a malformed-JSON response is usually
        # transient (sampling), so retry once before falling through.
        for attempt in range(2):
            try:
                if provider == "claude-cli" and config.CLAUDE_CODE_OAUTH_TOKEN:
                    result = _call_claude_cli(prompt, _model_for_tier(tier),
                                              max_tokens, system)
                elif provider == "claude" and config.ANTHROPIC_KEY:
                    result = _call_claude(prompt, _model_for_tier(tier), max_tokens, system)
                elif provider == "gemini" and config.GEMINI_KEY:
                    models = config.GEMINI_MODELS
                    result = None
                    for m in models:
                        try:
                            result = _call_gemini(prompt, m, max_tokens, system)
                            break
                        except Exception as e:  # try next gemini model
                            last_err = e
                    if result is None:
                        break  # no gemini model worked; next provider
                else:
                    break  # provider not configured; next provider
                if json_out:
                    result["json"] = parse_json(result["text"])
                return result
            except Exception as e:
                last_err = e
    raise LLMError(f"all providers failed: {last_err}")


# ── Batch path (Anthropic Message Batches API, 50% off) ───────────────────────
def submit_batch(requests: list[dict]) -> str:
    """requests: [{custom_id, prompt, tier?, max_tokens?, system?}, ...] → batch_id."""
    import anthropic
    client = anthropic.Anthropic(api_key=config.ANTHROPIC_KEY)
    entries = []
    for r in requests:
        params = {
            "model": _model_for_tier(r.get("tier", "mechanical")),
            "max_tokens": r.get("max_tokens", 2048),
            "messages": [{"role": "user", "content": r["prompt"]}],
        }
        if r.get("system"):
            params["system"] = r["system"]
        entries.append({"custom_id": r["custom_id"], "params": params})
    batch = client.messages.batches.create(requests=entries)
    return batch.id


def poll_batch(batch_id: str, interval: int = 30, timeout: int = 24 * 3600) -> dict:
    """Block until the batch ends; return {custom_id: result_dict} for succeeded
    entries; errored entries map to {"error": ...}."""
    import anthropic
    client = anthropic.Anthropic(api_key=config.ANTHROPIC_KEY)
    deadline = time.time() + timeout
    while True:
        batch = client.messages.batches.retrieve(batch_id)
        if batch.processing_status == "ended":
            break
        if time.time() > deadline:
            raise LLMError(f"batch {batch_id} timed out")
        time.sleep(interval)

    out = {}
    for entry in client.messages.batches.results(batch_id):
        if entry.result.type == "succeeded":
            msg = entry.result.message
            out[entry.custom_id] = {
                "text": msg.content[0].text,
                "provider": "claude",
                "model": msg.model,
                "input_tokens": msg.usage.input_tokens,
                "output_tokens": msg.usage.output_tokens,
                "cost": estimate_cost(msg.model, msg.usage.input_tokens,
                                      msg.usage.output_tokens, batch=True),
            }
        else:
            out[entry.custom_id] = {"error": entry.result.type}
    return out
