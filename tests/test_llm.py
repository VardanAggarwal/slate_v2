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


def test_estimate_cost_haiku_batch_half_price():
    full = estimate_cost("claude-haiku-4-5", 1_000_000, 0)
    half = estimate_cost("claude-haiku-4-5", 1_000_000, 0, batch=True)
    assert full == 1.0
    assert half == 0.5
