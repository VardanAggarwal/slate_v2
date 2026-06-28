"""C12 — calibration persistence + push-down (offline; the fit loop is SR@B-gated).

Covers the persistence MECHANISM (store table + merge/push, per-user). The
`retrieve.assemble_context` pickup wiring lands with the P3/P4 retrieve commit
(it depends on that branch's assembly changes); a `value_floor`-bearing default
profile is what it merges over."""
from core import calibration as calib, store
from tests.conftest import UID

# Minimal stand-in for a stage's in-code defaults (don't import retrieve here).
DEFAULT_CALIBRATION = {"gain_floor": 0.35, "max_items": 24, "value_floor": None}


def test_merged_is_defaults_when_nothing_fitted(conn):
    out = calib.merged(conn, DEFAULT_CALIBRATION, UID)
    assert out == DEFAULT_CALIBRATION
    assert out is not DEFAULT_CALIBRATION       # fresh dict, defaults not mutated


def test_set_get_roundtrip(conn):
    with conn:
        store.set_calibration(conn, UID, {"value_floor": 0.2, "max_items": 5})
    assert store.get_calibration(conn, UID) == {"value_floor": 0.2, "max_items": 5}


def test_merged_overlays_persisted_over_defaults(conn):
    with conn:
        store.set_calibration(conn, UID, {"value_floor": 0.25})
    out = calib.merged(conn, DEFAULT_CALIBRATION, UID)
    assert out["value_floor"] == 0.25                      # persisted wins
    assert out["gain_floor"] == DEFAULT_CALIBRATION["gain_floor"]  # default kept


def test_push_updates_keys_and_ignores_none(conn):
    with conn:
        calib.push(conn, UID, value_floor=0.18)
        calib.push(conn, UID, value_floor=None, max_items=7)  # None is a no-op
    prof = store.get_calibration(conn, UID)
    assert prof["value_floor"] == 0.18 and prof["max_items"] == 7


def test_calibration_is_per_user(conn):
    other = "usr_other"
    with conn:
        store.set_calibration(conn, UID, {"value_floor": 0.3})
    assert calib.merged(conn, DEFAULT_CALIBRATION, other) == DEFAULT_CALIBRATION
    assert calib.merged(conn, DEFAULT_CALIBRATION, UID)["value_floor"] == 0.3
