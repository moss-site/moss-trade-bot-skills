"""Tests for ambush/fingerprint.py — the Python mirror of
internal/ambush/backtest/fingerprint.go.

The whole point of fingerprint.py is to produce a hex SHA-256 that is
BYTE-IDENTICAL to the Go side, so the server's `fingerprint_mismatch` check on
/backtest/verify-job passes. Any silent change to the canonical encoding
(key order, fixed-point precision, separators, direction normalization) would
break that parity. These tests pin the current encoding so such a change fails
loudly here instead of as a confusing verify-job rejection.

Run:
    cd scripts && python3 -m unittest ambush.fingerprint_test
"""

from __future__ import annotations

import os
import sys
import unittest
from decimal import Decimal
from types import SimpleNamespace

_HERE = os.path.dirname(os.path.abspath(__file__))
_SCRIPTS = os.path.dirname(_HERE)
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

from ambush import fingerprint as fp


def _leg(**overrides):
    base = dict(
        cooldown_bars=0, leverage=3, max_hold_hours=30, momentum_bars=0,
        position_pct=Decimal("0.2"), sl_atr_mult=Decimal("0"),
        stop_loss_pct=Decimal("0.08"), trailing_pct=Decimal("0.05"),
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def _params(**leg_overrides):
    return SimpleNamespace(
        direction="balanced",
        long=_leg(),
        short=_leg(entry_delay_bars=0),
        rhythm=SimpleNamespace(max_trades_per_event=1, same_coin_dedup_days=7),
        **leg_overrides,
    )


# Pinned hex for the fixed input below. If this changes, the Python canonical
# encoding drifted away from the Go side — DO NOT just update the constant;
# confirm internal/ambush/backtest/fingerprint.go produces the same hex first.
_PINNED_HEX = "f5ccdde129517af09c4ad402495e9eb3218bf431ff610bef3dd9eadb3e887fae"


class FingerprintCanonicalTest(unittest.TestCase):
    def test_pinned_canonical_hex(self):
        h = fp.canonical("ambush", "v1", "abc123", Decimal("10000"), _params())
        self.assertEqual(h, _PINNED_HEX)

    def test_deterministic(self):
        a = fp.canonical("ambush", "v1", "abc123", Decimal("10000"), _params())
        b = fp.canonical("ambush", "v1", "abc123", Decimal("10000"), _params())
        self.assertEqual(a, b)

    def test_param_change_changes_hash(self):
        base = fp.canonical("ambush", "v1", "abc123", Decimal("10000"), _params())
        changed = fp.canonical(
            "ambush", "v1", "abc123", Decimal("10000"),
            SimpleNamespace(
                direction="balanced",
                long=_leg(leverage=5),
                short=_leg(entry_delay_bars=0),
                rhythm=SimpleNamespace(max_trades_per_event=1, same_coin_dedup_days=7),
            ),
        )
        self.assertNotEqual(base, changed)

    def test_dataset_sha_changes_hash(self):
        a = fp.canonical("ambush", "v1", "sha_A", Decimal("10000"), _params())
        b = fp.canonical("ambush", "v1", "sha_B", Decimal("10000"), _params())
        self.assertNotEqual(a, b)

    def test_direction_normalized(self):
        # direction is lowercased + stripped (must match Go ToLower/TrimSpace),
        # so "  Balanced " fingerprints identically to "balanced".
        a = fp.canonical("ambush", "v1", "abc123", Decimal("10000"), _params())
        p2 = _params()
        p2.direction = "  Balanced "
        b = fp.canonical("ambush", "v1", "abc123", Decimal("10000"), p2)
        self.assertEqual(a, b)


class FixedPointTest(unittest.TestCase):
    def test_half_up_rounding(self):
        # shopspring StringFixed uses ROUND_HALF_UP, not banker's rounding.
        self.assertEqual(fp._fixed(Decimal("0.125"), 2), "0.13")
        self.assertEqual(fp._fixed(Decimal("0.135"), 2), "0.14")

    def test_fixed_places(self):
        self.assertEqual(fp._fixed(Decimal("0.2"), 8), "0.20000000")
        self.assertEqual(fp._fixed(Decimal("10000"), 2), "10000.00")


if __name__ == "__main__":
    unittest.main()
