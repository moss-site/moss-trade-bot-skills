"""Cross-language fingerprint parity test.

Pins the Python fingerprint output for a known input. The Go side has a
twin test (TestFingerprintGoldenAmbush in internal/ambush/backtest/
fingerprint_test.go) that pins the same hex.

Any divergence between the two hexes is a cross-language bug that will
surface as fingerprint_mismatch on /backtest/verify-job at upload time —
catch it here first.

Golden hex: 4f2b1aea88c8f69cd28345a77c07f41096e06e78e9bc55ffb297c1a87eab6090
  Captured from: Go TestFingerprintGoldenAmbush (test-v4 branch)
  Input: strategy_type=ambush, harness_version=v1, dataset_sha256=deadbeef,
         initial_capital=10000, params=(balanced / lev3 / positions 0.20 / etc.)
"""

from __future__ import annotations

import decimal
from decimal import Decimal
from pathlib import Path

import pytest

import sys
_scripts_dir = Path(__file__).resolve().parent.parent / "scripts"
if str(_scripts_dir) not in sys.path:
    sys.path.insert(0, str(_scripts_dir))

from ambush.fingerprint import canonical, dataset_sha256, CURRENT_HARNESS_VERSION


# ---------------------------------------------------------------------------
# Golden test — hex must match Go's TestFingerprintGoldenAmbush.
# ---------------------------------------------------------------------------

# Pinned by running Go TestFingerprintGoldenAmbush on test-v4 branch.
GOLDEN_HEX = "4f2b1aea88c8f69cd28345a77c07f41096e06e78e9bc55ffb297c1a87eab6090"


class _GoldenParams:
    """Duck-typed AmbushBotParams for the golden test input.
    Mirrors the Go test's domain.AmbushBotParams literal exactly.
    """
    direction = "balanced"

    class long:
        leverage = 3
        position_pct = Decimal("0.20")
        stop_loss_pct = Decimal("0.08")
        trailing_pct = Decimal("0.30")
        max_hold_hours = 30
        momentum_bars = 0
        cooldown_bars = 1
        sl_atr_mult = Decimal("2.0")

    class short:
        leverage = 3
        position_pct = Decimal("0.20")
        stop_loss_pct = Decimal("0.40")
        trailing_pct = Decimal("0.25")
        max_hold_hours = 168
        cooldown_bars = 15
        entry_delay_bars = 16
        sl_atr_mult = Decimal("2.5")

    class rhythm:
        max_trades_per_event = 1
        same_coin_dedup_days = 7


def test_known_input_known_hex():
    """The Python canonical() must produce GOLDEN_HEX for the standard input.

    This test and Go's TestFingerprintGoldenAmbush both pin the same hex
    string; if they diverge, one side drifted.
    """
    hex_actual = canonical(
        strategy_type="ambush",
        harness_version=CURRENT_HARNESS_VERSION,
        dataset_sha256="deadbeef",
        initial_capital=Decimal("10000"),
        params=_GoldenParams,
    )
    assert hex_actual == GOLDEN_HEX, (
        f"cross-language drift: py={hex_actual!r} want={GOLDEN_HEX!r}"
    )


def test_canonical_stable():
    """Same input → same hex on multiple calls (no non-determinism)."""
    p = _GoldenParams
    h1 = canonical("ambush", "v1", "abc123", Decimal("10000"), p)
    h2 = canonical("ambush", "v1", "abc123", Decimal("10000"), p)
    assert h1 == h2
    assert len(h1) == 64


def test_canonical_sensitivity():
    """Any field change must produce a different fingerprint."""
    base_hex = canonical("ambush", "v1", "abc123", Decimal("10000"), _GoldenParams)

    class LongChanged(_GoldenParams):
        class long(_GoldenParams.long):
            leverage = 5  # changed

    cases = [
        ("harness_version", lambda: canonical("ambush", "v2", "abc123", Decimal("10000"), _GoldenParams)),
        ("dataset_sha256",  lambda: canonical("ambush", "v1", "abc124", Decimal("10000"), _GoldenParams)),
        ("initial_capital", lambda: canonical("ambush", "v1", "abc123", Decimal("20000"), _GoldenParams)),
        ("long_leverage",   lambda: canonical("ambush", "v1", "abc123", Decimal("10000"), LongChanged)),
    ]
    for name, fn in cases:
        changed_hex = fn()
        assert changed_hex != base_hex, (
            f"fingerprint unchanged after mutating {name!r} — encoding is insensitive"
        )


# ---------------------------------------------------------------------------
# DatasetSHA256 stability test
# ---------------------------------------------------------------------------

_DATA_DIR = Path(__file__).resolve().parent.parent / "scripts" / "data_cache" / "ambush"


def test_dataset_sha256_stable():
    """Same dataset dir → same SHA on two calls (no randomness in file walk)."""
    if not _DATA_DIR.exists():
        pytest.skip(f"data_cache not found at {_DATA_DIR}")
    a = dataset_sha256(_DATA_DIR)
    b = dataset_sha256(_DATA_DIR)
    assert a == b, f"dataset SHA not stable: {a} vs {b}"
    assert len(a) == 64, f"SHA length={len(a)} want 64"
