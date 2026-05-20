"""Unit tests for ambush.features.compute_features — boundary cases for each
of the 5 balanced_decide_v0 rules."""
from __future__ import annotations

import os
import sys
import unittest

_HERE = os.path.dirname(os.path.abspath(__file__))
_SCRIPTS = os.path.dirname(_HERE)
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

from ambush import features


def _bars_with_last_jump(prev_close: float, last_open: float, last_close: float,
                        n: int = 96) -> list[dict]:
    """Build a 96-bar series where:
      - The 95th bar (index 94, "prev") closes at `prev_close`
      - The 96th bar (index 95, "current/last") opens at `last_open` and
        closes at `last_close`
      - All earlier bars close at a stable value so chg_24h_pct ≈
        (last_close - first_close) / first_close * 100
    """
    bars = []
    base = prev_close
    for _ in range(n - 1):
        bars.append({"open": base, "high": base, "low": base, "close": base})
    last_high = max(last_open, last_close) * 1.001
    last_low = min(last_open, last_close) * 0.999
    bars.append({
        "open": last_open, "high": last_high,
        "low": last_low, "close": last_close,
    })
    return bars


class TestComputeFeatures(unittest.TestCase):
    def test_long_momentum_init_shape(self):
        # surge ∈ (0.10, 0.15), chg_24h < 10
        # last bar: open=100, close=112 → surge=0.12
        # baseline chg: last_close vs bars[-96].close = 112 vs 100 → +12%? but
        # we want chg<10 — pick prev=110 so first_close=110, last=112 → +1.8%
        bars = _bars_with_last_jump(prev_close=110.0, last_open=100.0, last_close=112.0)
        f = features.compute_features(bars)
        self.assertAlmostEqual(f["surge_15m"], 0.12, places=4)
        self.assertLess(f["chg_24h_pct"], 10.0)

    def test_short_spike_extreme_shape(self):
        # surge > 0.25 (single-bar spike >25%)
        bars = _bars_with_last_jump(prev_close=100.0, last_open=100.0, last_close=130.0)
        f = features.compute_features(bars)
        self.assertGreater(f["surge_15m"], 0.25)

    def test_chg_24h_pct_full_range(self):
        # 24h change = (close - close_96_bars_ago) / close_96_bars_ago * 100.
        # 96 bars × 15m = 1440min = exactly 24h, so the first bar IS the
        # 24h baseline.
        bars = [{"open": 100, "high": 100, "low": 100, "close": 100} for _ in range(95)]
        bars.append({"open": 100, "high": 150, "low": 100, "close": 150})
        f = features.compute_features(bars)
        self.assertAlmostEqual(f["chg_24h_pct"], 50.0, places=4)

    def test_zero_open_returns_zero_surge_not_div_by_zero(self):
        # Pathological bar with open=0 must not crash; surge defaults to 0.
        bars = [{"open": 100, "high": 100, "low": 100, "close": 100} for _ in range(95)]
        bars.append({"open": 0, "high": 1, "low": 0, "close": 1})
        f = features.compute_features(bars)
        self.assertEqual(f["surge_15m"], 0.0)

    def test_rsi_falls_through(self):
        # compute_features delegates to indicators.compute_rsi; verify the
        # value lands in the expected 0..100 range.
        bars = [{"open": i, "high": i + 1, "low": i - 1, "close": i} for i in range(100, 196)]
        f = features.compute_features(bars)
        self.assertGreaterEqual(f["rsi_14"], 0.0)
        self.assertLessEqual(f["rsi_14"], 100.0)

    def test_raises_when_not_enough_bars(self):
        bars = [{"open": 100, "high": 100, "low": 100, "close": 100} for _ in range(10)]
        with self.assertRaises(ValueError):
            features.compute_features(bars)


if __name__ == "__main__":
    unittest.main()
