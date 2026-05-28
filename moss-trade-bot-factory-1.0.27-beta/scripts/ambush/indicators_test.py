"""Unit tests for ambush.indicators — pure-Python ATR(14) + RSI(14)."""
from __future__ import annotations

import os
import sys
import unittest

_HERE = os.path.dirname(os.path.abspath(__file__))
_SCRIPTS = os.path.dirname(_HERE)
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

from ambush import indicators as ind


def _bar(open_, high, low, close):
    return {"open": float(open_), "high": float(high), "low": float(low), "close": float(close)}


class TestATR(unittest.TestCase):
    def test_constant_bars_gives_zero_atr(self):
        # 20 identical bars: TR = 0 every bar, ATR must be 0.
        bars = [_bar(100, 100, 100, 100) for _ in range(20)]
        self.assertAlmostEqual(ind.compute_atr(bars, period=14), 0.0, places=8)

    def test_known_atr_sequence(self):
        # 16 bars where every bar has high-low=10 and prev_close is bar's close.
        # TR = high - low = 10 every bar => EWM-smoothed ATR converges to 10.
        bars = [_bar(100, 110, 100, 105) for _ in range(16)]
        atr = ind.compute_atr(bars, period=14)
        self.assertAlmostEqual(atr, 10.0, places=4)

    def test_raises_when_insufficient_bars(self):
        bars = [_bar(100, 110, 100, 105) for _ in range(10)]
        with self.assertRaises(ValueError):
            ind.compute_atr(bars, period=14)


class TestRSI(unittest.TestCase):
    def test_all_gains_gives_high_rsi(self):
        closes = [100 + i for i in range(20)]  # strictly increasing
        bars = [_bar(c - 1, c + 0.5, c - 1.5, c) for c in closes]
        rsi = ind.compute_rsi(bars, period=14)
        self.assertGreater(rsi, 70.0)  # strong uptrend → RSI > 70
        self.assertLessEqual(rsi, 100.0)

    def test_all_losses_gives_low_rsi(self):
        closes = [120 - i for i in range(20)]  # strictly decreasing
        bars = [_bar(c + 1, c + 1.5, c - 0.5, c) for c in closes]
        rsi = ind.compute_rsi(bars, period=14)
        self.assertLess(rsi, 30.0)  # strong downtrend → RSI < 30
        self.assertGreaterEqual(rsi, 0.0)

    def test_neutral_oscillation_around_50(self):
        # alternating +1 / -1 closes → RSI hovers around 50
        closes = []
        for i in range(20):
            closes.append(100 + (i % 2))
        bars = [_bar(c, c + 0.5, c - 0.5, c) for c in closes]
        rsi = ind.compute_rsi(bars, period=14)
        self.assertGreater(rsi, 40.0)
        self.assertLess(rsi, 60.0)

    def test_raises_when_insufficient_bars(self):
        bars = [_bar(100, 101, 99, 100) for _ in range(10)]
        with self.assertRaises(ValueError):
            ind.compute_rsi(bars, period=14)


if __name__ == "__main__":
    unittest.main()
