"""Regression tests for Go verifier parity helpers.

Run from repo root:
    python3 -m unittest discover -s skill/production/scripts/tests
"""

from __future__ import annotations

import os
import sys
import unittest

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPTS = os.path.dirname(HERE)
if SCRIPTS not in sys.path:
    sys.path.insert(0, SCRIPTS)

from core.backtest import _go_replay_quote_at


class GoReplayQuoteParityTest(unittest.TestCase):
    def test_exact_15m_boundary_uses_first_synthetic_1m_close(self) -> None:
        df = pd.DataFrame(
            [
                {"timestamp": "2025-10-06T00:00:00Z", "open": 100.0, "high": 116.0, "low": 99.0, "close": 115.0, "volume": 15.0},
                {"timestamp": "2025-10-06T00:15:00Z", "open": 115.0, "high": 131.0, "low": 114.0, "close": 130.0, "volume": 15.0},
            ]
        )
        step = pd.Timedelta(minutes=15)

        mark, idx = _go_replay_quote_at(df, pd.Timestamp("2025-10-06T00:15:00Z"), step)

        self.assertEqual(idx, 1)
        self.assertEqual(mark, 116.0)

    def test_inside_15m_bar_uses_last_synthetic_minute_at_or_before_time(self) -> None:
        df = pd.DataFrame(
            [
                {"timestamp": "2025-10-06T00:00:00Z", "open": 100.0, "high": 116.0, "low": 99.0, "close": 115.0, "volume": 15.0},
            ]
        )
        step = pd.Timedelta(minutes=15)

        mark, idx = _go_replay_quote_at(df, pd.Timestamp("2025-10-06T00:14:30Z"), step)

        self.assertEqual(idx, 0)
        self.assertEqual(mark, 115.0)


if __name__ == "__main__":
    unittest.main()
