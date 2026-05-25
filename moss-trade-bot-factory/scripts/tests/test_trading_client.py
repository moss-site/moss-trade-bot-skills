"""Unit tests for TradingClient helpers — no network, just pure logic.

Symbol normalization and reasoning-pair preservation are pre-network gates;
this test pins them so refactors of either helper trigger a clear failure
rather than a silent change in order semantics.

Run from repo root:
    python3 -m unittest discover -s skill/production/scripts/tests
"""

from __future__ import annotations

import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPTS = os.path.dirname(HERE)
if SCRIPTS not in sys.path:
    sys.path.insert(0, SCRIPTS)

from trading_client import TradingClient


class NormalizeSymbolTest(unittest.TestCase):
    """Strip separators and uppercase; do NOT cross USDT/USDC quotes (1.0.24-dev
    is strict on quote currency)."""

    def test_btc_usdc_unchanged(self):
        self.assertEqual(TradingClient._normalize_symbol("BTCUSDC"), "BTCUSDC")

    def test_strips_slash(self):
        self.assertEqual(TradingClient._normalize_symbol("ETH/USDC"), "ETHUSDC")

    def test_strips_colon(self):
        self.assertEqual(TradingClient._normalize_symbol("BTC:USDC"), "BTCUSDC")

    def test_strips_dash(self):
        self.assertEqual(TradingClient._normalize_symbol("DOGE-USDC"), "DOGEUSDC")

    def test_uppercases(self):
        self.assertEqual(TradingClient._normalize_symbol("eth/usdc"), "ETHUSDC")

    def test_mixed_case_and_separator(self):
        self.assertEqual(TradingClient._normalize_symbol("Sol-USDC"), "SOLUSDC")

    def test_usdt_and_usdc_not_collapsed(self):
        """Per 1.0.24-dev design: BTCUSDT != BTCUSDC. Cross-quote matching is
        the platform's job, not the client's."""
        self.assertNotEqual(
            TradingClient._normalize_symbol("BTCUSDT"),
            TradingClient._normalize_symbol("BTCUSDC"),
        )


if __name__ == "__main__":
    unittest.main()
