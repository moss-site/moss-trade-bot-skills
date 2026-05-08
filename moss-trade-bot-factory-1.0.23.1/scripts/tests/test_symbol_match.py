"""Unit tests for cross-quote symbol matching (BTCUSDT == BTCUSDC).

Regression test for the e2e issue where _resolve_open_position rejected
the open position because the user passed --symbol BTCUSDT but the platform
stored the position as BTCUSDC.

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


class BaseAssetTest(unittest.TestCase):
    def test_strips_usdt(self):
        self.assertEqual(TradingClient._base_asset("BTCUSDT"), "BTC")

    def test_strips_usdc(self):
        self.assertEqual(TradingClient._base_asset("BTCUSDC"), "BTC")

    def test_strips_busd(self):
        self.assertEqual(TradingClient._base_asset("BTCBUSD"), "BTC")

    def test_strips_usd(self):
        self.assertEqual(TradingClient._base_asset("BTCUSD"), "BTC")

    def test_handles_slash(self):
        self.assertEqual(TradingClient._base_asset("BTC/USDT"), "BTC")
        self.assertEqual(TradingClient._base_asset("ETH/USDC"), "ETH")

    def test_handles_colon_perp_form(self):
        self.assertEqual(TradingClient._base_asset("BTC/USDC:USDC"), "BTC")

    def test_lowercase_normalized(self):
        self.assertEqual(TradingClient._base_asset("btcusdt"), "BTC")

    def test_unknown_quote_left_intact(self):
        self.assertEqual(TradingClient._base_asset("FOOBAR"), "FOOBAR")

    def test_pure_base_left_intact(self):
        self.assertEqual(TradingClient._base_asset("BTC"), "BTC")


class SymbolsMatchTest(unittest.TestCase):
    def test_btcusdt_matches_btcusdc(self):
        """The exact regression: e2e agent's close failed because BTCUSDT != BTCUSDC literally."""
        self.assertTrue(TradingClient._symbols_match("BTCUSDT", "BTCUSDC"))

    def test_btcusdt_matches_slash_form(self):
        self.assertTrue(TradingClient._symbols_match("BTCUSDT", "BTC/USDT"))
        self.assertTrue(TradingClient._symbols_match("BTC/USDT", "BTC/USDC"))

    def test_btcusdt_matches_perp_form(self):
        self.assertTrue(TradingClient._symbols_match("BTCUSDT", "BTC/USDC:USDC"))

    def test_different_base_does_not_match(self):
        self.assertFalse(TradingClient._symbols_match("BTCUSDT", "ETHUSDT"))
        self.assertFalse(TradingClient._symbols_match("BTCUSDC", "ETHUSDC"))

    def test_pure_base_matches_quoted(self):
        self.assertTrue(TradingClient._symbols_match("BTC", "BTCUSDT"))

    def test_case_insensitive(self):
        self.assertTrue(TradingClient._symbols_match("btcusdt", "BTCUSDC"))


if __name__ == "__main__":
    unittest.main()
