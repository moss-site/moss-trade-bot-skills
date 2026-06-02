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


class ClosePositionSymbolScopeTest(unittest.TestCase):
    """Regression for the ambush kline-driven close path, which calls
    ``client.close_position(symbol=...)``. A MagicMock client masks the
    signature mismatch, so this exercises the REAL TradingClient: the
    method must accept ``symbol``, scope ``self.symbol`` for the close, and
    restore it afterwards. Before the fix this raised TypeError, which
    ``close_monitor._close_now`` swallowed → the position was never closed."""

    def _client(self) -> TradingClient:
        return TradingClient(
            api_key="k", api_secret="s",
            base_url="http://localhost:8088",
            bot_id="agt_test", symbol="ORIGUSDC",
        )

    def test_symbol_kwarg_scopes_and_restores(self):
        tc = self._client()
        seen = {}

        def fake_resolve(position_side):
            seen["symbol_during_close"] = tc.symbol
            return {"side": "LONG", "qty": "10", "leverage": 3}

        tc._resolve_open_position = fake_resolve
        tc._submit_market_order = lambda *a, **k: {"order": {"order_id": "1", "status": "filled"}}

        out = tc.close_position(position_side="LONG", symbol="SAGAUSDC", reasoning="x")

        self.assertEqual(out["order"]["order_id"], "1")
        self.assertEqual(seen["symbol_during_close"], "SAGAUSDC")  # scoped during the close
        self.assertEqual(tc.symbol, "ORIGUSDC")                    # restored afterwards

    def test_symbol_restored_even_on_error(self):
        tc = self._client()

        def boom(position_side):
            raise ValueError("no open position")

        tc._resolve_open_position = boom
        with self.assertRaises(ValueError):
            tc.close_position(position_side="LONG", symbol="SAGAUSDC")
        self.assertEqual(tc.symbol, "ORIGUSDC")  # finally-block restores on exception

    def test_no_symbol_keeps_current(self):
        tc = self._client()
        tc._resolve_open_position = lambda side: {"side": "LONG", "qty": "5", "leverage": 1}
        tc._submit_market_order = lambda *a, **k: {"order": {"order_id": "2", "status": "filled"}}
        tc.close_position(position_side="LONG")  # no symbol → unchanged
        self.assertEqual(tc.symbol, "ORIGUSDC")


if __name__ == "__main__":
    unittest.main()
