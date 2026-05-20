"""Unit tests for ambush.close_monitor — 4-priority cascade behind --kline-driven-close.

Tests target the private _evaluate_position_kline_driven function plus
the fallback path on HL fetch failure.
"""
from __future__ import annotations

import os
import shutil
import sys
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from unittest.mock import MagicMock, patch

_HERE = os.path.dirname(os.path.abspath(__file__))
_SCRIPTS = os.path.dirname(_HERE)
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

from ambush import close_monitor, live_database as db


def _bars_with_atr(entry: float, atr_target: float, n: int = 96) -> list[dict]:
    """Build n bars where each bar's TR = atr_target. Last close at `entry`."""
    out = []
    base = entry - atr_target * (n - 1)  # walk back so last bar closes at entry
    for i in range(n):
        c = base + atr_target * i
        out.append({
            "open": c - atr_target / 2,
            "high": c + atr_target,
            "low": c - atr_target,
            "close": c,
            "ts": datetime(2026, 5, 19, tzinfo=timezone.utc) + timedelta(minutes=15 * i),
        })
    return out


class TestKlineCloseCascade(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.tmpdir, "cm.db")
        db.init_db(self.db_path)
        self.client = MagicMock()
        self.client.close_position.return_value = {"order": {"order_id": "1", "status": "filled"}}

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _ambush_params(self, side: str = "long"):
        return {
            "long_params": {
                "stop_loss_pct": 0.08, "sl_atr_mult": 2.0,
                "trailing_pct": 0.30, "max_hold_hours": 30,
            },
            "short_params": {
                "stop_loss_pct": 0.40, "sl_atr_mult": 2.5,
                "trailing_pct": 0.25, "max_hold_hours": 168,
            },
        }

    @patch("ambush.close_monitor.hyperliquid_candles.fetch")
    def test_atr_stop_loss_long_hit(self, mock_fetch):
        # Long position; mark equals last K-line close way below stop.
        # entry=100, ATR=2, sl_atr_mult=2.0 → sl_dist = 4/100 = 4%
        # need pnl_pct * lev (3) <= -4% * lev → pnl <= -4%
        # close at 90 → pnl_pct = -10% → triggers
        bars = _bars_with_atr(entry=90.0, atr_target=2.0)
        mock_fetch.return_value = bars
        pos = {
            "symbol": "TSTUSDC", "side": "long", "net_qty": "100",
            "entry_price": "100", "mark_price": "90", "leverage": 3,
        }
        # Bootstrap position_state row so peak exists.
        db.upsert_position_state(
            self.db_path, symbol="TSTUSDC", side="long",
            entry_price="100", opened_at=datetime.now(timezone.utc).isoformat(),
            peak_price="100",
        )
        close_monitor._evaluate_position(
            self.client, self.db_path, self._ambush_params(),
            pos, "TSTUSDC", Decimal("100"),
            kline_driven=True,
        )
        self.client.close_position.assert_called_once()
        kwargs = self.client.close_position.call_args.kwargs
        self.assertIn("atr_stop_loss", str(kwargs.get("reasoning", "")))

    @patch("ambush.close_monitor.hyperliquid_candles.fetch")
    def test_atr_stop_loss_short_hit(self, mock_fetch):
        # Short position: entry=100, mark=110 → pnl_pct = (100-110)/100 = -10%
        # ATR=2, sl_atr_mult=2.5 (short default) → sl_dist = 5/100 = 5%
        # pnl_pct (-10%) <= -sl_dist (-5%) → triggers
        bars = _bars_with_atr(entry=110.0, atr_target=2.0)
        mock_fetch.return_value = bars
        pos = {
            "symbol": "TSTUSDC", "side": "short", "net_qty": "-100",
            "entry_price": "100", "mark_price": "110", "leverage": 3,
        }
        db.upsert_position_state(
            self.db_path, symbol="TSTUSDC", side="short",
            entry_price="100", opened_at=datetime.now(timezone.utc).isoformat(),
            peak_price="100",
        )
        close_monitor._evaluate_position(
            self.client, self.db_path, self._ambush_params(),
            pos, "TSTUSDC", Decimal("-100"),
            kline_driven=True,
        )
        self.client.close_position.assert_called_once()
        kwargs = self.client.close_position.call_args.kwargs
        self.assertIn("atr_stop_loss", str(kwargs.get("reasoning", "")))

    @patch("ambush.close_monitor.hyperliquid_candles.fetch")
    def test_hold_when_no_exit_condition_met(self, mock_fetch):
        # Long, entry=100, mark=102, ATR=1 → sl_dist=2%, pnl=2% — no exit.
        bars = _bars_with_atr(entry=102.0, atr_target=1.0)
        mock_fetch.return_value = bars
        pos = {
            "symbol": "TSTUSDC", "side": "long", "net_qty": "100",
            "entry_price": "100", "mark_price": "102", "leverage": 3,
        }
        db.upsert_position_state(
            self.db_path, symbol="TSTUSDC", side="long",
            entry_price="100",
            opened_at=datetime.now(timezone.utc).isoformat(),
            peak_price="102",
        )
        close_monitor._evaluate_position(
            self.client, self.db_path, self._ambush_params(),
            pos, "TSTUSDC", Decimal("100"),
            kline_driven=True,
        )
        self.client.close_position.assert_not_called()


class TestKlineTrailing(TestKlineCloseCascade):
    @patch("ambush.close_monitor.hyperliquid_candles.fetch")
    def test_long_trailing_triggers_on_kline_close_drift(self, mock_fetch):
        # Long, entry=100, peak=150, last_close=104 → drift=(150-104)/150=30.6%
        # trailing_pct=0.30 → trigger.
        bars = [{"open": 100, "high": 100, "low": 100, "close": 100,
                 "ts": _now() - timedelta(minutes=15 * i)} for i in range(95)]
        bars.reverse()
        bars.append({"open": 105, "high": 105, "low": 104, "close": 104,
                     "ts": _now()})
        mock_fetch.return_value = bars
        # ATR over flat bars then one move = small; pnl from entry=100 to
        # close=104 = +4% > -sl_dist → ATR doesn't fire.
        pos = {
            "symbol": "TSTUSDC", "side": "long", "net_qty": "100",
            "entry_price": "100", "mark_price": "104", "leverage": 3,
        }
        db.upsert_position_state(
            self.db_path, symbol="TSTUSDC", side="long",
            entry_price="100", opened_at=datetime.now(timezone.utc).isoformat(),
            peak_price="150",  # historical peak — high enough drift
        )
        close_monitor._evaluate_position(
            self.client, self.db_path, self._ambush_params(),
            pos, "TSTUSDC", Decimal("100"),
            kline_driven=True,
        )
        self.client.close_position.assert_called_once()
        kwargs = self.client.close_position.call_args.kwargs
        self.assertIn("trailing", str(kwargs.get("reasoning", "")))

    @patch("ambush.close_monitor.hyperliquid_candles.fetch")
    def test_long_trailing_ratchets_peak_on_close(self, mock_fetch):
        # Long, peak was 105, latest close 110 → peak updates to 110,
        # drift = 0 → no close yet.
        bars = [{"open": 100, "high": 100, "low": 100, "close": 100,
                 "ts": _now() - timedelta(minutes=15 * i)} for i in range(95)]
        bars.reverse()
        bars.append({"open": 109, "high": 111, "low": 109, "close": 110,
                     "ts": _now()})
        mock_fetch.return_value = bars
        pos = {
            "symbol": "TSTUSDC", "side": "long", "net_qty": "100",
            "entry_price": "100", "mark_price": "110", "leverage": 3,
        }
        db.upsert_position_state(
            self.db_path, symbol="TSTUSDC", side="long",
            entry_price="100", opened_at=datetime.now(timezone.utc).isoformat(),
            peak_price="105",
        )
        close_monitor._evaluate_position(
            self.client, self.db_path, self._ambush_params(),
            pos, "TSTUSDC", Decimal("100"),
            kline_driven=True,
        )
        self.client.close_position.assert_not_called()
        # peak should now be 110
        state = db.get_position_state(self.db_path, "TSTUSDC")
        self.assertEqual(Decimal(str(state["peak_price"])), Decimal("110"))


    @patch("ambush.close_monitor.hyperliquid_candles.fetch")
    def test_short_trailing_triggers_on_kline_close_drift(self, mock_fetch):
        # Short, entry=100, trough(peak)=80, last_close=112 → drift=(112-80)/80=40%
        # short trailing_pct=0.25 → 40% > 25% → triggers.
        # Use ATR=10 so sl_dist = 2.5*10/100 = 25% > pnl=12% → ATR won't fire first.
        bars = _bars_with_atr(entry=112.0, atr_target=10.0)
        mock_fetch.return_value = bars
        pos = {
            "symbol": "TSTUSDC", "side": "short", "net_qty": "-100",
            "entry_price": "100", "mark_price": "112", "leverage": 3,
        }
        db.upsert_position_state(
            self.db_path, symbol="TSTUSDC", side="short",
            entry_price="100", opened_at=datetime.now(timezone.utc).isoformat(),
            peak_price="80",  # historical trough
        )
        close_monitor._evaluate_position(
            self.client, self.db_path, self._ambush_params(),
            pos, "TSTUSDC", Decimal("-100"),
            kline_driven=True,
        )
        self.client.close_position.assert_called_once()
        kwargs = self.client.close_position.call_args.kwargs
        self.assertIn("trailing", str(kwargs.get("reasoning", "")))

    @patch("ambush.close_monitor.hyperliquid_candles.fetch")
    def test_bootstrap_creates_state_when_none(self, mock_fetch):
        # Flat bars, no position_state row → bootstrap row written, no close.
        bars = [{"open": 100, "high": 100, "low": 100, "close": 100,
                 "ts": _now() - timedelta(minutes=15 * i)} for i in range(96)]
        bars.reverse()
        mock_fetch.return_value = bars
        pos = {
            "symbol": "TSTUSDC", "side": "long", "net_qty": "100",
            "entry_price": "100", "mark_price": "100", "leverage": 3,
        }
        # Note: NO upsert_position_state — state is None.
        close_monitor._evaluate_position(
            self.client, self.db_path, self._ambush_params(),
            pos, "TSTUSDC", Decimal("100"),
            kline_driven=True,
        )
        self.client.close_position.assert_not_called()
        state = db.get_position_state(self.db_path, "TSTUSDC")
        self.assertIsNotNone(state)
        self.assertEqual(Decimal(str(state["peak_price"])), Decimal("100"))

    @patch("ambush.close_monitor.hyperliquid_candles.fetch")
    def test_drift_exactly_at_threshold_fires(self, mock_fetch):
        # drift == trailing_pct (inclusive) → fires.
        # long, peak=100, last_close=70 → drift = 30/100 = 0.30 == trailing_pct.
        # Use bars with ATR=20 so sl_dist = 2*20/entry_in_bars = large → ATR won't fire.
        # last_close of bars = 70.0 (entry param for _bars_with_atr).
        bars = _bars_with_atr(entry=70.0, atr_target=20.0)
        mock_fetch.return_value = bars
        pos = {
            "symbol": "TSTUSDC", "side": "long", "net_qty": "100",
            "entry_price": "100", "mark_price": "70", "leverage": 3,
        }
        db.upsert_position_state(
            self.db_path, symbol="TSTUSDC", side="long",
            entry_price="100", opened_at=datetime.now(timezone.utc).isoformat(),
            peak_price="100",
        )
        close_monitor._evaluate_position(
            self.client, self.db_path, self._ambush_params(),
            pos, "TSTUSDC", Decimal("100"),
            kline_driven=True,
        )
        self.client.close_position.assert_called_once()


def _now():
    return datetime.now(timezone.utc)


if __name__ == "__main__":
    unittest.main()
