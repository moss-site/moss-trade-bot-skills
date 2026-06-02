"""
Focused tests for Ambush live event handling helpers.

Run:
    cd scripts && python3 -m unittest ambush.event_handler_test
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
import unittest
import json
from datetime import datetime, timezone, timedelta
from unittest.mock import MagicMock, patch
from urllib.request import urlopen


def _utc(i: int) -> datetime:
    return datetime(2026, 5, 19, tzinfo=timezone.utc) + timedelta(minutes=15 * i)

_HERE = os.path.dirname(os.path.abspath(__file__))
_SCRIPTS = os.path.dirname(_HERE)
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

from ambush import event_handler
from ambush import action_history_server
from ambush import live_database as db
from ambush import decision as decision_mod
from decimal import Decimal


class TestTimestampParsing(unittest.TestCase):
    def test_parse_trigger_ts_accepts_zulu(self):
        got = event_handler._parse_trigger_ts(
            {"trigger_ts": "2026-05-18T10:11:12Z"}
        )
        self.assertIsNotNone(got)
        self.assertEqual(got.tzinfo, timezone.utc)
        self.assertEqual(got.isoformat(), "2026-05-18T10:11:12+00:00")

    def test_parse_trigger_ts_rejects_malformed(self):
        self.assertIsNone(event_handler._parse_trigger_ts({"trigger_ts": "bad"}))


class TestDeferredOpenQueue(unittest.TestCase):
    def test_pending_and_running_deferred_open_rows_resume(self):
        with tempfile.TemporaryDirectory() as td:
            db_path = os.path.join(td, "ambush.sqlite")
            db.init_db(db_path)
            event = {
                "event_id": 101,
                "hl_symbol": "HYPE",
                "trigger_ts": "2026-05-18T10:11:12Z",
            }
            self.assertTrue(db.upsert_event(db_path, event))
            db.record_decision(
                db_path,
                101,
                "long",
                "user_config",
                order_status="deferred_open",
            )
            db.enqueue_deferred_open(
                db_path,
                event_id=101,
                due_at="2026-05-18T10:26:12+00:00",
                delay_param_name="momentum_bars",
                delay_bars=1,
            )

            rows = db.list_pending_deferred_opens(db_path)
            self.assertEqual([r["event_id"] for r in rows], [101])
            self.assertEqual(rows[0]["event"]["hl_symbol"], "HYPE")

            db.mark_deferred_open_running(db_path, 101)
            rows = db.list_pending_deferred_opens(db_path)
            self.assertEqual([r["event_id"] for r in rows], [101])

            db.mark_deferred_open_done(db_path, 101, status="completed")
            self.assertEqual(db.list_pending_deferred_opens(db_path), [])


class TestSymbolActionHistory(unittest.TestCase):
    def test_list_symbol_action_history_filters_recent_rows(self):
        with tempfile.TemporaryDirectory() as td:
            db_path = os.path.join(td, "ambush.sqlite")
            db.init_db(db_path)
            db.record_symbol_action(db_path, "hype", "open", event_id=1, order_id="o1")
            db.record_symbol_action(db_path, "HYPE", "close", event_id=1, order_id="o2")
            db.record_symbol_action(db_path, "BTC", "open", event_id=2, order_id="o3")

            rows = db.list_symbol_action_history(db_path, hl_symbol="hype", limit=10)
            self.assertEqual([r["action"] for r in rows], ["close", "open"])
            self.assertEqual({r["hl_symbol"] for r in rows}, {"HYPE"})

            rows = db.list_symbol_action_history(db_path, action="open", limit=10)
            self.assertEqual([r["order_id"] for r in rows], ["o3", "o1"])

            with self.assertRaises(ValueError):
                db.list_symbol_action_history(db_path, action="hold")

    def test_action_history_rest_reads_local_sqlite(self):
        with tempfile.TemporaryDirectory() as td:
            db_path = os.path.join(td, "ambush.sqlite")
            db.init_db(db_path)
            db.record_symbol_action(db_path, "HYPE", "open", event_id=1, order_id="o1")
            server = action_history_server.start(db_path, host="127.0.0.1", port=0)
            try:
                host, port = server.server_address[:2]
                with urlopen(
                    f"http://{host}:{port}/symbol-action-history?symbol=hype&limit=5",
                    timeout=2,
                ) as resp:
                    body = json.loads(resp.read().decode("utf-8"))
                self.assertEqual(body["count"], 1)
                self.assertEqual(body["items"][0]["hl_symbol"], "HYPE")
                self.assertEqual(body["items"][0]["order_id"], "o1")
            finally:
                server.shutdown()
                server.server_close()


class TestRecomputeAudit(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.tmpdir, "audit.db")
        db.init_db(self.db_path)

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_columns_added_idempotently(self):
        # Re-init should not raise even though columns already exist.
        db.init_db(self.db_path)
        with db.get_conn(self.db_path) as conn:
            cols = {row[1] for row in conn.execute("PRAGMA table_info(decisions)")}
        for c in ("recompute_surge_15m", "recompute_rsi_14", "recompute_chg_24h_pct"):
            self.assertIn(c, cols)

    def test_record_recompute_audit_updates_existing_row(self):
        db.upsert_event(self.db_path, {
            "event_id": 42,
            "hl_symbol": "BTC",
            "trigger_ts": "2026-05-20T00:00:00Z",
        })
        db.record_decision(
            self.db_path, 42, "long", "rule_long_momentum_init",
            order_status="placed",
        )
        db.record_recompute_audit(
            self.db_path, 42,
            surge_15m=0.123, rsi_14=55.5, chg_24h_pct=8.2,
        )
        with db.get_conn(self.db_path) as conn:
            row = conn.execute(
                "SELECT recompute_surge_15m, recompute_rsi_14, recompute_chg_24h_pct "
                "FROM decisions WHERE event_id=?", (42,)
            ).fetchone()
        self.assertAlmostEqual(row[0], 0.123, places=6)
        self.assertAlmostEqual(row[1], 55.5, places=4)
        self.assertAlmostEqual(row[2], 8.2, places=4)

    def test_record_recompute_audit_no_decision_row_is_noop(self):
        # event_id=99 was never decided; UPDATE affects 0 rows, no exception.
        db.record_recompute_audit(
            self.db_path, 99,
            surge_15m=0.0, rsi_14=50.0, chg_24h_pct=0.0,
        )
        # Confirm no phantom row was inserted.
        with db.get_conn(self.db_path) as conn:
            count = conn.execute(
                "SELECT COUNT(*) FROM decisions WHERE event_id=?", (99,)
            ).fetchone()[0]
        self.assertEqual(count, 0)


    def test_record_recompute_audit_accepts_none_values(self):
        db.upsert_event(self.db_path, {
            "event_id": 7,
            "hl_symbol": "ETH",
            "trigger_ts": "2026-05-20T00:00:00Z",
        })
        db.record_decision(
            self.db_path, 7, "short", "rule_short_momentum_init",
            order_status="placed",
        )
        db.record_recompute_audit(
            self.db_path, 7,
            surge_15m=None, rsi_14=None, chg_24h_pct=None,
        )
        with db.get_conn(self.db_path) as conn:
            row = conn.execute(
                "SELECT recompute_surge_15m, recompute_rsi_14, recompute_chg_24h_pct "
                "FROM decisions WHERE event_id=?", (7,)
            ).fetchone()
        self.assertIsNone(row[0])
        self.assertIsNone(row[1])
        self.assertIsNone(row[2])


class TestKlineDrivenOpen(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.tmpdir, "kdo.db")
        db.init_db(self.db_path)

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _make_handler(self, kline_driven_open: bool):
        client = MagicMock()
        ambush_params = {
            "direction": "balanced",
            "long_params": {"leverage": 3, "position_pct": 0.20,
                            "stop_loss_pct": 0.08, "trailing_pct": 0.30,
                            "max_hold_hours": 30, "momentum_bars": 0,
                            "cooldown_bars": 1},
            "short_params": {"leverage": 3, "position_pct": 0.20,
                             "stop_loss_pct": 0.40, "trailing_pct": 0.25,
                             "max_hold_hours": 168, "cooldown_bars": 15,
                             "entry_delay_bars": 16},
            "rhythm": {"max_trades_per_event": 1, "same_coin_dedup_days": 0},
        }
        return event_handler.EventHandler(client, ambush_params,
                                          kline_driven_open=kline_driven_open)

    @patch("ambush.event_handler.hyperliquid_candles.fetch")
    def test_recompute_replaces_snapshot_when_flag_on(self, mock_fetch):
        # Envelope says rule_long_momentum_init shape; recompute produces SKIP.
        mock_fetch.return_value = [
            {"open": 100, "high": 100, "low": 100, "close": 100,
             "ts": _utc(i)} for i in range(96)
        ]  # All-flat bars → surge=0, rsi=100 (zero deltas), chg=0 → SKIP
        h = self._make_handler(kline_driven_open=True)
        now_ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        event = {
            "event_id": 1,
            "hl_symbol": "SAGA",
            "trigger_ts": now_ts,
            "trigger_price": "0.02",
            "surge_15m": "0.12",   # envelope says momentum_init
            "rsi_14": "55",
            "change_before_24h_pct": "8",
        }
        # Just call process_event and assert the audit columns are stamped
        # with the recompute values (not the snapshot).
        event_handler.process_event(h, self.db_path, event, source="ws_live")
        with db.get_conn(self.db_path) as conn:
            row = conn.execute(
                "SELECT decision, recompute_surge_15m, recompute_rsi_14, recompute_chg_24h_pct "
                "FROM decisions WHERE event_id=?", (1,)
            ).fetchone()
        self.assertEqual(row[0], "skip")
        self.assertAlmostEqual(row[1], 0.0, places=4)
        self.assertAlmostEqual(row[2], 100.0, places=1)  # RSI on flat bars
        self.assertAlmostEqual(row[3], 0.0, places=4)

    @patch("ambush.event_handler.hyperliquid_candles.fetch")
    def test_fetch_failure_falls_back_to_snapshot(self, mock_fetch):
        mock_fetch.side_effect = event_handler.hyperliquid_candles.HyperliquidCandleError("502")
        h = self._make_handler(kline_driven_open=True)
        now_ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        event = {
            "event_id": 2,
            "hl_symbol": "SAGA",
            "trigger_ts": now_ts,
            "trigger_price": "0.02",
            "surge_15m": "0.30",   # rule_short_spike_extreme
            "rsi_14": "50",
            "change_before_24h_pct": "5",
        }
        event_handler.process_event(h, self.db_path, event, source="ws_live")
        with db.get_conn(self.db_path) as conn:
            row = conn.execute(
                "SELECT decision, reason FROM decisions WHERE event_id=?", (2,)
            ).fetchone()
        # Falls back to envelope → short_spike_extreme would be deferred,
        # so decision is "short" with a deferred reason. Either way decision
        # is not "skip" (which is what happens on fetch error blocking).
        self.assertEqual(row[0], "short")

    def test_flag_off_uses_envelope_unchanged(self):
        h = self._make_handler(kline_driven_open=False)
        # Note: we don't even patch hyperliquid_candles.fetch — flag off
        # means we never call it. If we did, the test would error.
        now_ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        event = {
            "event_id": 3, "hl_symbol": "SAGA",
            "trigger_ts": now_ts, "trigger_price": "0.02",
            "surge_15m": "0.30", "rsi_14": "50", "change_before_24h_pct": "5",
        }
        event_handler.process_event(h, self.db_path, event, source="ws_live")
        with db.get_conn(self.db_path) as conn:
            row = conn.execute(
                "SELECT decision FROM decisions WHERE event_id=?", (3,)
            ).fetchone()
        self.assertEqual(row[0], "short")

    @patch("ambush.event_handler.hyperliquid_candles.fetch")
    def test_short_bars_falls_back_to_snapshot(self, mock_fetch):
        mock_fetch.return_value = [
            {"open": 100, "high": 100, "low": 100, "close": 100,
             "ts": _utc(i)} for i in range(10)  # < 15 → too few
        ]
        h = self._make_handler(kline_driven_open=True)
        now_ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        event = {
            "event_id": 4, "hl_symbol": "SAGA",
            "trigger_ts": now_ts, "trigger_price": "0.02",
            "surge_15m": "0.30", "rsi_14": "50", "change_before_24h_pct": "5",
        }
        event_handler.process_event(h, self.db_path, event, source="ws_live")
        with db.get_conn(self.db_path) as conn:
            row = conn.execute(
                "SELECT decision, recompute_surge_15m FROM decisions WHERE event_id=?", (4,)
            ).fetchone()
        # Falls back to envelope → short decision; audit columns must be NULL (no recompute)
        self.assertEqual(row[0], "short")
        self.assertIsNone(row[1])

    @patch("ambush.event_handler.hyperliquid_candles.fetch")
    def test_post_decision_called_after_record(self, mock_fetch):
        """C1: skill must POST decision back to server after every record_decision."""
        mock_fetch.return_value = [
            {"open": 100, "high": 100, "low": 100, "close": 100,
             "ts": _utc(i)} for i in range(96)
        ]
        client = MagicMock()
        client.post_ambush_decision.return_value = {"ok": True}
        handler = event_handler.EventHandler(
            client,
            {"long_params": {}, "short_params": {}, "rhythm": {}},
            kline_driven_open=True,
        )
        # Use a fresh trigger_ts to bypass stale-signal early-return
        now_ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        event = {
            "event_id": 999_001, "hl_symbol": "TST",
            "trigger_ts": now_ts, "trigger_price": "0.02",
            "surge_15m": "0.30", "rsi_14": "50", "change_before_24h_pct": "5",
        }
        event_handler.process_event(handler, self.db_path, event, source="ws_live")
        # Flat bars → recompute disagrees with envelope → decision = skip
        # Whatever the final decision, post_ambush_decision MUST be called.
        assert client.post_ambush_decision.called, "post_ambush_decision was not called"
        args = client.post_ambush_decision.call_args
        assert args.kwargs.get("event_id") == 999_001, args.kwargs
        assert args.kwargs.get("decision") in ("skip", "long", "short"), args.kwargs

    @patch("ambush.event_handler.hyperliquid_candles.fetch")
    def test_post_decision_failure_does_not_break_handler(self, mock_fetch):
        """C1: post_ambush_decision raising must NOT block the order path."""
        mock_fetch.return_value = [
            {"open": 100, "high": 100, "low": 100, "close": 100,
             "ts": _utc(i)} for i in range(96)
        ]
        client = MagicMock()
        client.post_ambush_decision.side_effect = RuntimeError("simulated network failure")
        handler = event_handler.EventHandler(
            client,
            {"long_params": {}, "short_params": {}, "rhythm": {}},
            kline_driven_open=True,
        )
        now_ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        event = {
            "event_id": 999_002, "hl_symbol": "TST",
            "trigger_ts": now_ts, "trigger_price": "0.02",
            "surge_15m": "0.30", "rsi_14": "50", "change_before_24h_pct": "5",
        }
        # Should NOT raise even though post_ambush_decision raises
        event_handler.process_event(handler, self.db_path, event, source="ws_live")
        # The decision should still be recorded locally
        with db.get_conn(self.db_path) as conn:
            row = conn.execute(
                "SELECT decision FROM decisions WHERE event_id=?", (999_002,)
            ).fetchone()
        assert row is not None, "local SQLite decision must still be recorded"


class TestDeferredOpenRecompute(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.tmpdir, "defer.db")
        db.init_db(self.db_path)

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _decision(self):
        return decision_mod.Decision(
            side=decision_mod.SIDE_SHORT,
            reason=decision_mod.REASON_SHORT_SPIKE_EXTREME,
        )

    @patch("ambush.event_handler.hyperliquid_candles.fetch")
    def test_wake_recomputes_and_skips_when_no_longer_matches(self, mock_fetch):
        # Fresh bars now show neutral/no rule fires.
        mock_fetch.return_value = [
            {"open": 100, "high": 100, "low": 100, "close": 100,
             "ts": _utc(i)} for i in range(96)
        ]
        client = MagicMock()
        handler = event_handler.EventHandler(
            client, {"long_params": {}, "short_params": {}, "rhythm": {}},
            kline_driven_open=True,
        )
        db.upsert_event(self.db_path, {
            "event_id": 42, "hl_symbol": "DYM",
            "trigger_ts": "2026-05-20T00:00:00Z",
        })
        db.record_decision(self.db_path, event_id=42, decision="short",
                           reason="rule_short_spike_extreme", order_status="deferred_open")
        db.enqueue_deferred_open(
            self.db_path, event_id=42,
            due_at=datetime.now(timezone.utc).isoformat(),
            delay_param_name="entry_delay_bars", delay_bars=16,
        )
        event_handler._run_deferred_open(
            handler, self.db_path, event_id=42, decision=self._decision(),
            target_symbol="DYMUSDC", notional=Decimal("100"), leverage=3,
            client_order_id="ambush-42", reasoning_zh="zh", reasoning_en="en",
            delay_seconds=0.0, sym_base="DYM",
        )
        client.open_short.assert_not_called()
        with db.get_conn(self.db_path) as conn:
            status = conn.execute(
                "SELECT status FROM deferred_opens WHERE event_id=?", (42,)
            ).fetchone()[0]
        self.assertEqual(status, "deferred_expired_recompute")

    @patch("ambush.event_handler.hyperliquid_candles.fetch")
    def test_wake_recomputes_and_fires_when_still_matches(self, mock_fetch):
        # Fresh bars still satisfy spike_extreme: last bar surge >0.25.
        mock_fetch.return_value = [
            {"open": 100, "high": 100, "low": 100, "close": 100,
             "ts": _utc(i)} for i in range(95)
        ] + [{"open": 100, "high": 140, "low": 100, "close": 135,
              "ts": _utc(95)}]
        client = MagicMock()
        client.open_short.return_value = {"order_id": "777"}
        handler = event_handler.EventHandler(
            client, {"long_params": {}, "short_params": {}, "rhythm": {}},
            kline_driven_open=True,
        )
        db.upsert_event(self.db_path, {
            "event_id": 43, "hl_symbol": "DYM",
            "trigger_ts": "2026-05-20T00:00:00Z",
        })
        db.record_decision(self.db_path, event_id=43, decision="short",
                           reason="rule_short_spike_extreme", order_status="deferred_open")
        db.enqueue_deferred_open(
            self.db_path, event_id=43,
            due_at=datetime.now(timezone.utc).isoformat(),
            delay_param_name="entry_delay_bars", delay_bars=16,
        )
        event_handler._run_deferred_open(
            handler, self.db_path, event_id=43, decision=self._decision(),
            target_symbol="DYMUSDC", notional=Decimal("100"), leverage=3,
            client_order_id="ambush-43", reasoning_zh="zh", reasoning_en="en",
            delay_seconds=0.0, sym_base="DYM",
        )
        client.open_short.assert_called_once()
        with db.get_conn(self.db_path) as conn:
            status = conn.execute(
                "SELECT status FROM deferred_opens WHERE event_id=?", (43,)
            ).fetchone()[0]
        self.assertNotEqual(status, "deferred_expired_recompute")

    @patch("ambush.event_handler.hyperliquid_candles.fetch")
    def test_wake_fetch_failure_fires_on_original_decision(self, mock_fetch):
        mock_fetch.side_effect = event_handler.hyperliquid_candles.HyperliquidCandleError("502")
        client = MagicMock()
        client.open_short.return_value = {"order_id": "888"}
        handler = event_handler.EventHandler(
            client, {"long_params": {}, "short_params": {}, "rhythm": {}},
            kline_driven_open=True,
        )
        db.upsert_event(self.db_path, {
            "event_id": 44, "hl_symbol": "DYM",
            "trigger_ts": datetime.now(timezone.utc).isoformat(),
        })
        db.record_decision(self.db_path, event_id=44, decision="short",
                           reason="rule_short_spike_extreme", order_status="deferred_open")
        db.enqueue_deferred_open(
            self.db_path, event_id=44,
            due_at=datetime.now(timezone.utc).isoformat(),
            delay_param_name="entry_delay_bars", delay_bars=16,
        )
        event_handler._run_deferred_open(
            handler, self.db_path, event_id=44, decision=self._decision(),
            target_symbol="DYMUSDC", notional=Decimal("100"), leverage=3,
            client_order_id="ambush-44", reasoning_zh="zh", reasoning_en="en",
            delay_seconds=0.0, sym_base="DYM",
        )
        client.open_short.assert_called_once()

    @patch("ambush.event_handler.hyperliquid_candles.fetch")
    def test_wake_short_bars_fires_on_original_decision(self, mock_fetch):
        # Only 10 bars — below 15-bar threshold for compute_features
        mock_fetch.return_value = [
            {"open": 100, "high": 100, "low": 100, "close": 100,
             "ts": _utc(i)} for i in range(10)
        ]
        client = MagicMock()
        client.open_short.return_value = {"order_id": "999"}
        handler = event_handler.EventHandler(
            client, {"long_params": {}, "short_params": {}, "rhythm": {}},
            kline_driven_open=True,
        )
        db.upsert_event(self.db_path, {
            "event_id": 45, "hl_symbol": "DYM",
            "trigger_ts": datetime.now(timezone.utc).isoformat(),
        })
        db.record_decision(self.db_path, event_id=45, decision="short",
                           reason="rule_short_spike_extreme", order_status="deferred_open")
        db.enqueue_deferred_open(
            self.db_path, event_id=45,
            due_at=datetime.now(timezone.utc).isoformat(),
            delay_param_name="entry_delay_bars", delay_bars=16,
        )
        event_handler._run_deferred_open(
            handler, self.db_path, event_id=45, decision=self._decision(),
            target_symbol="DYMUSDC", notional=Decimal("100"), leverage=3,
            client_order_id="ambush-45", reasoning_zh="zh", reasoning_en="en",
            delay_seconds=0.0, sym_base="DYM",
        )
        client.open_short.assert_called_once()


class TestPositionLockSerializesConcurrent(unittest.TestCase):
    """C2: handler._position_lock (RLock) serializes concurrent process_event calls.

    Without the lock, two events arriving simultaneously on WS + poller
    threads can both pass the single-position pre-check before either has
    written. With the lock, the second call blocks until the first releases.
    """

    def setUp(self):
        import threading as _threading
        import time as _time
        self._threading = _threading
        self._time = _time
        self.tmpdir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.tmpdir, "lock_test.db")
        db.init_db(self.db_path)

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_position_lock_is_reentrant(self):
        """RLock allows same-thread re-entry; recursive process_event calls
        from within a process_event (rare but possible via callbacks) don't
        deadlock. Also validates the lock field is present."""
        handler = event_handler.EventHandler(
            MagicMock(),
            {"long_params": {}, "short_params": {}, "rhythm": {}},
        )
        # Acquire twice from same thread — must not deadlock.
        with handler._position_lock:
            with handler._position_lock:
                pass  # re-entry must succeed

    def test_concurrent_process_event_calls_serialize_on_lock(self):
        """Two concurrent process_event calls must serialize via handler._position_lock.

        Thread A holds the lock (simulated via slow list_positions).
        Thread B must block at the lock acquisition and must NOT call
        list_positions until A releases. This verifies the race window is
        closed at the skill level. Server-side enforceAmbushPositionLock
        remains the cross-process safety net.
        """
        threading = self._threading
        time = self._time

        barrier_acquired = threading.Event()
        can_proceed = threading.Event()
        call_order = []

        # Slow down get_positions so the race window is observable:
        # thread A will be stuck inside the lock while thread B tries to enter.
        def slow_list_positions(*a, **kw):
            call_order.append("list_positions")
            barrier_acquired.set()      # signal that A is inside the lock
            can_proceed.wait(timeout=2.0)  # hold A until the test releases it
            return []                   # no positions → no lock reject

        client = MagicMock()
        # Return a real account dict so _compute_notional produces non-zero notional.
        client.get_account.return_value = {"wallet_balance": "1000"}
        # get_positions is the method called in the single-position pre-check.
        # Patch it with the slow version so thread A stays inside the lock.
        client.get_positions.side_effect = slow_list_positions
        client.open_short.return_value = {"order_id": "o1"}
        client.post_ambush_decision.return_value = {"ok": True}

        handler = event_handler.EventHandler(
            client,
            {
                # Non-empty params so _compute_notional picks up position_pct
                # and decision path reaches the single-position pre-check.
                "long_params": {"position_pct": "0.10", "leverage": 1},
                "short_params": {"position_pct": "0.10", "leverage": 1},
                "rhythm": {},
            },
            kline_driven_open=False,  # skip K-line fetch
        )

        now_ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        event_a = {
            "event_id": 9001, "hl_symbol": "AAA",
            "trigger_ts": now_ts, "trigger_price": "1",
            "surge_15m": "0.30", "rsi_14": "50", "change_before_24h_pct": "5",
        }
        event_b = {**event_a, "event_id": 9002, "hl_symbol": "BBB"}

        t_a = threading.Thread(
            target=event_handler.process_event,
            args=(handler, self.db_path, event_a, "ws_live"),
            name="test-thread-A",
        )
        t_b = threading.Thread(
            target=event_handler.process_event,
            args=(handler, self.db_path, event_b, "ws_live"),
            name="test-thread-B",
        )

        t_a.start()
        # Wait until thread A has acquired the lock and entered list_positions.
        barrier_acquired.wait(timeout=2.0)
        t_b.start()
        # Give thread B a moment to reach the lock acquisition attempt.
        time.sleep(0.1)

        # At this point thread A is still inside the locked region (blocked
        # on can_proceed). Thread B must be blocked on the lock — it should
        # NOT have called get_positions yet.
        positions_called_count = len([c for c in call_order if c == "list_positions"])
        self.assertEqual(
            positions_called_count, 1,
            f"expected thread B to be blocked on lock, but call_order={call_order}",
        )

        # Release thread A; both threads should now complete cleanly.
        can_proceed.set()
        t_a.join(timeout=3.0)
        t_b.join(timeout=3.0)
        self.assertFalse(t_a.is_alive(), "thread A should have completed")
        self.assertFalse(t_b.is_alive(), "thread B should have completed")


class TestPositionsCache(unittest.TestCase):
    """get_positions_cached collapses a burst of exit_signal lookups into one
    network call (fixes the bootstrap-drain backlog where 100+ broadcast
    exit_signals each hit get_positions). Invalidation after open/close keeps
    it from serving a stale snapshot across a position change."""

    def _make_handler(self):
        client = MagicMock()
        params = {
            "direction": "balanced",
            "long_params": {"leverage": 3, "position_pct": 0.20, "stop_loss_pct": 0.08,
                            "trailing_pct": 0.30, "max_hold_hours": 30, "momentum_bars": 0,
                            "cooldown_bars": 1},
            "short_params": {"leverage": 3, "position_pct": 0.20, "stop_loss_pct": 0.40,
                             "trailing_pct": 0.25, "max_hold_hours": 168, "cooldown_bars": 15,
                             "entry_delay_bars": 16},
            "rhythm": {"max_trades_per_event": 1, "same_coin_dedup_days": 7},
        }
        return event_handler.EventHandler(client, params), client

    def test_burst_collapses_to_one_fetch(self):
        h, client = self._make_handler()
        client.get_positions.return_value = []
        # 50 exit_signals in the same window → 1 network call
        for _ in range(50):
            self.assertEqual(h.get_positions_cached(ttl=5.0), [])
        self.assertEqual(client.get_positions.call_count, 1,
                         "burst within TTL must collapse to a single get_positions")

    def test_ttl_expiry_refetches(self):
        h, client = self._make_handler()
        client.get_positions.return_value = []
        h.get_positions_cached(ttl=0.0)  # ttl=0 → always stale
        h.get_positions_cached(ttl=0.0)
        self.assertEqual(client.get_positions.call_count, 2,
                         "expired cache must refetch")

    def test_invalidate_forces_refetch(self):
        h, client = self._make_handler()
        client.get_positions.return_value = []
        h.get_positions_cached(ttl=600)          # fetch 1, cached
        h.get_positions_cached(ttl=600)          # served from cache
        self.assertEqual(client.get_positions.call_count, 1)
        h.invalidate_positions_cache()           # position changed
        h.get_positions_cached(ttl=600)          # fetch 2
        self.assertEqual(client.get_positions.call_count, 2,
                         "invalidate must drop cache so next call refetches")

    def test_fetch_error_propagates(self):
        h, client = self._make_handler()
        client.get_positions.side_effect = RuntimeError("network down")
        with self.assertRaises(RuntimeError):
            h.get_positions_cached()


class ClassifyCloseResultTransientTest(unittest.TestCase):
    """Regression: a server STALE_MARK_PRICE on the reduce_only close path
    (ambush small-cap with a momentarily cold/stale mark) must be treated as
    TRANSIENT so a redelivery retries, instead of being marked terminal —
    which would mark the exit processed and leave the position open."""

    def test_stale_mark_price_is_transient(self):
        for msg in ("mark price is stale", "mark price unavailable",
                    "mark price window is unstable"):
            action, _order_id, _err, transient = event_handler._classify_close_result(
                {"code": "STALE_MARK_PRICE", "message": msg})
            self.assertEqual(action, "failed")
            self.assertTrue(transient, f"STALE_MARK_PRICE ({msg!r}) must retry")

    def test_unknown_terminal_code_not_transient(self):
        action, _order_id, _err, transient = event_handler._classify_close_result(
            {"code": "INVALID_ARGUMENT", "message": "bad qty"})
        self.assertEqual(action, "failed")
        self.assertFalse(transient)


if __name__ == "__main__":
    unittest.main()
