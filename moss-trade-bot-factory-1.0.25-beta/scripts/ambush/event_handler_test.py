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
from datetime import timezone
from urllib.request import urlopen

_HERE = os.path.dirname(os.path.abspath(__file__))
_SCRIPTS = os.path.dirname(_HERE)
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

from ambush import event_handler
from ambush import action_history_server
from ambush import live_database as db


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


if __name__ == "__main__":
    unittest.main()
