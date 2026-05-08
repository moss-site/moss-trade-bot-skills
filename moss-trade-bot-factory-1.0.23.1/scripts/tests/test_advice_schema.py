"""Unit tests for advisor-mode advice JSON schema.

Run from repo root:
    python3 -m unittest discover -s skill/production/scripts/tests
"""

from __future__ import annotations

import json
import os
import sys
import unittest
from datetime import datetime, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPTS = os.path.dirname(HERE)
if SCRIPTS not in sys.path:
    sys.path.insert(0, SCRIPTS)

from core.decision import DecisionParams
from advise import (
    ADVICE_SCHEMA_VERSION,
    LIVE_TRADE_PATH,
    _build_advice_payload,
    _build_dispatch_command,
)


def _cmd_arg(cmd, flag: str) -> str:
    idx = cmd.index(flag)
    return cmd[idx + 1]


def _build(action: str, **overrides):
    base = dict(
        cycle=42,
        interval_min=15,
        symbol="BTCUSDT",
        timeframe="15m",
        data_source="hyperliquid",
        action=action,
        direction=None,
        exit_reason=None,
        mark_price=65000.0,
        change_24h_pct=0.0123,
        regime="BULL",
        signal_value=0,
        free_margin=9876.54,
        wallet_balance=10500.0,
        position=None,
        params=DecisionParams(),
        suggestion=None,
        creds_path="/tmp/fake_creds.json",
        platform_url="",
    )
    base.update(overrides)
    return _build_advice_payload(**base)


class AdviceTopLevelTest(unittest.TestCase):
    REQUIRED_KEYS = {
        "version", "cycle", "issued_at", "valid_until",
        "symbol", "timeframe", "data_source",
        "action", "direction", "exit_reason",
        "context", "params_snapshot", "suggestion", "reasoning_draft", "dispatch_command",
    }

    def test_open_advice_has_all_required_keys(self):
        p = _build(
            "open",
            direction="LONG",
            signal_value=1,
            suggestion={"leverage": 10, "notional_usdt": "1000.00",
                        "client_order_id_prefix": "advisor-42-x"},
        )
        self.assertEqual(set(p.keys()), self.REQUIRED_KEYS)
        self.assertEqual(p["version"], ADVICE_SCHEMA_VERSION)
        self.assertEqual(p["action"], "open")
        self.assertEqual(p["direction"], "LONG")
        self.assertIsNone(p["exit_reason"])
        self.assertIsNotNone(p["reasoning_draft"])
        self.assertGreaterEqual(len(p["reasoning_draft"]["zh"]), 120)
        self.assertGreaterEqual(len(p["reasoning_draft"]["en"]), 120)
        self.assertIn("signal_value=1", p["reasoning_draft"]["zh"])

    def test_valid_until_equals_issued_at_plus_interval(self):
        p = _build("wait")
        issued = datetime.strptime(p["issued_at"], "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
        until = datetime.strptime(p["valid_until"], "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
        delta_sec = (until - issued).total_seconds()
        self.assertEqual(delta_sec, 15 * 60)

    def test_iso_format_is_rfc3339_z(self):
        p = _build("wait")
        for key in ("issued_at", "valid_until"):
            self.assertTrue(p[key].endswith("Z"), f"{key} must end with Z")
            datetime.strptime(p[key], "%Y-%m-%dT%H:%M:%SZ")  # parse or raise


class AdviceContextTest(unittest.TestCase):
    def test_context_serializes_position(self):
        position = {"side": "LONG", "qty": "0.0153", "entry_price": "62000.00",
                    "leverage": 10, "unrealized_pnl": "61.20"}
        p = _build("close", exit_reason="take_profit", position=position)
        ctx_pos = p["context"]["position"]
        self.assertIsNotNone(ctx_pos)
        self.assertEqual(ctx_pos["side"], "LONG")
        self.assertEqual(ctx_pos["qty"], "0.0153")
        self.assertAlmostEqual(ctx_pos["entry_price"], 62000.00)
        self.assertAlmostEqual(ctx_pos["leverage"], 10.0)

    def test_context_position_is_null_when_flat(self):
        p = _build("wait", position=None)
        self.assertIsNone(p["context"]["position"])


class AdviceActionBranchesTest(unittest.TestCase):
    def test_open_includes_dispatch_with_open_long(self):
        p = _build(
            "open",
            direction="LONG",
            signal_value=1,
            suggestion={"leverage": 10, "notional_usdt": "1000.00",
                        "client_order_id_prefix": "x"},
            platform_url="https://ai.moss.site",
        )
        cmd = p["dispatch_command"]
        self.assertIsNotNone(cmd)
        self.assertEqual(cmd[0], "python3")
        self.assertEqual(cmd[1], LIVE_TRADE_PATH)
        self.assertEqual(cmd[2], "open-long")
        self.assertIn("--reasoning-zh", cmd)
        self.assertIn("--reasoning-en", cmd)
        reasoning_zh = _cmd_arg(cmd, "--reasoning-zh")
        reasoning_en = _cmd_arg(cmd, "--reasoning-en")
        self.assertNotIn("<TODO", reasoning_zh)
        self.assertNotIn("<TODO", reasoning_en)
        self.assertGreaterEqual(len(reasoning_zh), 120)
        self.assertGreaterEqual(len(reasoning_en), 120)
        self.assertIn("--platform-url", cmd)
        self.assertIn("https://ai.moss.site", cmd)

    def test_open_short_subcommand(self):
        p = _build(
            "open",
            direction="SHORT",
            signal_value=-1,
            suggestion={"leverage": 5, "notional_usdt": "500.00",
                        "client_order_id_prefix": "x"},
        )
        self.assertEqual(p["dispatch_command"][2], "open-short")

    def test_close_dispatch_carries_position_side(self):
        position = {"side": "SHORT", "qty": "0.0234", "entry_price": "64000.00",
                    "leverage": 15, "unrealized_pnl": "-35.0"}
        p = _build("close", exit_reason="stop_loss", position=position)
        cmd = p["dispatch_command"]
        self.assertIsNotNone(cmd)
        self.assertEqual(cmd[2], "close")
        side_idx = cmd.index("--side")
        self.assertEqual(cmd[side_idx + 1], "SHORT")
        self.assertEqual(p["exit_reason"], "stop_loss")
        self.assertIsNotNone(p["reasoning_draft"])
        self.assertIn("触发止损", p["reasoning_draft"]["zh"])

    def test_hold_has_no_dispatch_or_suggestion(self):
        position = {"side": "LONG", "qty": "0.01", "entry_price": "60000.00",
                    "leverage": 10, "unrealized_pnl": "0"}
        p = _build("hold", position=position)
        self.assertIsNone(p["dispatch_command"])
        self.assertIsNone(p["suggestion"])
        self.assertIsNone(p["reasoning_draft"])

    def test_wait_has_no_dispatch_or_suggestion_or_position(self):
        p = _build("wait")
        self.assertIsNone(p["dispatch_command"])
        self.assertIsNone(p["suggestion"])
        self.assertIsNone(p["context"]["position"])
        self.assertIsNone(p["reasoning_draft"])


class AdviceJsonSerializableTest(unittest.TestCase):
    def test_roundtrip_through_json(self):
        p = _build(
            "open", direction="LONG", signal_value=1,
            suggestion={"leverage": 10, "notional_usdt": "1000.00",
                        "client_order_id_prefix": "x"},
        )
        encoded = json.dumps(p, ensure_ascii=False)
        decoded = json.loads(encoded)
        self.assertEqual(decoded["action"], "open")
        self.assertEqual(decoded["direction"], "LONG")


class DispatchCommandHelperTest(unittest.TestCase):
    def test_unknown_action_returns_none(self):
        self.assertIsNone(
            _build_dispatch_command(
                action="hold", direction=None,
                creds_path="/tmp/x.json", symbol="BTCUSDT",
            )
        )


if __name__ == "__main__":
    unittest.main()
