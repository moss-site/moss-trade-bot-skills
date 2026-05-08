"""Unit tests for the reasoning gate: required + bilingual + length.

Run from repo root:
    python3 -m unittest discover -s skill/production/scripts/tests
"""

from __future__ import annotations

import argparse
import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPTS = os.path.dirname(HERE)
if SCRIPTS not in sys.path:
    sys.path.insert(0, SCRIPTS)

from trading_client import TradingClient
from live_trade import resolve_reasoning


VALID_ZH = (
    "本轮按15m周期评估BTC/USDT，标记价跌破关键区间后24小时下跌2.60%，"
    "regime 转弱且 signal_value=-1，方向选择开空。仓位按 risk_per_trade 和 "
    "max_position_pct 限制名义敞口，使用计划杠杆但不追单放大风险；若下一周期"
    "信号反转、价格重新站回压力位或触及止损/止盈，优先减仓或平仓。"
)
VALID_EN = (
    "This 15m review for BTC/USDT sees price break below the key area with a 24h "
    "decline of 2.60%, weaker regime, and signal_value=-1, so the plan is to open "
    "a short position. Sizing follows risk_per_trade and max_position_pct instead "
    "of chasing extra leverage; if the next cycle reverses, reclaims resistance, "
    "or reaches stop/take-profit pressure, exposure should be reduced or closed."
)
THIN_ZH = "价格跌破79480，短期趋势走弱。24小时内下跌2.6%，阻力位下移。顺势开空。"
THIN_EN = "Opening long while controlling risk."


class RequireBilingualReasoningTest(unittest.TestCase):
    """trading_client._require_bilingual_reasoning is the order-gate enforcer."""

    def test_pass_with_zh_and_en(self) -> None:
        zh, en = TradingClient._require_bilingual_reasoning(
            VALID_ZH,
            VALID_EN,
        )
        self.assertIn("开空", zh)
        self.assertNotIn("开空", en)

    def test_strips_surrounding_whitespace(self) -> None:
        zh, en = TradingClient._require_bilingual_reasoning(
            f"  {VALID_ZH}  ", f"  {VALID_EN}  "
        )
        self.assertEqual(zh, VALID_ZH)
        self.assertEqual(en, VALID_EN)

    def test_missing_zh_raises(self) -> None:
        with self.assertRaises(ValueError):
            TradingClient._require_bilingual_reasoning("", VALID_EN)

    def test_missing_en_raises(self) -> None:
        with self.assertRaises(ValueError):
            TradingClient._require_bilingual_reasoning(VALID_ZH, "")

    def test_zh_without_han_raises(self) -> None:
        with self.assertRaises(ValueError):
            TradingClient._require_bilingual_reasoning("Open long", VALID_EN)

    def test_en_with_han_raises(self) -> None:
        with self.assertRaises(ValueError):
            TradingClient._require_bilingual_reasoning(
                VALID_ZH, f"{VALID_EN} 多"
            )

    def test_thin_zh_raises(self) -> None:
        with self.assertRaises(ValueError):
            TradingClient._require_bilingual_reasoning(THIN_ZH, VALID_EN)

    def test_thin_en_raises(self) -> None:
        with self.assertRaises(ValueError):
            TradingClient._require_bilingual_reasoning(VALID_ZH, THIN_EN)

    def test_zh_too_long_raises(self) -> None:
        zh = "按多头信号开多。" * 200
        with self.assertRaises(ValueError):
            TradingClient._require_bilingual_reasoning(zh, VALID_EN)

    def test_en_too_long_raises(self) -> None:
        en = "Opening long position. " * 100
        with self.assertRaises(ValueError):
            TradingClient._require_bilingual_reasoning(VALID_ZH, en)


class ResolveReasoningTest(unittest.TestCase):
    """live_trade.resolve_reasoning rejects missing or malformed bilingual input."""

    @staticmethod
    def _ns(zh: str = "", en: str = "") -> argparse.Namespace:
        return argparse.Namespace(reasoning_zh=zh, reasoning_en=en)

    def test_pass_returns_tuple(self) -> None:
        zh, en = resolve_reasoning(
            self._ns(
                zh=VALID_ZH,
                en=VALID_EN,
            )
        )
        self.assertEqual(zh, VALID_ZH)
        self.assertEqual(en, VALID_EN)

    def test_missing_zh_exits_2(self) -> None:
        with self.assertRaises(SystemExit) as ctx:
            resolve_reasoning(self._ns(zh="", en=VALID_EN))
        self.assertEqual(ctx.exception.code, 2)

    def test_missing_en_exits_2(self) -> None:
        with self.assertRaises(SystemExit) as ctx:
            resolve_reasoning(self._ns(zh=VALID_ZH, en=""))
        self.assertEqual(ctx.exception.code, 2)

    def test_zh_without_han_exits_2(self) -> None:
        with self.assertRaises(SystemExit) as ctx:
            resolve_reasoning(self._ns(zh="open long", en=VALID_EN))
        self.assertEqual(ctx.exception.code, 2)

    def test_short_reasoning_exits_2(self) -> None:
        with self.assertRaises(SystemExit) as ctx:
            resolve_reasoning(self._ns(zh=THIN_ZH, en=VALID_EN))
        self.assertEqual(ctx.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
