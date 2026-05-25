"""Unit tests for the bilingual reasoning gate.

Pins the rules enforced before every order: bilingual presence, length ceiling,
Chinese-in-zh / no-Chinese-in-en. Tightening any of these without updating tests
should fail loudly.

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

from text_i18n import contains_han, default_text, validate_bilingual_text
from live_trade import resolve_reasoning


VALID_ZH = (
    "本轮按15m周期评估BTC，标记价跌破关键区间后24小时下跌2.60%，"
    "regime 转弱且 signal_value=-1，方向选择开空。仓位按 risk_per_trade 和 "
    "max_position_pct 限制名义敞口，使用计划杠杆但不追单放大风险。"
)
VALID_EN = (
    "This 15m review for BTC sees price break below the key area with a 24h "
    "decline of 2.60%, weaker regime, and signal_value=-1, so the plan is to open "
    "a short position. Sizing follows risk_per_trade and max_position_pct."
)


class ContainsHanTest(unittest.TestCase):
    def test_zh_phrase(self):
        self.assertTrue(contains_han("开空"))

    def test_en_phrase(self):
        self.assertFalse(contains_han("open short"))

    def test_mixed_phrase(self):
        self.assertTrue(contains_han("Open short 开空"))

    def test_empty(self):
        self.assertFalse(contains_han(""))

    def test_none(self):
        self.assertFalse(contains_han(None))


class DefaultTextTest(unittest.TestCase):
    def test_prefers_zh(self):
        self.assertEqual(default_text({"zh": "中文", "en": "english"}), "中文")

    def test_falls_back_to_en_when_zh_empty(self):
        self.assertEqual(default_text({"zh": "", "en": "english"}), "english")

    def test_strips_surrounding_whitespace(self):
        self.assertEqual(default_text({"zh": "  中文  ", "en": ""}), "中文")

    def test_handles_none_input(self):
        self.assertEqual(default_text(None), "")


class ValidateBilingualTextTest(unittest.TestCase):
    def test_valid_pair_passes(self):
        out = validate_bilingual_text("reasoning", {"zh": VALID_ZH, "en": VALID_EN}, 512)
        self.assertEqual(out, {"zh": VALID_ZH, "en": VALID_EN})

    def test_strips_whitespace(self):
        out = validate_bilingual_text(
            "reasoning", {"zh": f"  {VALID_ZH}  ", "en": f"\t{VALID_EN}\n"}, 512
        )
        self.assertEqual(out["zh"], VALID_ZH)
        self.assertEqual(out["en"], VALID_EN)

    def test_missing_zh_raises(self):
        with self.assertRaisesRegex(ValueError, "zh and"):
            validate_bilingual_text("reasoning", {"zh": "", "en": VALID_EN}, 512)

    def test_missing_en_raises(self):
        with self.assertRaisesRegex(ValueError, "zh and"):
            validate_bilingual_text("reasoning", {"zh": VALID_ZH, "en": ""}, 512)

    def test_both_missing_raises(self):
        with self.assertRaisesRegex(ValueError, "required"):
            validate_bilingual_text("reasoning", {"zh": "", "en": ""}, 512)

    def test_zh_too_long_raises(self):
        zh = "按多头信号开多。" * 100
        with self.assertRaisesRegex(ValueError, "too long"):
            validate_bilingual_text("reasoning", {"zh": zh, "en": VALID_EN}, 512)

    def test_en_too_long_raises(self):
        en = "Opening long position. " * 100
        with self.assertRaisesRegex(ValueError, "too long"):
            validate_bilingual_text("reasoning", {"zh": VALID_ZH, "en": en}, 512)

    def test_zh_without_han_raises(self):
        with self.assertRaisesRegex(ValueError, "Chinese text"):
            validate_bilingual_text(
                "reasoning", {"zh": "Open long", "en": VALID_EN}, 512
            )

    def test_en_with_han_raises(self):
        with self.assertRaisesRegex(ValueError, "without Chinese"):
            validate_bilingual_text(
                "reasoning", {"zh": VALID_ZH, "en": f"{VALID_EN} 多"}, 512
            )

    def test_none_value_handled_as_missing(self):
        with self.assertRaisesRegex(ValueError, "required"):
            validate_bilingual_text("reasoning", None, 512)


class ResolveReasoningTest(unittest.TestCase):
    """live_trade.resolve_reasoning wraps validate_bilingual_text on the CLI side."""

    @staticmethod
    def _ns(zh="", en="", reasoning=""):
        return argparse.Namespace(reasoning_zh=zh, reasoning_en=en, reasoning=reasoning)

    def test_both_empty_returns_legacy_reasoning(self):
        zh, en = resolve_reasoning(self._ns(reasoning="legacy fallback"))
        self.assertEqual(zh, "legacy fallback")
        self.assertEqual(en, "")

    def test_both_empty_no_legacy_returns_empty(self):
        zh, en = resolve_reasoning(self._ns())
        self.assertEqual((zh, en), ("", ""))

    def test_valid_pair_returns_zh_and_en(self):
        zh, en = resolve_reasoning(self._ns(zh=VALID_ZH, en=VALID_EN))
        self.assertEqual(zh, VALID_ZH)
        self.assertEqual(en, VALID_EN)

    def test_missing_zh_raises(self):
        with self.assertRaises(ValueError):
            resolve_reasoning(self._ns(zh="", en=VALID_EN))

    def test_missing_en_raises(self):
        with self.assertRaises(ValueError):
            resolve_reasoning(self._ns(zh=VALID_ZH, en=""))

    def test_zh_without_han_raises(self):
        with self.assertRaises(ValueError):
            resolve_reasoning(self._ns(zh="open long", en=VALID_EN))


if __name__ == "__main__":
    unittest.main()
