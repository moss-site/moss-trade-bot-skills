"""
Contract tests for `decision.py`.

The 8 rule-coverage cases mirror Go `internal/ambush/strategy/rule_decision_test.go`.
If either side changes a rule, both tests must move in lock-step or
production routes will diverge.

(Note: these unit cases are NOT the same thing as the "8 sanity events"
that used to live in data_cache/ambush/sanity_events.json — that
integration-level concept was removed 2026-05-17 as redundant with this
unit test + the 216-event backtest. See the 2026-05-17 design spec.)

Run:
    python3 -m unittest scripts.ambush.decision_test
or:
    cd scripts && python3 -m unittest ambush.decision_test
"""

from __future__ import annotations

import os
import sys
import unittest

_HERE = os.path.dirname(os.path.abspath(__file__))
_SCRIPTS = os.path.dirname(_HERE)
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

from ambush import decision as d


class TestBalancedDecide(unittest.TestCase):
    """Each case is (label, surge_15m, rsi, chg_24h_prior, expected_side, expected_reason)."""

    cases = [
        # SHORT rules
        (
            "compound_overstretch", 0.22, 50.0, 25.0,
            d.SIDE_SHORT, d.REASON_SHORT_COMPOUND_OVERSTRETCH,
        ),
        (
            "extreme_pullback", 0.05, 72.0, 120.0,
            d.SIDE_SHORT, d.REASON_SHORT_EXTREME_PULLBACK,
        ),
        (
            "spike_extreme", 0.30, 50.0, 5.0,
            d.SIDE_SHORT, d.REASON_SHORT_SPIKE_EXTREME,
        ),

        # LONG rules
        (
            "momentum_init", 0.12, 55.0, 8.0,
            d.SIDE_LONG, d.REASON_LONG_MOMENTUM_INIT,
        ),
        (
            "momentum_extend", 0.05, 78.0, 50.0,
            d.SIDE_LONG, d.REASON_LONG_MOMENTUM_EXTEND,
        ),

        # SKIP cases
        (
            "skip_low_signal", 0.05, 50.0, 5.0,
            d.SIDE_SKIP, d.REASON_RULE_NO_MATCH,
        ),
        (
            "skip_long_init_rsi_too_high", 0.12, 65.0, 5.0,
            d.SIDE_SKIP, d.REASON_RULE_NO_MATCH,
        ),
        (
            "skip_chg_in_dead_zone", 0.05, 78.0, 25.0,
            d.SIDE_SKIP, d.REASON_RULE_NO_MATCH,
        ),
    ]

    def test_balanced_cases(self):
        for label, surge, rsi, chg, want_side, want_reason in self.cases:
            with self.subTest(case=label):
                inp = d.DecisionInput(
                    surge_15m=surge, rsi_14=rsi, chg_before_24h=chg,
                    direction=d.DIRECTION_BALANCED,
                )
                got = d.balanced_decide_v0(inp)
                self.assertEqual(
                    got.side, want_side,
                    f"{label}: side want={want_side} got={got.side} (reason={got.reason})",
                )
                self.assertEqual(
                    got.reason, want_reason,
                    f"{label}: reason want={want_reason} got={got.reason}",
                )


class TestDispatch(unittest.TestCase):
    def test_long_direction_allows_long_signal(self):
        inp = d.DecisionInput(0.12, 55.0, 8.0, d.DIRECTION_LONG)
        got = d.decide(inp)
        self.assertEqual(got.side, d.SIDE_LONG)
        self.assertEqual(got.reason, d.REASON_LONG_MOMENTUM_INIT)

    def test_long_direction_filters_short_signal(self):
        inp = d.DecisionInput(0.22, 50.0, 25.0, d.DIRECTION_LONG)
        got = d.decide(inp)
        self.assertEqual(got.side, d.SIDE_SKIP)
        self.assertEqual(got.reason, d.REASON_DIRECTION_MISMATCH)

    def test_short_direction_allows_short_signal(self):
        inp = d.DecisionInput(0.22, 50.0, 25.0, d.DIRECTION_SHORT)
        got = d.decide(inp)
        self.assertEqual(got.side, d.SIDE_SHORT)
        self.assertEqual(got.reason, d.REASON_SHORT_COMPOUND_OVERSTRETCH)

    def test_short_direction_filters_long_signal(self):
        inp = d.DecisionInput(0.12, 55.0, 8.0, d.DIRECTION_SHORT)
        got = d.decide(inp)
        self.assertEqual(got.side, d.SIDE_SKIP)
        self.assertEqual(got.reason, d.REASON_DIRECTION_MISMATCH)

    def test_invalid_direction(self):
        inp = d.DecisionInput(0.0, 0.0, 0.0, "rotational")
        got = d.decide(inp)
        self.assertEqual(got.side, d.SIDE_SKIP)
        self.assertEqual(got.reason, d.REASON_INVALID_DIRECTION)


class TestEnvelopeAdapter(unittest.TestCase):
    def test_decision_from_event_decimal_strings(self):
        """Server sends decimal-as-string for ratio fields; adapter must parse."""
        event = {
            "surge_15m": "0.22",          # decimal string
            "rsi_14": "50",
            "change_before_24h_pct": "25",  # percent, not ratio
        }
        got = d.decision_from_event(event, d.DIRECTION_BALANCED)
        self.assertEqual(got.side, d.SIDE_SHORT)
        self.assertEqual(got.reason, d.REASON_SHORT_COMPOUND_OVERSTRETCH)

    def test_missing_fields_default_zero(self):
        got = d.decision_from_event({}, d.DIRECTION_BALANCED)
        self.assertEqual(got.side, d.SIDE_SKIP)
        self.assertEqual(got.reason, d.REASON_RULE_NO_MATCH)


if __name__ == "__main__":
    unittest.main()
