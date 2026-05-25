"""Unit tests for run_evolve_backtest invariants pinned in evolution_guide.md.

Two big promises in the doc that the worker layer relies on:
  1. "性格参数永远不改" — `lock_personality` overwrites any LLM-proposed
     personality change with the initial value.
  2. "战术参数 (float) ±30% 漂移上限" — `clamp_tactical_drift` clamps each
     TACTICAL_FLOAT_FIELDS entry to [initial × 0.7, initial × 1.3]. Int and
     bool fields are NOT clamped (doc clarified in 2026-05-11 R1 followup).

These are user-facing contracts; if either silently weakens the agent's
strategies can wander outside the bot owner's risk budget without warning.

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

from run_evolve_backtest import (
    MAX_DRIFT_PCT,
    PERSONALITY_FIELDS,
    TACTICAL_FLOAT_FIELDS,
    clamp_tactical_drift,
    lock_personality,
)


class DriftLimitConstantTest(unittest.TestCase):
    def test_max_drift_pct_is_30_percent(self):
        # If this changes, evolution_guide.md needs to change in lockstep.
        self.assertEqual(MAX_DRIFT_PCT, 0.30)


class TacticalFloatFieldsListTest(unittest.TestCase):
    """The exact 9 fields documented in evolution_guide.md as float-clamped."""

    def test_contains_expected_fields(self):
        expected = {
            "entry_threshold", "exit_threshold",
            "sl_atr_mult", "tp_rr_ratio",
            "trailing_activation_pct", "trailing_distance_atr",
            "regime_sensitivity",
            "supertrend_mult", "trend_strength_min",
        }
        self.assertEqual(set(TACTICAL_FLOAT_FIELDS), expected)


class PersonalityFieldsListTest(unittest.TestCase):
    """Personality fields are the locked set: signals + risk shape + rolling."""

    def test_contains_core_personality_fields(self):
        for field in (
            "long_bias",
            "base_leverage", "max_leverage",
            "risk_per_trade", "max_position_pct",
            "rolling_enabled",
            "trend_weight", "momentum_weight", "mean_revert_weight",
        ):
            self.assertIn(field, PERSONALITY_FIELDS)


class ClampTacticalDriftTest(unittest.TestCase):
    INITIAL = {
        "entry_threshold": 0.20,
        "exit_threshold": 0.10,
        "sl_atr_mult": 2.0,
        "tp_rr_ratio": 2.5,
        "trailing_activation_pct": 0.05,
        "trailing_distance_atr": 1.5,
        "regime_sensitivity": 1.0,
        "supertrend_mult": 3.0,
        "trend_strength_min": 25.0,
        # int / bool — must NOT be clamped
        "rsi_period": 14,
        "fast_ma_period": 9,
        "slow_ma_period": 21,
        "exit_on_regime_change": True,
    }

    def test_in_range_value_unchanged(self):
        out = clamp_tactical_drift({"entry_threshold": 0.22}, self.INITIAL)
        self.assertAlmostEqual(out["entry_threshold"], 0.22, places=9)

    def test_above_upper_band_clamped(self):
        out = clamp_tactical_drift({"entry_threshold": 0.50}, self.INITIAL)
        # initial 0.20 * 1.30 = 0.26
        self.assertAlmostEqual(out["entry_threshold"], 0.26, places=9)

    def test_below_lower_band_clamped(self):
        out = clamp_tactical_drift({"entry_threshold": 0.01}, self.INITIAL)
        # initial 0.20 * 0.70 = 0.14
        self.assertAlmostEqual(out["entry_threshold"], 0.14, places=9)

    def test_sl_atr_mult_band(self):
        # initial 2.0 → band [1.4, 2.6]
        self.assertAlmostEqual(
            clamp_tactical_drift({"sl_atr_mult": 5.0}, self.INITIAL)["sl_atr_mult"],
            2.6,
            places=9,
        )
        self.assertAlmostEqual(
            clamp_tactical_drift({"sl_atr_mult": 0.5}, self.INITIAL)["sl_atr_mult"],
            1.4,
            places=9,
        )

    def test_int_field_not_clamped(self):
        """rsi_period is int, must pass through unchanged."""
        out = clamp_tactical_drift({"rsi_period": 25}, self.INITIAL)
        self.assertEqual(out["rsi_period"], 25)

    def test_bool_field_not_clamped(self):
        out = clamp_tactical_drift({"exit_on_regime_change": False}, self.INITIAL)
        self.assertEqual(out["exit_on_regime_change"], False)

    def test_missing_initial_is_passthrough(self):
        """If field not in initial_params, leave current value alone."""
        out = clamp_tactical_drift({"entry_threshold": 0.05}, {"sl_atr_mult": 2.0})
        self.assertEqual(out["entry_threshold"], 0.05)

    def test_zero_initial_skips_clamp(self):
        """init_val == 0 makes lo=hi=0; the code skips this case to avoid lock at 0."""
        out = clamp_tactical_drift(
            {"entry_threshold": 0.30},
            {"entry_threshold": 0.0},
        )
        self.assertAlmostEqual(out["entry_threshold"], 0.30, places=9)

    def test_other_fields_preserved(self):
        current = {"entry_threshold": 0.99, "rsi_period": 14, "exit_on_regime_change": True}
        out = clamp_tactical_drift(current, self.INITIAL)
        # Clamped
        self.assertAlmostEqual(out["entry_threshold"], 0.26, places=9)
        # Not clamped
        self.assertEqual(out["rsi_period"], 14)
        self.assertEqual(out["exit_on_regime_change"], True)


class LockPersonalityTest(unittest.TestCase):
    def test_overwrites_modified_personality_field(self):
        initial = {"long_bias": 0.5, "base_leverage": 3.0, "max_leverage": 5.0}
        modified = {"long_bias": 0.9, "base_leverage": 10.0, "max_leverage": 40.0}
        out = lock_personality(modified, initial)
        self.assertEqual(out["long_bias"], 0.5)
        self.assertEqual(out["base_leverage"], 3.0)
        self.assertEqual(out["max_leverage"], 5.0)

    def test_passes_tactical_fields_through(self):
        initial = {"long_bias": 0.5}
        modified = {"long_bias": 0.9, "entry_threshold": 0.22, "exit_threshold": 0.15}
        out = lock_personality(modified, initial)
        self.assertEqual(out["long_bias"], 0.5)
        self.assertEqual(out["entry_threshold"], 0.22)
        self.assertEqual(out["exit_threshold"], 0.15)

    def test_missing_initial_field_passthrough(self):
        out = lock_personality({"long_bias": 0.9}, {"base_leverage": 3.0})
        # long_bias not in initial → preserved
        self.assertEqual(out["long_bias"], 0.9)


if __name__ == "__main__":
    unittest.main()
