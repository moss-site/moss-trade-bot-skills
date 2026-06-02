"""
Ambush balanced-mode 5-rule decision logic.

Ported 1:1 from `internal/ambush/strategy/rule_decision.go` (BalancedDecideV0).
The skill-side live runner uses this to decide long / short / skip per event.

Keep this in lock-step with the Go version — any rule change to one must
ship in both. Contract tests on the 8 rule-coverage cases (one per side
of each rule boundary) live in `scripts/ambush/decision_test.py`.
"""

from __future__ import annotations

from typing import NamedTuple


# Side values matching Go `internal/ambush/strategy.SideLong/SideShort/SideSkip`.
SIDE_LONG = "long"
SIDE_SHORT = "short"
SIDE_SKIP = "skip"

# Direction values matching Go `internal/ambush/strategy.Direction*`.
DIRECTION_LONG = "long"
DIRECTION_SHORT = "short"
DIRECTION_BALANCED = "balanced"

# Reason labels matching Go `internal/ambush/strategy.Reason*`.
REASON_USER_CONFIG = "user_config"
REASON_INVALID_DIRECTION = "invalid_direction"
REASON_RULE_NO_MATCH = "rule_no_match"
REASON_DIRECTION_MISMATCH = "direction_mismatch"
REASON_SHORT_COMPOUND_OVERSTRETCH = "rule_short_compound_overstretch"
REASON_SHORT_EXTREME_PULLBACK = "rule_short_extreme_pullback"
REASON_SHORT_SPIKE_EXTREME = "rule_short_spike_extreme"
REASON_LONG_MOMENTUM_INIT = "rule_long_momentum_init"
REASON_LONG_MOMENTUM_EXTEND = "rule_long_momentum_extend"


class DecisionInput(NamedTuple):
    """Minimum feature set needed for decisions.

    surge_15m:     触发 K 线的 (close-open)/open as decimal (0.10 = 10%)
    rsi_14:        触发瞬间 RSI(14)
    chg_before_24h: 触发前 24h 价格变化 as percent (28.5 = +28.5%)
    direction:     'long' / 'short' / 'balanced'
    """

    surge_15m: float
    rsi_14: float
    chg_before_24h: float
    direction: str


class Decision(NamedTuple):
    side: str  # 'long' / 'short' / 'skip'
    reason: str


def decide(inp: DecisionInput) -> Decision:
    """Signal-first decision with direction used as an allowlist."""
    if inp.direction == DIRECTION_BALANCED:
        return balanced_decide_v0(inp)
    if inp.direction in (DIRECTION_SHORT, DIRECTION_LONG):
        signal = balanced_decide_v0(inp)
        if signal.side in (SIDE_SKIP, inp.direction):
            return signal
        return Decision(SIDE_SKIP, REASON_DIRECTION_MISMATCH)
    return Decision(SIDE_SKIP, REASON_INVALID_DIRECTION)


def balanced_decide_v0(inp: DecisionInput) -> Decision:
    """5-rule balanced decider. Ported from Go `strategy.BalancedDecideV0`.

    Matches in order, first hit returns.

    SHORT signals:
      rule_short_compound_overstretch: surge > 20% AND chg > 20% (compound climb)
      rule_short_extreme_pullback:     chg > 100% AND rsi > 70 (extreme top)
      rule_short_spike_extreme:        surge > 25% (single-K spike)

    LONG signals:
      rule_long_momentum_init:   10% < surge < 15% AND rsi < 60 AND chg < 10%
      rule_long_momentum_extend: 30 ≤ chg ≤ 80 AND rsi > 75
    """
    s = inp.surge_15m
    r = inp.rsi_14
    c = inp.chg_before_24h

    # SHORT signals
    if s > 0.20 and c > 20:
        return Decision(SIDE_SHORT, REASON_SHORT_COMPOUND_OVERSTRETCH)
    if c > 100 and r > 70:
        return Decision(SIDE_SHORT, REASON_SHORT_EXTREME_PULLBACK)
    if s > 0.25:
        return Decision(SIDE_SHORT, REASON_SHORT_SPIKE_EXTREME)

    # LONG signals
    if s > 0.10 and s < 0.15 and r < 60 and c < 10:
        return Decision(SIDE_LONG, REASON_LONG_MOMENTUM_INIT)
    if c >= 30 and c <= 80 and r > 75:
        return Decision(SIDE_LONG, REASON_LONG_MOMENTUM_EXTEND)

    return Decision(SIDE_SKIP, REASON_RULE_NO_MATCH)


def decision_from_event(event: dict, direction: str) -> Decision:
    """Build a Decision from a raw WS / bootstrap event payload.

    Reads surge_15m / rsi_14 / change_before_24h_pct from the envelope
    (decimal-string fields parsed to float).
    """
    return decide(
        DecisionInput(
            surge_15m=_as_float(event.get("surge_15m", 0)),
            # rsi_14 default must match backtest.py's `event.get("rsi_14") or 50`
            # — a missing/empty rsi defaults to the NEUTRAL 50, not 0 (which is
            # extreme oversold). Keeping these in lock-step ensures the same
            # event decides identically in live vs backtest.
            rsi_14=_as_float(event.get("rsi_14") or 50),
            chg_before_24h=_as_float(event.get("change_before_24h_pct", 0)),
            direction=direction,
        )
    )


def _as_float(v) -> float:
    """Coerce decimal-string / number / None to float. Returns 0.0 on parse fail."""
    if v is None:
        return 0.0
    if isinstance(v, (int, float)):
        return float(v)
    try:
        return float(str(v))
    except (ValueError, TypeError):
        return 0.0
