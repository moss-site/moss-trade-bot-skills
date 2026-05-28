"""Tests for ChainedHarness — cross-validate against Go harness expected behavior.

The Go side has twin tests in internal/ambush/backtest/harness_test.go:
  TestRunSinglePositionLockSkip — parallel to test_chained_single_position_lock
  TestRunInsufficientCapitalSkip — parallel to test_chained_zero_capital

When outputs diverge from Go, one of the two harnesses has drifted.
"""

from __future__ import annotations

import datetime
import decimal
from decimal import Decimal
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

import sys
# conftest.py adds scripts/ to sys.path — but also guard here for direct runs.
_scripts_dir = Path(__file__).resolve().parent.parent / "scripts"
if str(_scripts_dir) not in sys.path:
    sys.path.insert(0, str(_scripts_dir))

from ambush.chained_harness import (
    AmbushBotParams,
    ChainedHarness,
    Dataset,
    EventRow,
    load_dataset,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_DATA_DIR = Path(__file__).resolve().parent.parent / "data_cache" / "ambush"


def _short_only_params() -> AmbushBotParams:
    """Minimal 'short' direction params used in tests that need predictable
    decisions — direction='short' forces all long-signal events to skip."""
    return AmbushBotParams.from_dict({
        "direction": "short",
        "long_params": {
            "leverage": 2, "position_pct": "0.20", "stop_loss_pct": "0.08",
            "trailing_pct": "0.30", "max_hold_hours": 6, "momentum_bars": 0,
            "cooldown_bars": 1, "sl_atr_mult": "2.0",
        },
        "short_params": {
            "leverage": 2, "position_pct": "0.20", "stop_loss_pct": "0.40",
            "trailing_pct": "0.25", "max_hold_hours": 6, "cooldown_bars": 1,
            "entry_delay_bars": 0, "sl_atr_mult": "2.5",
        },
        "rhythm": {"max_trades_per_event": 1, "same_coin_dedup_days": 7},
    })


def _balanced_params() -> AmbushBotParams:
    return AmbushBotParams.from_dict({
        "direction": "balanced",
        "long_params": {
            "leverage": 3, "position_pct": "0.20", "stop_loss_pct": "0.08",
            "trailing_pct": "0.30", "max_hold_hours": 30, "momentum_bars": 0,
            "cooldown_bars": 1, "sl_atr_mult": "2.0",
        },
        "short_params": {
            "leverage": 3, "position_pct": "0.20", "stop_loss_pct": "0.40",
            "trailing_pct": "0.25", "max_hold_hours": 168, "cooldown_bars": 15,
            "entry_delay_bars": 16, "sl_atr_mult": "2.5",
        },
        "rhythm": {"max_trades_per_event": 1, "same_coin_dedup_days": 7},
    })


def _make_short_trigger_event(
    base: str, symbol: str, trigger_ts: datetime.datetime, price: float = 1.0
) -> EventRow:
    """Create an event that always triggers a short decision (surge=0.30 > 0.25)."""
    return EventRow(
        symbol=symbol,
        base=base,
        trigger_ts=trigger_ts,
        trigger_date=trigger_ts.strftime("%Y-%m-%d"),
        price_at_trigger=price,
        surge_15m=0.30,   # rule_short_spike_extreme (> 0.25)
        rsi_14=50.0,
        chg_before_24h=15.0,
        oi_mc=0.5,
    )


def _make_klines_df(base_ts: datetime.datetime, n: int, price: float = 1.0) -> pd.DataFrame:
    """Build a minimal klines DataFrame of n 15m bars starting at base_ts."""
    rows = []
    for i in range(n):
        ts = base_ts + datetime.timedelta(minutes=15 * (i + 1))
        rows.append({
            "timestamp": ts,
            "ts": pd.Timestamp(ts),
            "open": price,
            "high": price * 1.01,
            "low": price * 0.99,
            "close": price * (1 - 0.01 * (i + 1)),  # slowly falling (helps short)
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Test 1: single_position_lock skip
# ---------------------------------------------------------------------------

def test_chained_single_position_lock():
    """Two events on different coins within the same 15m window:
    the second must be recorded as skip/single_position_lock.

    Mirrors Go TestRunSinglePositionLockSkip.
    """
    ts1 = datetime.datetime(2025, 6, 1, 10, 0, 0)
    ts2 = datetime.datetime(2025, 6, 1, 10, 15, 0)   # 15min later — within lock

    ev1 = _make_short_trigger_event("AAA", "AAA/USDT", ts1, price=1.0)
    ev2 = _make_short_trigger_event("BBB", "BBB/USDT", ts2, price=2.0)

    # Build klines for AAA: 40 bars so the trade has enough post-trigger bars
    # to fire max_hold at bar 24 (6h × 4 = 24 bars) and the lock extends ~6h.
    aaa_klines = _make_klines_df(ts1, 40, price=1.0)
    # BBB klines not needed — it will be locked before we check klines.

    dataset = Dataset(
        events=[ev1, ev2],
        klines_cache={"AAA": aaa_klines, "BBB": None},
    )

    params = _short_only_params()
    result = ChainedHarness(params).run(dataset, Decimal("10000"))

    assert result.events_total == 2, f"events_total={result.events_total} want 2"
    # At least one single_position_lock skip (ev2 might be locked if ev1 trades).
    # ev1 triggers a short (surge=0.30 > 0.25); trade closes within 6h max_hold.
    # ev2 arrives 15m after ev1 trigger (not after the close), so it must be locked.
    lock_count = result.skips_by_reason.get("single_position_lock", 0)
    assert lock_count >= 1, (
        f"expected single_position_lock skip, got skips_by_reason={result.skips_by_reason}"
    )

    # The skip trade row must carry AAA/USDT as the locked symbol.
    lock_trades = [t for t in result.trades if t.reason == "single_position_lock"]
    assert lock_trades, "no single_position_lock Trade row found"
    assert lock_trades[0].symbol == "AAA/USDT", (
        f"locked symbol={lock_trades[0].symbol!r} want AAA/USDT"
    )


# ---------------------------------------------------------------------------
# Test 2: zero initial capital — all skips
# ---------------------------------------------------------------------------

def test_chained_zero_capital():
    """With $0 initial capital every trade is notional_zero or
    insufficient_capital — no trades executed.

    Mirrors Go TestRunInsufficientCapitalSkip.
    """
    if not _DATA_DIR.exists():
        pytest.skip(f"data_cache not found at {_DATA_DIR}")

    dataset = load_dataset(_DATA_DIR)
    params = _short_only_params()
    result = ChainedHarness(params).run(dataset, Decimal("0"))

    assert result.events_traded == 0, (
        f"events_traded={result.events_traded} want 0 with $0 capital"
    )
    skips = result.skips_by_reason
    assert skips.get("insufficient_capital", 0) + skips.get("notional_zero", 0) > 0, (
        f"expected insufficient_capital or notional_zero skips, got {skips}"
    )


# ---------------------------------------------------------------------------
# Test 3: equity curve seed + final_wallet consistency
# ---------------------------------------------------------------------------

def test_chained_equity_curve_seed_and_final_wallet():
    """EquityCurve[0] must carry initial_capital, and final_wallet must equal
    the last point in the curve.

    Mirrors Go TestRunChainedWalletGrowsAfterWin assertions.
    """
    if not _DATA_DIR.exists():
        pytest.skip(f"data_cache not found at {_DATA_DIR}")

    dataset = load_dataset(_DATA_DIR)
    params = _balanced_params()
    initial = Decimal("10000")
    result = ChainedHarness(params).run(dataset, initial)

    if result.events_traded == 0:
        pytest.skip("no trades in this dataset with balanced params — test inconclusive")

    assert len(result.equity_curve) >= 2, (
        f"equity_curve too short: {len(result.equity_curve)}"
    )
    assert result.equity_curve[0].wallet == initial, (
        f"seed wallet={result.equity_curve[0].wallet} want {initial}"
    )
    assert result.final_wallet == result.equity_curve[-1].wallet, (
        f"final_wallet={result.final_wallet} vs last equity point={result.equity_curve[-1].wallet}"
    )
    assert result.initial_capital == initial


# ---------------------------------------------------------------------------
# Test 4: deterministic sort — same result on two runs
# ---------------------------------------------------------------------------

def test_chained_deterministic():
    """Running the harness twice on the same dataset must produce identical
    results (no randomness leaks through map iteration etc.)."""
    if not _DATA_DIR.exists():
        pytest.skip(f"data_cache not found at {_DATA_DIR}")

    dataset = load_dataset(_DATA_DIR)
    params = _balanced_params()
    initial = Decimal("10000")

    r1 = ChainedHarness(params).run(dataset, initial)
    r2 = ChainedHarness(params).run(dataset, initial)

    assert r1.events_total == r2.events_total
    assert r1.events_traded == r2.events_traded
    assert r1.final_wallet == r2.final_wallet
    assert r1.skips_by_reason == r2.skips_by_reason
    assert len(r1.equity_curve) == len(r2.equity_curve)
    for i, (p1, p2) in enumerate(zip(r1.equity_curve, r2.equity_curve)):
        assert p1.wallet == p2.wallet, f"equity_curve[{i}] mismatch"
