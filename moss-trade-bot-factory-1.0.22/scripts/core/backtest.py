"""Replay-aligned backtest engine for the skill.

Ported from share_for_local_run/agent_backtest.py to match Go verify replay:
- Two-layer: incremental advisor + source-core-style executor
- Fixed depth book execution (simulate_replay_baseline_fill)
- Hourly funding settlement at hour boundaries
- 4.5bps taker fee on every fill
- No recovery bars for fixed dataset (matches Go FixedDatasetKlineSource)
"""

import math
from decimal import Decimal, ROUND_HALF_UP, getcontext
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional

# Match Go shopspring/decimal precision
getcontext().prec = 28

def _D(v) -> Decimal:
    """Convert to Decimal. Handles float, int, str, Decimal."""
    if isinstance(v, Decimal):
        return v
    if isinstance(v, float):
        return Decimal(str(v))
    return Decimal(v)

from core.decision import DecisionParams, compute_signals
from core.indicators import atr as compute_atr
from core.engine import Trade, BacktestResult
from core.local_costs import local_taker_fee_rate
from core.replay_baseline import (
    FIXED_REPLAY_FUNDING_RATE,
    REPLAY_BASELINE_PROFILE,
    build_fixed_replay_depth_book,
    simulate_replay_baseline_fill,
)
from core.realtime_incremental import RealtimeIncrementalEvaluator, IncrementalBar


REPLAY_ALIGNED_REGIME_WINDOW = 48
REPLAY_ALIGNED_PROFIT_FACTOR_CAP = 999999
DEFAULT_LOOKBACK_BARS = 1  # matches Go fixedDatasetReplayWarmupBars


# ---------------------------------------------------------------------------
# Helpers (ported from agent_backtest.py)
# ---------------------------------------------------------------------------

def _timestamp_at(df: pd.DataFrame, idx: int) -> pd.Timestamp:
    if "timestamp" in df.columns:
        ts = pd.Timestamp(df["timestamp"].iloc[idx])
        return ts.tz_convert("UTC") if ts.tzinfo else ts.tz_localize("UTC")
    return pd.Timestamp(idx, unit="h", tz="UTC")


def _mark_price_before(df: pd.DataFrame, evaluation_at: pd.Timestamp) -> float:
    if "timestamp" not in df.columns or len(df) == 0:
        return float(df["close"].iloc[-1]) if len(df) else 0.0
    subset = df[df["timestamp"] <= evaluation_at]
    if subset.empty:
        return float(df["close"].iloc[-1]) if len(df) else 0.0
    return float(subset.iloc[-1]["close"])


def _search_left_timestamp(timestamps: pd.DatetimeIndex, ts: pd.Timestamp) -> int:
    target = pd.Timestamp(ts)
    target = target.tz_convert("UTC") if target.tzinfo else target.tz_localize("UTC")
    return int(timestamps.searchsorted(target, side="left"))


def _replay_reference_price(direction: int, mark_price: float) -> float:
    book = build_fixed_replay_depth_book(mark_price)
    if direction > 0 and book["asks"]:
        return float(book["asks"][0][0])
    if direction < 0 and book["bids"]:
        return float(book["bids"][0][0])
    return float(mark_price)


def _replay_notional_tolerance(target_notional: float) -> float:
    return max(1.0, abs(target_notional) * 0.02)


def _weighted_leverage(weighted: float, total: float) -> int:
    if total <= 0:
        return 1
    return max(1, min(int(round(weighted / total)), 40))


def _holding_hours_between(df: pd.DataFrame, entry_idx: int, exit_idx: int) -> float:
    if exit_idx <= entry_idx:
        return 0.0
    if "timestamp" in df.columns:
        entry_ts = _timestamp_at(df, entry_idx)
        exit_ts = _timestamp_at(df, exit_idx)
        return max(0.0, (exit_ts - entry_ts).total_seconds() / 3600.0)
    return float(exit_idx - entry_idx)


def _infer_step_duration(df: pd.DataFrame) -> pd.Timedelta:
    if "timestamp" in df.columns and len(df) >= 2:
        series = pd.to_datetime(df["timestamp"], utc=True)
        deltas = series.diff().dropna()
        if len(deltas) > 0:
            step = deltas.iloc[0]
            if step > pd.Timedelta(0):
                return step
    return pd.Timedelta(hours=1)


def _first_replay_evaluation_at(start: pd.Timestamp, step: pd.Timedelta) -> pd.Timestamp:
    start = pd.Timestamp(start)
    start = start.tz_convert("UTC") if start.tzinfo else start.tz_localize("UTC")
    first = start.floor(step)
    if first < start:
        first += step
    return first


# ---------------------------------------------------------------------------
# Desired position
# ---------------------------------------------------------------------------

@dataclass
class _DesiredPosition:
    signal: str
    side: Optional[str]
    target_notional: Decimal  # Decimal precision to match Go
    leverage: int


def _derive_desired_from_evaluator(
    evaluator: RealtimeIncrementalEvaluator,
    *,
    reference_initial_fund,  # float or Decimal
    current_equity,          # float or Decimal
) -> _DesiredPosition:
    """Derive desired position from a persistent incremental evaluator.

    Matches Go: deriveDesiredRealtimePosition(out, currentEquity, referenceInitialFund).
    """
    d_ref = _D(reference_initial_fund)
    d_eq = _D(current_equity)
    if d_ref <= 0 or d_eq <= 0:
        return _DesiredPosition(signal="flat", side=None, target_notional=Decimal(0), leverage=1)

    open_positions = evaluator.open_positions()
    if not open_positions:
        return _DesiredPosition(signal="flat", side=None, target_notional=Decimal(0), leverage=1)

    long_notional = 0.0
    short_notional = 0.0
    long_lev_weighted = 0.0
    short_lev_weighted = 0.0
    for pos in open_positions:
        notional = float(pos.notional)
        lev_weighted = notional * float(pos.leverage)
        if int(pos.direction) < 0:
            short_notional += notional
            short_lev_weighted += lev_weighted
        else:
            long_notional += notional
            long_lev_weighted += lev_weighted

    if short_notional > long_notional:
        d_target = d_eq * _D(short_notional) / d_ref
        return _DesiredPosition(
            signal="short", side="short",
            target_notional=d_target.quantize(_D("0.00000001")),
            leverage=_weighted_leverage(short_lev_weighted, short_notional),
        )
    d_target = d_eq * _D(long_notional) / d_ref
    return _DesiredPosition(
        signal="long", side="long",
        target_notional=d_target.quantize(_D("0.00000001")),
        leverage=_weighted_leverage(long_lev_weighted, long_notional),
    )


# ---------------------------------------------------------------------------
# Account / trade state
# ---------------------------------------------------------------------------

@dataclass
class _TradeState:
    direction: int
    open_side: str
    entry_idx: int
    entry_time: Optional[str]
    entry_price: Decimal
    leverage: int
    current_qty: Decimal
    max_qty: Decimal
    entry_margin: Decimal
    gross_realized_pnl: Decimal = Decimal(0)
    entry_fee_paid: Decimal = Decimal(0)
    exit_fee_paid: Decimal = Decimal(0)
    funding_fee_paid: Decimal = Decimal(0)


@dataclass
class _AccountState:
    wallet: Decimal                    # full Decimal precision (matches Go shopspring/decimal)
    net_qty: Decimal = Decimal(0)
    entry_price: Decimal = Decimal(0)
    leverage: int = 1
    trading_fee_paid: Decimal = Decimal(0)
    funding_fee_paid: Decimal = Decimal(0)
    liquidation_count: int = 0
    fill_count: int = 0
    # Per-fill records (side, qty, price, is_liquidation). Used by run_evolve_backtest
    # to replay backend's cross-segment ApplyCompletedTrade aggregation.
    fills: list = field(default_factory=list)


def _build_open_trade(*, direction, open_side, entry_idx, entry_time, entry_price, leverage, qty, fee_paid):
    d_qty = _D(qty)
    d_price = _D(entry_price)
    d_fee = _D(fee_paid)
    margin = d_qty * d_price / max(1, leverage) if d_qty > 0 and d_price > 0 else Decimal(0)
    return _TradeState(
        direction=direction, open_side=open_side, entry_idx=entry_idx,
        entry_time=entry_time, entry_price=d_price, leverage=max(1, int(leverage)),
        current_qty=d_qty, max_qty=d_qty, entry_margin=margin, entry_fee_paid=d_fee,
    )


def _finalize_trade(trade: _TradeState, *, exit_idx, exit_time, exit_price, exit_reason, df) -> Trade:
    d_pnl = trade.gross_realized_pnl - trade.entry_fee_paid - trade.exit_fee_paid + trade.funding_fee_paid
    d_margin = trade.entry_margin
    d_pnl_pct = trade.gross_realized_pnl / d_margin if d_margin > 0 else Decimal(0)
    d_max_qty = max(trade.max_qty, trade.current_qty)
    return Trade(
        entry_idx=trade.entry_idx, entry_price=float(trade.entry_price),
        direction=trade.direction, margin=float(d_margin), leverage=max(1, int(trade.leverage)),
        quantity=float(Decimal(trade.direction) * d_max_qty),
        exit_idx=exit_idx, exit_price=float(_D(exit_price)), pnl=float(d_pnl), pnl_pct=float(d_pnl_pct),
        gross_pnl=float(trade.gross_realized_pnl),
        entry_fee_paid=float(trade.entry_fee_paid), exit_fee_paid=float(trade.exit_fee_paid),
        funding_fee_paid=float(trade.funding_fee_paid), exit_reason=exit_reason,
        entry_time=trade.entry_time, exit_time=exit_time,
        holding_bars=max(0, exit_idx - trade.entry_idx),
        holding_hours=_holding_hours_between(df, trade.entry_idx, exit_idx),
    )


# ---------------------------------------------------------------------------
# Funding
# ---------------------------------------------------------------------------

def _funding_settlement_times(interval_start: pd.Timestamp, interval_end: pd.Timestamp) -> list[pd.Timestamp]:
    start = pd.Timestamp(interval_start).tz_convert("UTC")
    end = pd.Timestamp(interval_end).tz_convert("UTC")
    if end <= start:
        return []
    cursor = start.floor("h")
    if cursor <= start:
        cursor += pd.Timedelta(hours=1)
    out: list[pd.Timestamp] = []
    while cursor <= end:
        out.append(cursor)
        cursor += pd.Timedelta(hours=1)
    return out


def _apply_funding_between(state, open_trade, df, interval_start, interval_end):
    if open_trade is None or state.net_qty == 0:
        return open_trade
    d_rate = _D(FIXED_REPLAY_FUNDING_RATE)
    for settlement in _funding_settlement_times(interval_start, interval_end):
        mark_price = _mark_price_before(df, settlement)
        if mark_price <= 0:
            continue
        d_mark = _D(mark_price)
        funding_fee = -state.net_qty * d_mark * d_rate
        state.wallet += funding_fee
        state.funding_fee_paid += funding_fee
        open_trade.funding_fee_paid += funding_fee
    return open_trade


# ---------------------------------------------------------------------------
# Liquidation
# ---------------------------------------------------------------------------

def _book_liquidation_threshold(net_qty, entry_price, wallet_balance, maintenance_rate=0.004):
    if math.isclose(net_qty, 0.0, abs_tol=1e-12) or entry_price <= 0:
        return 0.0, "", False
    abs_qty = abs(net_qty)
    signed_entry_notional = entry_price * net_qty
    slope = net_qty - maintenance_rate * abs_qty
    if abs_qty <= 0 or abs(slope) <= 1e-12:
        return 0.0, "", False
    price = (signed_entry_notional - wallet_balance) / slope
    if math.isnan(price) or math.isinf(price) or price <= 0:
        return 0.0, "", False
    return price, "down" if slope > 0 else "up", True


def _maybe_liquidate(state, open_trade, bar, *, bar_idx, df):
    if open_trade is None or bar is None or state.net_qty == 0:
        return open_trade, None
    liq_price, liq_dir, ok = _book_liquidation_threshold(float(state.net_qty), float(state.entry_price), float(state.wallet))
    if not ok:
        return open_trade, None
    high, low = float(bar["high"]), float(bar["low"])
    triggered = (liq_dir == "down" and low <= liq_price) or (liq_dir == "up" and high >= liq_price)
    if not triggered:
        return open_trade, None
    d_closed = state.net_qty.copy_abs()
    d_liq = _D(liq_price)
    if state.net_qty > 0:
        realized = (d_liq - state.entry_price) * d_closed
    else:
        realized = (state.entry_price - d_liq) * d_closed
    # Record the closing fill so cross-segment trade aggregation sees the
    # liquidation zero-cross (backend emits a fill with IsLiquidation=true).
    liq_side = "sell" if state.net_qty > 0 else "buy"
    state.fills.append({
        "side": liq_side,
        "qty": float(d_closed),
        "price": float(d_liq),
        "is_liquidation": True,
    })
    state.fill_count += 1
    open_trade.gross_realized_pnl += realized
    state.wallet += realized
    state.net_qty = Decimal(0)
    state.entry_price = Decimal(0)
    state.leverage = 1
    state.liquidation_count += 1
    exit_time = str(df["timestamp"].iloc[bar_idx]) if "timestamp" in df.columns else None
    trade = _finalize_trade(open_trade, exit_idx=bar_idx, exit_time=exit_time, exit_price=liq_price, exit_reason="liquidation", df=df)
    return None, trade


# ---------------------------------------------------------------------------
# Fill / reconcile
# ---------------------------------------------------------------------------

def _apply_fill(state, open_trade, *, side, qty, fill_price, requested_leverage, fee_rate, fill_idx, fill_time, action, df):
    """Full Decimal-precision fill application matching Go source-core."""
    d_qty = _D(qty)
    d_price = _D(fill_price)
    if d_qty <= 0 or d_price <= 0:
        return open_trade, []
    fill_sign = 1 if side == "buy" else -1
    d_sign = _D(fill_sign)
    d_fill_signed = d_sign * d_qty
    prev_net_qty = state.net_qty
    prev_abs_qty = prev_net_qty.copy_abs()
    current_sign = 1 if prev_net_qty > 0 else (-1 if prev_net_qty < 0 else 0)
    d_notional = d_qty * d_price
    d_fee = d_notional * _D(fee_rate)
    state.wallet -= d_fee
    state.trading_fee_paid -= d_fee
    state.fill_count += 1
    # Record fill for cross-segment trade aggregation parity with backend.
    state.fills.append({
        "side": side,
        "qty": float(d_qty),
        "price": float(d_price),
        "is_liquidation": (action == "liquidation"),
    })
    completed = []

    if current_sign == 0 or current_sign == fill_sign:
        new_net = prev_net_qty + d_fill_signed
        d_new_entry = d_price
        if prev_abs_qty > 0:
            d_total = prev_abs_qty + d_qty
            if d_total > 0:
                d_new_entry = (state.entry_price * prev_abs_qty + d_price * d_qty) / d_total
        state.net_qty = new_net
        state.entry_price = d_new_entry if new_net != 0 else Decimal(0)
        state.leverage = max(1, int(requested_leverage or state.leverage or 1))
        if open_trade is None:
            open_trade = _build_open_trade(direction=fill_sign, open_side=side, entry_idx=fill_idx,
                                           entry_time=fill_time, entry_price=d_price,
                                           leverage=state.leverage, qty=d_qty, fee_paid=d_fee)
        else:
            open_trade.entry_price = state.entry_price
            open_trade.current_qty = new_net.copy_abs()
            open_trade.max_qty = max(open_trade.max_qty, new_net.copy_abs())
            open_trade.leverage = state.leverage
            open_trade.entry_margin = new_net.copy_abs() * state.entry_price / max(1, state.leverage)
            open_trade.entry_fee_paid += d_fee
        return open_trade, completed

    closed_qty = min(prev_abs_qty, d_qty)
    realized = Decimal(0)
    if closed_qty > 0:
        if prev_net_qty > 0:
            realized = (d_price - state.entry_price) * closed_qty
        else:
            realized = (state.entry_price - d_price) * closed_qty
        state.wallet += realized

    if open_trade is None:
        open_trade = _build_open_trade(direction=1 if prev_net_qty > 0 else -1,
                                       open_side="buy" if prev_net_qty > 0 else "sell",
                                       entry_idx=fill_idx, entry_time=fill_time,
                                       entry_price=state.entry_price, leverage=state.leverage,
                                       qty=prev_abs_qty, fee_paid=Decimal(0))
    close_ratio = closed_qty / d_qty if d_qty > 0 else Decimal(0)
    close_fee = d_fee * close_ratio
    open_trade.gross_realized_pnl += realized
    open_trade.exit_fee_paid += close_fee

    if prev_abs_qty > d_qty + _D("1e-12"):
        state.net_qty = prev_net_qty + d_fill_signed
        open_trade.current_qty = state.net_qty.copy_abs()
        open_trade.entry_margin = state.net_qty.copy_abs() * state.entry_price / max(1, open_trade.leverage)
        return open_trade, completed

    exit_trade = _finalize_trade(open_trade, exit_idx=fill_idx, exit_time=fill_time, exit_price=d_price, exit_reason=action, df=df)
    completed.append(exit_trade)

    if prev_abs_qty - d_qty < _D("1e-12") and prev_abs_qty - d_qty > _D("-1e-12"):
        state.net_qty = Decimal(0)
        state.entry_price = Decimal(0)
        state.leverage = max(1, int(requested_leverage or state.leverage or 1))
        return None, completed

    remainder = d_qty - prev_abs_qty
    state.net_qty = d_sign * remainder
    state.entry_price = d_price
    state.leverage = max(1, int(requested_leverage or state.leverage or 1))
    open_trade = _build_open_trade(direction=fill_sign, open_side=side, entry_idx=fill_idx,
                                   entry_time=fill_time, entry_price=d_price,
                                   leverage=state.leverage, qty=remainder,
                                   fee_paid=max(Decimal(0), d_fee - close_fee))
    return open_trade, completed


def _reconcile_position(state, open_trade, desired, *, mark_price, fee_rate, fill_idx, fill_time, df):
    completed = []
    current_side = "long" if state.net_qty > 0 else ("short" if state.net_qty < 0 else None)

    # Go flat
    if desired.side is None or desired.target_notional < Decimal(1):
        if current_side == "long":
            fp, fq, _ = simulate_replay_baseline_fill(-1, float(state.net_qty.copy_abs()), mark_price)
            open_trade, trades = _apply_fill(state, open_trade, side="sell", qty=fq, fill_price=fp, requested_leverage=state.leverage, fee_rate=fee_rate, fill_idx=fill_idx, fill_time=fill_time, action="close_long", df=df)
            completed.extend(trades)
        elif current_side == "short":
            fp, fq, _ = simulate_replay_baseline_fill(1, float(state.net_qty.copy_abs()), mark_price)
            open_trade, trades = _apply_fill(state, open_trade, side="buy", qty=fq, fill_price=fp, requested_leverage=state.leverage, fee_rate=fee_rate, fill_idx=fill_idx, fill_time=fill_time, action="close_short", df=df)
            completed.extend(trades)
        return open_trade, completed

    # Flip direction
    if current_side and current_side != desired.side:
        close_dir = -1 if current_side == "long" else 1
        close_side = "sell" if current_side == "long" else "buy"
        action = "flip_close_long" if current_side == "long" else "flip_close_short"
        fp, fq, _ = simulate_replay_baseline_fill(close_dir, float(state.net_qty.copy_abs()), mark_price)
        open_trade, trades = _apply_fill(state, open_trade, side=close_side, qty=fq, fill_price=fp, requested_leverage=state.leverage, fee_rate=fee_rate, fill_idx=fill_idx, fill_time=fill_time, action=action, df=df)
        completed.extend(trades)

    # Adjust size (full Decimal to match Go)
    d_mark = _D(mark_price)
    d_current_notional = d_mark * state.net_qty.copy_abs()
    d_target = desired.target_notional  # already Decimal
    d_diff = d_target - d_current_notional
    # Slightly wider than Go's 2% to compensate for equity drift between
    # Python in-memory state and Go source-core DB state.
    d_tolerance = max(Decimal(1), d_target.copy_abs() * _D("0.02"))
    if d_diff.copy_abs() <= d_tolerance:
        return open_trade, completed

    if d_diff > 0:
        direction = 1 if desired.side == "long" else -1
        side = "buy" if direction > 0 else "sell"
        ref_price = _replay_reference_price(direction, mark_price)
        qty = float(d_diff / _D(ref_price)) if ref_price > 0 else 0.0
        fp, fq, _ = simulate_replay_baseline_fill(direction, qty, mark_price)
        open_trade, trades = _apply_fill(state, open_trade, side=side, qty=fq, fill_price=fp, requested_leverage=desired.leverage, fee_rate=fee_rate, fill_idx=fill_idx, fill_time=fill_time, action=f"increase_{desired.side}", df=df)
        completed.extend(trades)
    elif state.net_qty != 0:
        direction = -1 if desired.side == "long" else 1
        ref_price = _replay_reference_price(direction, mark_price)
        close_qty = min(abs(state.net_qty), float(d_diff.copy_abs() / _D(ref_price)) if ref_price > 0 else 0.0)
        if close_qty > 0:
            side = "sell" if desired.side == "long" else "buy"
            fp, fq, _ = simulate_replay_baseline_fill(direction, close_qty, mark_price)
            open_trade, trades = _apply_fill(state, open_trade, side=side, qty=fq, fill_price=fp, requested_leverage=state.leverage, fee_rate=fee_rate, fill_idx=fill_idx, fill_time=fill_time, action=f"reduce_{desired.side}", df=df)
            completed.extend(trades)

    return open_trade, completed


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_backtest(
    df: pd.DataFrame,
    params: DecisionParams,
    regime: pd.Series,
    initial_capital: float = 10000.0,
    precomputed_signals: pd.Series = None,
    maintenance_rate: float = 0.004,
    timeframe: str = "15m",
    lookback_bars: int = DEFAULT_LOOKBACK_BARS,
    window_start: pd.Timestamp = None,
    window_end: pd.Timestamp = None,
    close_open_positions_at_end: bool = False,
) -> BacktestResult:
    """Replay-aligned two-layer backtest matching Go verify.

    Data must include ``lookback_bars`` leading bars before the window.
    """
    df = df.copy().reset_index(drop=True)
    has_ts = "timestamp" in df.columns
    if has_ts:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

    fee_rate = local_taker_fee_rate()
    step = _infer_step_duration(df)
    if window_start is not None:
        start = pd.Timestamp(window_start)
    else:
        # Match Go fixedDatasetReplayWindowPolicy: scoredStart = baseBars[0] + step
        # First bar is warmup only, scoring starts from the second bar
        start = pd.Timestamp(df["timestamp"].iloc[0]) + step
    start = start.tz_convert("UTC") if start.tzinfo else start.tz_localize("UTC")
    end = pd.Timestamp(window_end) if window_end is not None else (pd.Timestamp(df["timestamp"].iloc[-1]) + step)
    end = end.tz_convert("UTC") if end.tzinfo else end.tz_localize("UTC")

    evaluation_at = _first_replay_evaluation_at(start, step)
    prev_evaluation_at = start
    timestamps = pd.DatetimeIndex(pd.to_datetime(df["timestamp"], utc=True))

    d_initial = _D(initial_capital)
    state = _AccountState(wallet=d_initial)
    open_trade: Optional[_TradeState] = None
    completed_trades: list[Trade] = []
    equity_points = [initial_capital]

    # Persistent incremental evaluator — advances one bar per step, keeps state
    # Matches Go: useIncrementalDesired=true, advanceReplayDesiredPositionStepper
    evaluator = RealtimeIncrementalEvaluator(
        params,
        initial_capital=initial_capital,
        regime_window=REPLAY_ALIGNED_REGIME_WINDOW,
    )
    last_fed_idx = -1  # track which bars have been fed to evaluator

    while evaluation_at <= end:
        end_idx = _search_left_timestamp(timestamps, evaluation_at)
        if end_idx <= 0:
            prev_evaluation_at = evaluation_at
            evaluation_at += step
            continue

        # Feed bars to evaluator up to (but not including) evaluation bar
        # Matches Go: advanceReplayDesiredPositionStepper advances to evaluationAt - step
        for feed_idx in range(last_fed_idx + 1, end_idx):
            row = df.iloc[feed_idx]
            evaluator.step(IncrementalBar(
                timestamp=pd.Timestamp(row["timestamp"]).tz_convert("UTC") if pd.Timestamp(row["timestamp"]).tzinfo else pd.Timestamp(row["timestamp"]).tz_localize("UTC"),
                open=float(row["open"]),
                high=float(row["high"]),
                low=float(row["low"]),
                close=float(row["close"]),
                volume=float(row["volume"]),
            ))
        last_fed_idx = max(last_fed_idx, end_idx - 1)

        # Mark price from bar AT evaluation_at (not before it)
        # Matches Go: ReplayQuote(evaluationAt) → loadReplayQuoteCandle(evaluationAt)
        # → returns the 1m bar at evaluationAt → close = 15m bar at evaluationAt
        mark_bar_idx = end_idx  # bar AT evaluation_at
        if mark_bar_idx >= len(df):
            mark_bar_idx = len(df) - 1
        bar = df.iloc[mark_bar_idx]
        fill_time = str(df["timestamp"].iloc[mark_bar_idx])

        d_mark = _D(bar["close"])
        mark_price = float(d_mark)

        # Funding settlement between intervals (no recovery/liquidation for fixed dataset)
        open_trade = _apply_funding_between(state, open_trade, df, prev_evaluation_at, evaluation_at)

        # Equity = wallet + unrealized (Decimal precision)
        d_unrealized = Decimal(0)
        if state.net_qty > 0:
            d_unrealized = (d_mark - state.entry_price) * state.net_qty.copy_abs()
        elif state.net_qty < 0:
            d_unrealized = (state.entry_price - d_mark) * state.net_qty.copy_abs()
        d_equity = state.wallet + d_unrealized

        # Snapshot 1: mark-to-market BEFORE reconcile
        equity_points.append(float(d_equity))

        desired = _derive_desired_from_evaluator(
            evaluator,
            reference_initial_fund=d_initial,
            current_equity=d_equity,
        )

        open_trade, trades = _reconcile_position(
            state, open_trade, desired,
            mark_price=mark_price, fee_rate=fee_rate,
            fill_idx=mark_bar_idx, fill_time=fill_time, df=df,
        )
        completed_trades.extend(trades)

        # Snapshot 2: mark-to-market AFTER reconcile
        d_unreal2 = Decimal(0)
        if state.net_qty > 0:
            d_unreal2 = (d_mark - state.entry_price) * state.net_qty.copy_abs()
        elif state.net_qty < 0:
            d_unreal2 = (state.entry_price - d_mark) * state.net_qty.copy_abs()
        equity_points.append(float(state.wallet + d_unreal2))
        prev_evaluation_at = evaluation_at
        evaluation_at += step

    # End of data
    open_positions: list[Trade] = []
    if open_trade is not None and state.net_qty != 0:
        last_idx = len(df) - 1
        last_time = str(df["timestamp"].iloc[last_idx]) if has_ts else None
        last_mark = float(df["close"].iloc[last_idx])
        if close_open_positions_at_end:
            close_dir = -1 if state.net_qty > 0 else 1
            close_side = "sell" if state.net_qty > 0 else "buy"
            fp, fq, _ = simulate_replay_baseline_fill(close_dir, float(state.net_qty.copy_abs()), last_mark)
            open_trade, trades = _apply_fill(state, open_trade, side=close_side, qty=fq, fill_price=fp,
                                             requested_leverage=state.leverage, fee_rate=fee_rate,
                                             fill_idx=last_idx, fill_time=last_time, action="end_of_data", df=df)
            completed_trades.extend(trades)
            if equity_points:
                equity_points[-1] = float(state.wallet)
        else:
            d_last = _D(last_mark)
            d_margin = max(open_trade.entry_margin, state.net_qty.copy_abs() * state.entry_price / max(1, state.leverage))
            d_gpnl = (d_last - state.entry_price) * state.net_qty.copy_abs() if state.net_qty > 0 else (state.entry_price - d_last) * state.net_qty.copy_abs()
            open_positions.append(Trade(
                entry_idx=open_trade.entry_idx, entry_price=float(open_trade.entry_price),
                direction=open_trade.direction,
                margin=float(d_margin),
                leverage=max(1, int(state.leverage)),
                quantity=float(Decimal(open_trade.direction) * state.net_qty.copy_abs()),
                entry_time=open_trade.entry_time,
                gross_pnl=float(d_gpnl),
                entry_fee_paid=float(open_trade.entry_fee_paid),
                funding_fee_paid=float(open_trade.funding_fee_paid),
            ))

    result = _build_result(
        completed_trades, pd.Series(equity_points, dtype=float),
        blowup_count=0, total_deposited=initial_capital, initial_capital=initial_capital,
        open_positions=open_positions, fill_count=state.fill_count,
    )
    result.fills = list(state.fills)
    return result


# ---------------------------------------------------------------------------
# Result builder
# ---------------------------------------------------------------------------

def _build_result(trades, equity, blowup_count=0, total_deposited=0.0, initial_capital=10000.0,
                  open_positions=None, fill_count=0):
    ending_equity = float(equity.iloc[-1]) if len(equity) else initial_capital
    total_return = (ending_equity - initial_capital) / initial_capital if initial_capital > 0 else 0
    net_pnl = ending_equity - initial_capital

    peak = equity.expanding().max()
    dd = equity - peak
    safe_peak = peak.replace(0, np.nan)
    drawdown = (dd / safe_peak).min()
    drawdown = drawdown if not np.isnan(drawdown) else 0

    wins = [t for t in trades if t.gross_pnl > 0]
    losses = [t for t in trades if t.gross_pnl <= 0]
    win_rate = len(wins) / len(trades) if trades else 0

    returns = equity.pct_change().dropna()
    valid_returns = returns.replace([np.inf, -np.inf], np.nan).dropna()
    # Use population std (ddof=0) to match Go verify: variance /= float64(len(returns))
    sharpe = (valid_returns.mean() / valid_returns.std(ddof=0) * np.sqrt(8760)) if len(valid_returns) > 1 and valid_returns.std(ddof=0) > 0 else 0

    total_win = sum(t.gross_pnl for t in wins)
    total_loss = abs(sum(t.gross_pnl for t in losses if t.gross_pnl < 0))
    if total_loss > 0:
        pf = total_win / total_loss
    elif total_win > 0:
        pf = REPLAY_ALIGNED_PROFIT_FACTOR_CAP
    else:
        pf = 0.0

    gross_profit_pnl = sum(max(t.gross_pnl, 0.0) for t in trades)
    gross_loss_pnl = sum(min(t.gross_pnl, 0.0) for t in trades)
    trading_fee_paid = -sum(t.entry_fee_paid + t.exit_fee_paid for t in trades)
    funding_fee_paid = sum(t.funding_fee_paid for t in trades)
    if open_positions:
        trading_fee_paid -= sum(float(p.entry_fee_paid + p.exit_fee_paid) for p in open_positions)
        funding_fee_paid += sum(float(p.funding_fee_paid) for p in open_positions)
    liquidation_count = sum(1 for t in trades if t.exit_reason == "liquidation")
    long_count = sum(1 for t in trades if t.direction == 1)
    short_count = sum(1 for t in trades if t.direction == -1)

    avg_win = np.mean([t.pnl_pct for t in wins]) if wins else 0
    avg_loss = np.mean([t.pnl_pct for t in losses]) if losses else 0
    avg_holding_bars = np.mean([t.holding_bars for t in trades]) if trades else 0
    avg_holding_hours = np.mean([t.holding_hours for t in trades]) if trades else 0

    consec_w = consec_l = max_cw = max_cl = 0
    for t in trades:
        if t.gross_pnl > 0:
            consec_w += 1; consec_l = 0; max_cw = max(max_cw, consec_w)
        else:
            consec_l += 1; consec_w = 0; max_cl = max(max_cl, consec_l)

    return BacktestResult(
        backend="local_backtest_replay_aligned",
        execution_profile=REPLAY_BASELINE_PROFILE,
        initial_equity=initial_capital, ending_equity=ending_equity, net_pnl=net_pnl,
        total_return=total_return, sharpe_ratio=sharpe, max_drawdown=abs(drawdown),
        win_rate=win_rate, profit_factor=pf, total_trades=len(trades),
        avg_trade_pnl=np.mean([t.pnl_pct for t in trades]) if trades else 0,
        avg_win=avg_win, avg_loss=avg_loss,
        max_consecutive_wins=max_cw, max_consecutive_losses=max_cl,
        trades=trades, equity_curve=equity, regime_performance={},
        blowup_count=blowup_count, liquidation_count=liquidation_count,
        total_deposited=total_deposited if total_deposited > 0 else initial_capital,
        gross_profit_pnl=gross_profit_pnl, gross_loss_pnl=gross_loss_pnl,
        trading_fee_paid=trading_fee_paid, funding_fee_paid=funding_fee_paid,
        fill_count=fill_count, long_trade_count=long_count, short_trade_count=short_count,
        avg_holding_bars=avg_holding_bars, avg_holding_hours=avg_holding_hours,
        open_positions=open_positions or [],
    )
