#!/usr/bin/env python3
"""Segmented backtest with evolution checkpoints.

Supports two modes:
  1. Baseline: run all segments with same params, output per-segment results
  2. Evolution: apply pre-defined param changes per segment (evolution_schedule)

Capital is continuous across segments (not reset each segment).

Usage:
    python run_evolve_backtest.py --data data.csv --params-file params.json --segment-bars 672
    python run_evolve_backtest.py --data data.csv --evolution-file evolution_schedule.json
"""

import argparse
import json
import sys
import os
import copy
from datetime import timezone
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
from core.decision import DecisionParams, compute_signals
from core.regime import classify_regime
from core.backtest import run_backtest, _build_result
from core.indicators import atr as compute_atr


PERSONALITY_FIELDS = [
    "long_bias",
    "base_leverage", "max_leverage",
    "risk_per_trade", "max_position_pct",
    "rolling_enabled", "rolling_trigger_pct", "rolling_reinvest_pct",
    "rolling_max_times", "rolling_move_stop",
    "trend_weight", "momentum_weight", "mean_revert_weight",
    "volume_weight", "volatility_weight",
]

TACTICAL_FLOAT_FIELDS = [
    "entry_threshold", "exit_threshold",
    "sl_atr_mult", "tp_rr_ratio",
    "trailing_activation_pct", "trailing_distance_atr",
    "regime_sensitivity",
    "supertrend_mult", "trend_strength_min",
]

MAX_DRIFT_PCT = 0.30


def _aggregate_trades_backend_style(fills: list) -> dict:
    """Replay backend's AggregateRealtimeSourceTradeAggFromFillRows on merged fills.

    Uses Decimal throughout to match backend's shopspring/decimal precision and
    avoid spurious zero-crossings from float residuals.

    Backend tracks a single `positions` map across all segments' fills, so the
    trade count / gross profit / gross loss / win rate / profit factor reflect
    cross-segment phantom open-close crossings as well as intra-segment ones.
    This mirrors internal/repository/agent_trader.go:949.
    """
    from decimal import Decimal as D
    ZERO = D(0)
    # Python fills are float-derived from simulate_replay_baseline_fill; tiny
    # rounding drift (O(1e-16)) can turn an intended exact-close into over-flip
    # under strict Decimal equality, producing phantom trades + residual dust
    # that cascades across subsequent fills. Align with _apply_fill's 1e-12
    # tolerance and zero out sub-tolerance residuals so the aggregator treats
    # the same fills as _apply_fill does.
    TOL = D("1e-10")
    net_qty = ZERO
    entry_price = ZERO
    open_side = ""
    accum_realized = ZERO
    total_trades = 0
    wins = 0
    long_total = 0
    short_total = 0
    long_wins = 0
    short_wins = 0
    gross_profit = ZERO
    gross_loss = ZERO

    def apply_completed_trade(side: str, realized):
        nonlocal total_trades, wins, long_total, short_total, long_wins, short_wins, gross_profit, gross_loss
        total_trades += 1
        if side == "buy":
            long_total += 1
            if realized > ZERO:
                long_wins += 1
        elif side == "sell":
            short_total += 1
            if realized > ZERO:
                short_wins += 1
        if realized > ZERO:
            wins += 1
            gross_profit = gross_profit + realized
        elif realized < ZERO:
            gross_loss = gross_loss + (-realized)

    for fill in fills:
        side = fill["side"]
        qty = D(str(fill["qty"]))
        price = D(str(fill["price"]))
        trade_sign = 1 if side == "buy" else -1
        trade_signed_qty = D(trade_sign) * qty
        current_sign = 1 if net_qty > ZERO else (-1 if net_qty < ZERO else 0)
        prev_open_side = open_side

        if current_sign == 0 or current_sign == trade_sign:
            old_abs = abs(net_qty)
            total_abs = old_abs + qty
            if old_abs <= ZERO:
                entry_price = price
            elif total_abs > ZERO:
                entry_price = (entry_price * old_abs + price * qty) / total_abs
            net_qty = net_qty + trade_signed_qty
            realized = ZERO
        elif abs(net_qty) > qty + TOL:
            if net_qty > ZERO:
                realized = (price - entry_price) * qty
            else:
                realized = (entry_price - price) * qty
            net_qty = net_qty + trade_signed_qty
        elif abs(abs(net_qty) - qty) <= TOL:
            if net_qty > ZERO:
                realized = (price - entry_price) * qty
            else:
                realized = (entry_price - price) * qty
            net_qty = ZERO
            entry_price = ZERO
        else:
            closed_qty = abs(net_qty)
            if net_qty > ZERO:
                realized = (price - entry_price) * closed_qty
            else:
                realized = (entry_price - price) * closed_qty
            remainder = qty - closed_qty
            net_qty = D(trade_sign) * remainder
            entry_price = price

        if abs(net_qty) <= TOL:
            net_qty = ZERO
            entry_price = ZERO
        next_sign = 1 if net_qty > ZERO else (-1 if net_qty < ZERO else 0)

        if current_sign == 0:
            if next_sign != 0:
                open_side = side
                accum_realized = ZERO
        elif next_sign == 0:
            accum_realized = accum_realized + realized
            apply_completed_trade(prev_open_side, accum_realized)
            open_side = ""
            accum_realized = ZERO
        elif current_sign != next_sign:
            accum_realized = accum_realized + realized
            apply_completed_trade(prev_open_side, accum_realized)
            open_side = side
            accum_realized = ZERO
        else:
            accum_realized = accum_realized + realized

    if gross_loss == ZERO and gross_profit > ZERO:
        profit_factor = 999999.0
    elif gross_loss == ZERO:
        profit_factor = 0.0
    else:
        profit_factor = float(gross_profit / gross_loss)
    win_rate = wins / total_trades if total_trades > 0 else 0.0

    return {
        "total_trades": total_trades,
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "gross_profit": float(gross_profit),
        "gross_loss": float(gross_loss),
        "long_total": long_total,
        "short_total": short_total,
        "long_wins": long_wins,
        "short_wins": short_wins,
    }


def resolve_params_dict(raw_params: Optional[dict]) -> dict:
    """Materialize missing fields with DecisionParams defaults."""
    raw_params = raw_params or {}
    clean = {k: v for k, v in raw_params.items() if v is not None}
    return DecisionParams.from_dict(clean).to_dict()


def clamp_tactical_drift(current_params: dict, initial_params: dict) -> dict:
    """Clamp tactical float params to initial_value +/- 30%. Prevent grinding to no-trade."""
    result = current_params.copy()
    for field in TACTICAL_FLOAT_FIELDS:
        if field not in initial_params or field not in result:
            continue
        init_val = initial_params[field]
        curr_val = result[field]
        if isinstance(init_val, (int, float)) and init_val != 0:
            lo = init_val * (1 - MAX_DRIFT_PCT)
            hi = init_val * (1 + MAX_DRIFT_PCT)
            if lo > hi:
                lo, hi = hi, lo
            result[field] = max(lo, min(curr_val, hi))
    return result


def lock_personality(current_params: dict, initial_params: dict) -> dict:
    """Force personality fields back to initial values."""
    result = current_params.copy()
    for field in PERSONALITY_FIELDS:
        if field in initial_params:
            result[field] = initial_params[field]
    return result


def _to_rfc3339(ts) -> str:
    if ts is None:
        return ""
    if hasattr(ts, "tzinfo"):
        if ts.tzinfo is None:
            ts = ts.tz_localize(timezone.utc) if hasattr(ts, "tz_localize") else ts.replace(tzinfo=timezone.utc)
        else:
            ts = ts.tz_convert(timezone.utc) if hasattr(ts, "tz_convert") else ts.astimezone(timezone.utc)
        return ts.isoformat().replace("+00:00", "Z")
    return str(ts)


def main():
    parser = argparse.ArgumentParser(description="Segmented backtest with evolution")
    parser.add_argument("--data", required=True, help="OHLCV CSV path")
    parser.add_argument("--params", default=None, help="Initial params JSON string")
    parser.add_argument("--params-file", default=None, help="Initial params JSON file")
    parser.add_argument("--segment-bars", type=int, default=672, help="Bars per segment (672=7d@15m)")
    parser.add_argument("--capital", type=float, default=10000.0)
    parser.add_argument("--regime-version", default="v1")
    parser.add_argument("--evolution-file", default=None,
                        help="JSON file with per-segment params [{round, params}, ...]")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    if args.params:
        initial_params = json.loads(args.params)
    elif args.params_file:
        with open(args.params_file) as f:
            initial_params = json.load(f)
    else:
        print("Error: --params or --params-file required", file=sys.stderr)
        sys.exit(1)
    initial_params = resolve_params_dict(initial_params)

    evolution_schedule = None
    if args.evolution_file:
        with open(args.evolution_file) as f:
            evolution_schedule = json.load(f)

    df = pd.read_csv(args.data, parse_dates=["timestamp"])
    regime = classify_regime(df, version=args.regime_version)

    total_bars = len(df)
    n_segments = max(1, total_bars // args.segment_bars)
    has_ts = "timestamp" in df.columns

    current_params = copy.deepcopy(initial_params)
    evolution_log = []

    # Resolve per-segment params (applying evolution_schedule) and boundaries
    seg_boundaries = []  # (seg_idx, seg_start, seg_end, seg_start_time, seg_end_time, seg_params)
    current_params_track = copy.deepcopy(initial_params)
    for seg_idx in range(n_segments):
        seg_start = seg_idx * args.segment_bars
        seg_end = min((seg_idx + 1) * args.segment_bars, total_bars)
        if seg_end <= seg_start:
            break
        if evolution_schedule and seg_idx < len(evolution_schedule):
            evo_params = evolution_schedule[seg_idx].get("params", current_params_track)
            evo_params = lock_personality(evo_params, initial_params)
            evo_params = clamp_tactical_drift(evo_params, initial_params)
            current_params_track = resolve_params_dict(evo_params)
        seg_params = copy.deepcopy(current_params_track)
        # seg0 start skips CSV first bar (backend clamps to scoredStart = iloc[0] + step anyway).
        start_iloc = seg_start if seg_start > 0 else 1
        seg_start_time = _to_rfc3339(df["timestamp"].iloc[start_iloc]) if has_ts else f"bar_{seg_start}"
        seg_end_time = _to_rfc3339(df["timestamp"].iloc[seg_end - 1]) if has_ts else f"bar_{seg_end}"
        seg_boundaries.append((seg_idx, seg_start, seg_end, seg_start_time, seg_end_time, seg_params))

    # Per-segment INDEPENDENT backtests with fresh $capital — matches backend verify
    # replay: each segment runs ReplaySeedWindowInMemory with StartingEquityInitial.
    # Cap warmup to 1200 bars to match backend seed.LookbackBars default: backend's
    # stepper.Reset feeds ReplayBars(evaluationAt, 1200) on the first tick, so each
    # segment's evaluator sees at most 1200 prior bars, not the full dataset history.
    SEGMENT_WARMUP_BARS = 1200
    seg_results = []
    for seg_idx, seg_start, seg_end, seg_start_time, seg_end_time, seg_params in seg_boundaries:
        seg_params_obj = DecisionParams.from_dict(seg_params)
        window_start = pd.Timestamp(seg_start_time)
        window_end = pd.Timestamp(seg_end_time)
        warmup_start = max(0, seg_start - SEGMENT_WARMUP_BARS)
        # Slice df to [warmup_start : seg_end] so the evaluator inside run_backtest
        # can only feed at most SEGMENT_WARMUP_BARS before the segment window.
        seg_df = df.iloc[warmup_start:seg_end].reset_index(drop=True)
        seg_res = run_backtest(
            seg_df, seg_params_obj, regime,
            initial_capital=args.capital,
            window_start=window_start,
            window_end=window_end,
        )
        seg_results.append(seg_res)

    # Merge segment artifacts (mimics backend MergeFrom + CollectMetric).
    # Backend merged equity curve = [account.InitialEquity] + concat(segment snapshots)
    #                             + finalState (only if differs from last snapshot).
    # Python segment equity_curve starts with [initial_capital] + per-bar pre/post snapshots;
    # drop duplicate $initial prefix from segments 1..N so we have one $10k at the very start.
    merged_trades = []
    merged_equity_values = [args.capital]
    for r in seg_results:
        merged_trades.extend(r.trades)
        seg_eq = list(r.equity_curve)
        # run_backtest always prepends initial_capital; skip it to avoid duplicate resets.
        if seg_eq and abs(seg_eq[0] - args.capital) < 1e-9:
            seg_eq = seg_eq[1:]
        merged_equity_values.extend(seg_eq)
    merged_equity_series = pd.Series(merged_equity_values, dtype=float)

    full_result = _build_result(
        merged_trades, merged_equity_series,
        blowup_count=sum(getattr(r, "blowup_count", 0) for r in seg_results),
        total_deposited=args.capital,
        initial_capital=args.capital,
        open_positions=None,
        fill_count=sum(getattr(r, "fill_count", 0) for r in seg_results),
    )
    # Backend computes trade-level stats via AggregateRealtimeSourceTradeAggFromFillRows
    # over MERGED fills with a single shared `positions` map. Replay that here on our
    # per-segment fills so total_trades / profit_factor / win_rate line up.
    merged_fills = []
    for r in seg_results:
        merged_fills.extend(getattr(r, "fills", []))
    backend_agg = _aggregate_trades_backend_style(merged_fills)
    full_result.total_trades = backend_agg["total_trades"]
    full_result.win_rate = backend_agg["win_rate"]
    full_result.profit_factor = backend_agg["profit_factor"]
    full_result.long_trade_count = backend_agg["long_total"]
    full_result.short_trade_count = backend_agg["short_total"]

    # Build per-segment evolution_log entries from each segment's own result.
    cumulative_return = 0.0
    peak_return = 0.0
    current_params_track = copy.deepcopy(initial_params)
    for (seg_idx, seg_start, seg_end, seg_start_time, seg_end_time, seg_params), seg_res in zip(
        seg_boundaries, seg_results
    ):
        current_params_track = seg_params

        seg_return = seg_res.total_return
        seg_trades = list(seg_res.trades)
        seg_wins = [t for t in seg_trades if t.gross_pnl > 0]
        seg_losses = [t for t in seg_trades if t.gross_pnl <= 0]
        seg_wr = len(seg_wins) / len(seg_trades) if seg_trades else 0

        # Cumulative view keyed off the merged (independent-segment) run.
        cumulative_return = (cumulative_return + 1.0) * (seg_return + 1.0) - 1.0

        exit_reasons = {}
        for t in seg_trades:
            exit_reasons[t.exit_reason] = exit_reasons.get(t.exit_reason, 0) + 1

        avg_win_pct = np.mean([t.pnl_pct for t in seg_wins]) if seg_wins else 0
        avg_loss_pct = np.mean([t.pnl_pct for t in seg_losses]) if seg_losses else 0

        longs = sum(1 for t in seg_trades if t.direction == 1)
        shorts = sum(1 for t in seg_trades if t.direction == -1)

        recent_trades = []
        for t in seg_trades[-8:]:
            recent_trades.append({
                "direction": "LONG" if t.direction == 1 else "SHORT",
                "entry_price": round(t.entry_price, 2),
                "exit_price": round(t.exit_price, 2) if t.exit_price else None,
                "pnl_pct": round(t.pnl_pct * 100, 1),
                "leverage": t.leverage,
                "exit_reason": t.exit_reason,
            })

        seg_price_start = df["close"].iloc[seg_start]
        seg_price_end = df["close"].iloc[min(seg_end - 1, len(df) - 1)]
        seg_price_change = (seg_price_end / seg_price_start - 1) * 100

        peak_return = max(cumulative_return, peak_return) if seg_idx > 0 else cumulative_return
        drawdown_from_peak = peak_return - cumulative_return

        # Cumulative across segments processed so far (segment trades re-index per segment
        # since each runs independently; aggregate by summing each segment's trades).
        cum_wins = sum(sum(1 for t in seg_results[k].trades if t.gross_pnl > 0)
                       for k in range(seg_idx + 1))
        cum_total = sum(len(seg_results[k].trades) for k in range(seg_idx + 1))
        cum_wr = cum_wins / cum_total * 100 if cum_total else 0

        recent_seg_returns = [e["segment_result"]["total_return"] for e in evolution_log[-3:]] if evolution_log else []

        entry = {
            "round": seg_idx + 1,
            "time_range": [seg_start_time, seg_end_time],
            "bars": seg_end - seg_start,
            "params_used": current_params_track,
            "segment_result": {
                "total_return": round(seg_return, 4),
                "total_trades": len(seg_trades),
                "win_rate": round(seg_wr, 4),
                "blowup_count": 0,
                "exit_reasons": exit_reasons,
                "avg_win_pct": round(avg_win_pct * 100, 1),
                "avg_loss_pct": round(avg_loss_pct * 100, 1),
                "longs": longs,
                "shorts": shorts,
            },
            "market_context": {
                "price_start": round(seg_price_start, 2),
                "price_end": round(seg_price_end, 2),
                "price_change_pct": round(seg_price_change, 1),
                "regime": regime.iloc[seg_start],
            },
            "cumulative_context": {
                "cumulative_return": round(cumulative_return, 4),
                "peak_return": round(peak_return, 4),
                "drawdown_from_peak": round(drawdown_from_peak, 4),
                "total_trades_so_far": cum_total,
                "cumulative_win_rate": round(cum_wr, 1),
                "recent_3_seg_returns": [round(r * 100, 1) for r in recent_seg_returns],
            },
            "recent_trades": recent_trades,
        }
        evolution_log.append(entry)

        print(f"  Segment {seg_idx+1}/{len(seg_boundaries)}: "
              f"{seg_start_time[:10]}~{seg_end_time[:10]} | "
              f"seg={seg_return*100:+.1f}% | "
              f"trades={len(seg_trades)} (L{longs}/S{shorts}) | "
              f"exits={exit_reasons} | "
              f"cumulative={cumulative_return*100:+.1f}%",
              file=sys.stderr)

    trades_list = []
    for t in full_result.trades[:500]:
        trades_list.append({
            "entry_time": t.entry_time or "",
            "exit_time": t.exit_time or "",
            "direction": "LONG" if t.direction == 1 else "SHORT",
            "entry_price": round(t.entry_price, 2),
            "exit_price": round(t.exit_price, 2) if t.exit_price else None,
            "pnl_pct": round(t.pnl_pct, 4),
            "leverage": t.leverage,
            "margin": round(t.margin, 2),
            "exit_reason": t.exit_reason,
        })

    output = {
        "initial_params": initial_params,
        "evolution_log": evolution_log,
        "final_params": resolve_params_dict(current_params),
        "backtest_result": full_result.to_dict(),
        "equity_curve": full_result.equity_curve.tolist(),
        "trades": trades_list,
        "summary": {
            "segments": len(seg_boundaries),
            "segment_bars": args.segment_bars,
            "total_return": round(full_result.total_return, 4),
            "sharpe": round(full_result.sharpe_ratio, 4),
            "max_drawdown": round(full_result.max_drawdown, 4),
            "total_trades": full_result.total_trades,
            "win_rate": round(full_result.win_rate, 4),
            "blowup_count": full_result.blowup_count,
        },
    }

    if args.output:
        with open(args.output, "w") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        print(f"Result saved to {args.output}", file=sys.stderr)

    print(json.dumps(output["summary"], indent=2))
    print(json.dumps({"evolution_log": [{
        "round": e["round"],
        "time_range": e["time_range"],
        "seg_return": f"{e['segment_result']['total_return']*100:+.1f}%",
        "trades": e["segment_result"]["total_trades"],
        "cumulative": f"{e['cumulative_context']['cumulative_return']*100:+.1f}%",
    } for e in evolution_log]}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
