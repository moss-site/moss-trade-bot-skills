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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
from core.decision import DecisionParams, compute_signals
from core.regime import classify_regime
from core.backtest import run_backtest
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


def resolve_params_dict(raw_params: dict | None) -> dict:
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
    parser.add_argument("--regime-min-duration", type=int, default=192)
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
    regime = classify_regime(df, version=args.regime_version, min_duration=args.regime_min_duration)

    total_bars = len(df)
    n_segments = max(1, total_bars // args.segment_bars)
    has_ts = "timestamp" in df.columns

    current_params = copy.deepcopy(initial_params)
    evolution_log = []
    all_signals = pd.Series(0, index=df.index, dtype=int)

    # Pre-compute signals for all segments (with lookback)
    for seg_idx in range(n_segments):
        seg_start = seg_idx * args.segment_bars
        seg_end = min((seg_idx + 1) * args.segment_bars, total_bars)
        if seg_end <= seg_start:
            break

        if evolution_schedule and seg_idx < len(evolution_schedule):
            evo_params = evolution_schedule[seg_idx].get("params", current_params)
            evo_params = lock_personality(evo_params, initial_params)
            evo_params = clamp_tactical_drift(evo_params, initial_params)
            current_params = resolve_params_dict(evo_params)

        lookback = min(200, seg_start)
        calc_start = seg_start - lookback
        calc_df = df.iloc[calc_start:seg_end].reset_index(drop=True)
        calc_regime = regime.iloc[calc_start:seg_end].reset_index(drop=True)

        params_obj = DecisionParams.from_dict(current_params)
        signals = compute_signals(calc_df, params_obj, calc_regime)

        for j in range(lookback, len(signals)):
            global_idx = seg_start + (j - lookback)
            if global_idx < total_bars:
                all_signals.iloc[global_idx] = signals.iloc[j]

    # Run full backtest with stitched signals (continuous capital)
    final_params_obj = DecisionParams.from_dict(current_params)
    full_result = run_backtest(df, final_params_obj, regime,
                               initial_capital=args.capital,
                               precomputed_signals=all_signals)

    # Compute per-segment stats from the full result trades
    seg_boundaries = []
    for seg_idx in range(n_segments):
        seg_start = seg_idx * args.segment_bars
        seg_end = min((seg_idx + 1) * args.segment_bars, total_bars)
        if seg_end <= seg_start:
            break
        seg_start_time = _to_rfc3339(df["timestamp"].iloc[seg_start]) if has_ts else f"bar_{seg_start}"
        seg_end_time = _to_rfc3339(df["timestamp"].iloc[seg_end - 1]) if has_ts else f"bar_{seg_end}"
        seg_boundaries.append((seg_idx, seg_start, seg_end, seg_start_time, seg_end_time))

    equity = full_result.equity_curve
    cumulative_return = 0.0
    peak_return = 0.0

    current_params_track = copy.deepcopy(initial_params)
    for seg_idx, seg_start, seg_end, seg_start_time, seg_end_time in seg_boundaries:
        if evolution_schedule and seg_idx < len(evolution_schedule):
            evo_params = evolution_schedule[seg_idx].get("params", current_params_track)
            evo_params = lock_personality(evo_params, initial_params)
            evo_params = clamp_tactical_drift(evo_params, initial_params)
            current_params_track = resolve_params_dict(evo_params)

        eq_start = equity.iloc[seg_start] if seg_start < len(equity) else args.capital
        eq_end = equity.iloc[min(seg_end, len(equity) - 1)]
        seg_return = (eq_end - eq_start) / max(eq_start, 1) if eq_start > 0 else 0

        seg_trades = [t for t in full_result.trades
                      if t.entry_idx >= seg_start and t.entry_idx < seg_end]
        seg_wins = [t for t in seg_trades if t.pnl > 0]
        seg_losses = [t for t in seg_trades if t.pnl <= 0]
        seg_wr = len(seg_wins) / len(seg_trades) if seg_trades else 0

        cumulative_return = (eq_end - args.capital) / args.capital

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

        all_trades_so_far = [t for t in full_result.trades if t.entry_idx < seg_end]
        cum_wins = sum(1 for t in all_trades_so_far if t.pnl > 0)
        cum_total = len(all_trades_so_far)
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
