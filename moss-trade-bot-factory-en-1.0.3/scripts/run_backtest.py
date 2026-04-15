#!/usr/bin/env python3
"""Run backtest with given params JSON and data CSV.

Usage:
    python run_backtest.py --data data.csv --params '{"trend_weight":0.45,...}'
    python run_backtest.py --data data.csv --params-file bot_params.json --capital 10000
"""

import argparse
import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
from core.decision import DecisionParams
from core.regime import classify_regime
from core.backtest import run_backtest


def main():
    parser = argparse.ArgumentParser(description="Run backtest")
    parser.add_argument("--data", required=True, help="Path to OHLCV CSV")
    parser.add_argument("--params", default=None, help="Params as JSON string")
    parser.add_argument("--params-file", default=None, help="Path to params JSON file")
    parser.add_argument("--capital", type=float, default=10000.0)
    parser.add_argument("--regime-version", default="v1")
    parser.add_argument("--regime-min-duration", type=int, default=192)
    parser.add_argument("--output", default=None, help="Output result JSON path")
    args = parser.parse_args()

    if args.params:
        params_dict = json.loads(args.params)
    elif args.params_file:
        with open(args.params_file) as f:
            params_dict = json.load(f)
    else:
        print("Error: provide --params or --params-file", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(args.data, parse_dates=["timestamp"])
    regime = classify_regime(df, version=args.regime_version, min_duration=args.regime_min_duration)
    params = DecisionParams.from_dict(params_dict)
    result = run_backtest(df, params, regime, initial_capital=args.capital)

    trades_list = []
    for t in result.trades:
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

    rd = result.to_dict()
    rd["total_return"] = float(result.total_return)
    rd["sharpe_ratio"] = float(result.sharpe_ratio)
    rd["max_drawdown"] = float(result.max_drawdown)
    rd["win_rate"] = float(result.win_rate)
    rd["profit_factor"] = float(result.profit_factor)

    output = {
        "backtest_result": rd,
        "equity_curve": result.equity_curve.tolist(),
        "trades": trades_list,
    }

    if args.output:
        with open(args.output, "w") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        print(f"Result saved to {args.output}")
    else:
        print(json.dumps(output, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
