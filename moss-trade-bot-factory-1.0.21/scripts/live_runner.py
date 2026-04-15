#!/usr/bin/env python3
"""Auto-running live trading bot. Executes trading decisions every N minutes.

Usage:
    # Run with credentials file + bot params:
    python live_runner.py --creds ~/.moss-trade-bot/agent_creds.json --params-file bot_params.json --interval 15
    # --platform-url should be site origin only, e.g. https://ai.moss.site

    # US users can switch live signal input to Coinbase spot candles:
    python live_runner.py --creds ~/.moss-trade-bot/agent_creds.json --params-file bot_params.json --interval 15 --data-source coinbase

    # With evolution (reflect every N cycles):
    python live_runner.py --creds creds.json --params-file params.json --interval 15 --evolve-every 96
"""

import argparse
import json
import sys
import os
import time
import signal
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
from core.decision import DecisionParams, compute_signals
from core.regime import classify_regime
from core.indicators import atr as compute_atr
from core.fetcher import fetch_live_ohlcv
from trading_client import TradingClient

RUNNING = True
PLATFORM_URL_HELP = "Platform site origin only, e.g. https://ai.moss.site. The client appends API paths automatically."


def _handle_stop(signum, frame):
    global RUNNING
    print(f"\n[{_now()}] Received stop signal, finishing current cycle...", file=sys.stderr)
    RUNNING = False


def _now():
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def _log(msg, log_file=None):
    line = f"[{_now()}] {msg}"
    print(line, file=sys.stderr)
    if log_file:
        with open(log_file, "a") as f:
            f.write(line + "\n")


def compute_current_signal(df: pd.DataFrame, params: DecisionParams) -> int:
    """Compute signal for the latest bar using full history for indicator warmup."""
    regime = classify_regime(df, version="v1", min_duration=0)
    signals = compute_signals(df, params, regime)
    return int(signals.iloc[-1])


def check_exit_conditions(
    client: TradingClient, position: dict, params: DecisionParams, df: pd.DataFrame
) -> str | None:
    """Check if current position should be closed. Returns exit reason or None."""
    price_data = client.get_price()
    mark_price = float(price_data.get("mark_price", 0))
    if mark_price <= 0:
        return None

    entry_price = float(position["entry_price"])
    leverage = position["leverage"]
    side = position["position_side"]

    if side == "LONG":
        pnl_pct = (mark_price - entry_price) / entry_price * leverage
    else:
        pnl_pct = (entry_price - mark_price) / entry_price * leverage

    atr_series = compute_atr(df, 14)
    atr_val = atr_series.iloc[-1] if not np.isnan(atr_series.iloc[-1]) else mark_price * 0.02
    sl_dist_pct = params.sl_atr_mult * atr_val / entry_price
    tp_dist_pct = sl_dist_pct * params.tp_rr_ratio

    if pnl_pct <= -sl_dist_pct * leverage:
        return "stop_loss"

    if pnl_pct >= tp_dist_pct * leverage:
        return "take_profit"

    signal = compute_current_signal(df, params)
    if side == "LONG" and signal == -1:
        return "signal_reverse"
    if side == "SHORT" and signal == 1:
        return "signal_reverse"

    return None


def run_cycle(client: TradingClient, params: DecisionParams, timeframe: str,
              cycle_num: int, data_source: str, log_file: str = None) -> dict:
    """Run one trading decision cycle. Returns cycle result dict."""

    _log(f"Cycle #{cycle_num}: fetching {data_source} market data...", log_file)
    try:
        df = fetch_live_ohlcv(
            "BTC/USDT",
            timeframe,
            days=14,
            data_source=data_source,
            use_cache=False,
        )
    except Exception as e:
        _log(f"Data fetch failed: {e}", log_file)
        return {"action": "error", "detail": str(e)}

    price_data = client.get_price()
    mark_price = float(price_data.get("mark_price", 0))

    positions = client.get_positions()
    account = client.get_account()
    free_margin = float(account.get("free_margin", 0))

    _log(f"  BTC=${mark_price:,.2f} | balance=${float(account.get('wallet_balance',0)):,.2f} | "
         f"free=${free_margin:,.2f} | positions={len(positions)}", log_file)

    if positions:
        pos = positions[0]
        exit_reason = check_exit_conditions(client, pos, params, df)
        if exit_reason:
            _log(f"  EXIT: {exit_reason} for {pos['position_side']} @ entry={pos['entry_price']}", log_file)
            result = client.close_position(pos["position_side"])
            _log(f"  Closed: pnl={result.get('realized_pnl', '?')}", log_file)
            return {"action": "close", "reason": exit_reason, "result": result}
        else:
            _log(f"  HOLD: {pos['position_side']} qty={pos['qty']} unrealized={pos.get('unrealized_pnl','0')}", log_file)
            return {"action": "hold"}

    signal = compute_current_signal(df, params)

    if signal == 0:
        _log(f"  NO SIGNAL: waiting", log_file)
        return {"action": "wait"}

    direction = "LONG" if signal == 1 else "SHORT"
    leverage = int(min(params.base_leverage, params.max_leverage))
    notional = free_margin * params.risk_per_trade * leverage
    notional = min(notional, free_margin * params.max_position_pct * leverage)
    notional = max(notional, 10)

    order_id = f"auto-{cycle_num}-{int(time.time())}"
    _log(f"  OPEN {direction}: ${notional:,.0f} @ {leverage}x (order_id={order_id})", log_file)

    if direction == "LONG":
        result = client.open_long(f"{notional:.2f}", leverage, order_id)
    else:
        result = client.open_short(f"{notional:.2f}", leverage, order_id)

    if "order_id" in result:
        _log(f"  FILLED: price={result['fill_price']} qty={result['fill_qty']}", log_file)
    else:
        _log(f"  ORDER FAILED: {result}", log_file)

    return {"action": "open", "direction": direction, "result": result}


def main():
    parser = argparse.ArgumentParser(description="Live trading bot runner")
    parser.add_argument("--creds", required=True, help="Agent credentials JSON")
    parser.add_argument("--params", default=None, help="Bot params JSON string")
    parser.add_argument("--params-file", default=None, help="Bot params JSON file")
    parser.add_argument("--interval", type=int, default=15, help="Decision interval in minutes")
    parser.add_argument("--timeframe", default=None, help="Override K-line timeframe (default: from interval)")
    parser.add_argument(
        "--data-source",
        choices=["binanceusdm", "coinbase"],
        default="binanceusdm",
        help="Live-only market data source for signal generation. "
             "Coinbase is allowed only in live mode; backtests must still use Binance/CSV.",
    )
    parser.add_argument("--max-cycles", type=int, default=0, help="Stop after N cycles (0=unlimited)")
    parser.add_argument("--log", default="", help="Log file path")
    parser.add_argument("--platform-url", default="", help=PLATFORM_URL_HELP + " Otherwise reuse base_url from creds file.")
    args = parser.parse_args()

    with open(args.creds) as f:
        creds = json.load(f)

    if args.params:
        params_dict = json.loads(args.params)
    elif args.params_file:
        with open(args.params_file) as f:
            params_dict = json.load(f)
    else:
        print("Error: --params or --params-file required", file=sys.stderr)
        sys.exit(1)

    params = DecisionParams.from_dict(params_dict)

    if args.timeframe:
        timeframe = args.timeframe
    else:
        tf_map = {1: "1m", 5: "5m", 15: "15m", 60: "1h", 240: "4h"}
        timeframe = tf_map.get(args.interval, f"{args.interval}m")

    client = TradingClient(
        api_key=creds["api_key"],
        api_secret=creds["api_secret"],
        base_url=args.platform_url or creds.get("base_url", ""),
        bot_id=creds.get("bot_id", ""),
    )

    signal.signal(signal.SIGINT, _handle_stop)
    signal.signal(signal.SIGTERM, _handle_stop)

    log_file = args.log or None
    interval_sec = args.interval * 60

    _log(
        f"Bot started: interval={args.interval}m timeframe={timeframe} "
        f"leverage={params.base_leverage}x data_source={args.data_source}",
        log_file,
    )
    _log(f"  bot_id={creds.get('bot_id','?')} long_bias={params.long_bias}", log_file)

    cycle = 0
    while RUNNING:
        cycle += 1
        try:
            run_cycle(client, params, timeframe, cycle, args.data_source, log_file)
        except Exception as e:
            _log(f"Cycle #{cycle} error: {e}", log_file)

        if args.max_cycles > 0 and cycle >= args.max_cycles:
            _log(f"Reached max cycles ({args.max_cycles}), stopping.", log_file)
            break

        if RUNNING:
            _log(f"Next cycle in {args.interval}m...", log_file)
            for _ in range(interval_sec):
                if not RUNNING:
                    break
                time.sleep(1)

    _log("Bot stopped.", log_file)


if __name__ == "__main__":
    main()
