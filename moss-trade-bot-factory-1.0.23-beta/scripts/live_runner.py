#!/usr/bin/env python3
"""Auto-running live trading bot. Executes trading decisions every N minutes.

Usage:
    # Run with credentials file + bot params:
    python live_runner.py --creds ~/.moss-trade-bot/agent_creds.json --params-file bot_params.json --interval 15
    # --platform-url should be site origin only, e.g. https://beta-api.moss.site

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
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
from core.decision import DecisionParams, compute_signals
from core.regime import classify_regime
from core.indicators import atr as compute_atr
from core.fetcher import fetch_live_ohlcv
from trading_client import TradingClient

RUNNING = True
PLATFORM_URL_HELP = "Platform site origin only, e.g. https://beta-api.moss.site. The client appends API paths automatically."


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


def _to_float(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def compute_current_signal(df: pd.DataFrame, params: DecisionParams) -> int:
    """Compute signal for the latest bar using full history for indicator warmup."""
    regime = classify_regime(df, version="v1", min_duration=0)
    signals = compute_signals(df, params, regime)
    return int(signals.iloc[-1])


def check_exit_conditions(
    position: dict, params: DecisionParams, df: pd.DataFrame, mark_price: float
) -> Optional[str]:
    """Check if current position should be closed. Returns exit reason or None."""
    if mark_price <= 0:
        return None

    entry_price = _to_float(position.get("entry_price"), 0.0)
    if entry_price <= 0:
        return None
    leverage = _to_float(position.get("leverage"), 1.0)
    side = str(position.get("position_side") or position.get("side") or "").upper()
    if side not in ("LONG", "SHORT", "BUY", "SELL"):
        return None

    if side in ("LONG", "BUY"):
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
              cycle_num: int, data_source: str, symbol: str = "BTC/USDT",
              log_file: str = None) -> dict:
    """Run one trading decision cycle. Returns cycle result dict."""

    _log(f"Cycle #{cycle_num}: fetching {data_source} market data...", log_file)
    try:
        df = fetch_live_ohlcv(
            symbol,
            timeframe,
            days=14,
            data_source=data_source,
            use_cache=False,
        )
    except Exception as e:
        _log(f"Data fetch failed: {e}", log_file)
        return {"action": "error", "detail": str(e)}

    price_data = client.get_price()
    if isinstance(price_data, dict) and price_data.get("code"):
        _log(f"Price query failed: {price_data}", log_file)
        return {"action": "error", "detail": price_data}
    mark_price = _to_float(price_data.get("mark_price"), 0.0)
    if mark_price <= 0:
        _log(f"Price unavailable for {symbol}: {price_data}", log_file)
        return {"action": "wait", "reason": "price_unavailable"}

    positions = client.get_positions()
    account = client.get_account()
    if isinstance(account, dict) and account.get("code"):
        _log(f"Account query failed: {account}", log_file)
        return {"action": "error", "detail": account}
    if isinstance(positions, dict):
        _log(f"Positions query failed: {positions}", log_file)
        return {"action": "error", "detail": positions}
    if not isinstance(positions, list):
        positions = []
    free_margin = _to_float(account.get("free_margin", account.get("available_equity")), 0.0)
    wallet_balance = _to_float(account.get("wallet_balance", account.get("account_value")), 0.0)

    base_asset = symbol.split("/")[0] if "/" in symbol else symbol.rstrip("USDT") or symbol
    _log(f"  {base_asset}=${mark_price:,.2f} | balance=${wallet_balance:,.2f} | "
         f"free=${free_margin:,.2f} | positions={len(positions)}", log_file)

    if positions:
        pos = positions[0]
        exit_reason = check_exit_conditions(pos, params, df, mark_price)
        pos_side = str(pos.get("position_side") or pos.get("side") or "").upper()
        pos_qty = pos.get("qty", pos.get("net_qty", "0"))
        pos_entry = pos.get("entry_price", "?")
        if exit_reason:
            _log(f"  EXIT: {exit_reason} for {pos_side} @ entry={pos_entry}", log_file)
            close_order_id = f"auto-close-{cycle_num}-{int(time.time())}"
            result = client.close_position(pos_side, "", close_order_id, "")
            _log(f"  Closed: pnl={result.get('realized_pnl', '?')}", log_file)
            return {"action": "close", "reason": exit_reason, "result": result}
        else:
            _log(f"  HOLD: {pos_side} qty={pos_qty} unrealized={pos.get('unrealized_pnl','0')}", log_file)
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
        result = client.open_long(f"{notional:.2f}", leverage, order_id, "")
    else:
        result = client.open_short(f"{notional:.2f}", leverage, order_id, "")

    if "order_id" in result:
        _log(f"  FILLED: price={result.get('fill_price', '?')} qty={result.get('fill_qty', '?')}", log_file)
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
        choices=["hyperliquid"],
        default="hyperliquid",
        help="Market data source for signal generation (Hyperliquid perpetuals).",
    )
    parser.add_argument("--symbol", default="BTC/USDT", help="Trading pair, e.g. BTC/USDT or ETH/USDT (default: BTC/USDT)")
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
        symbol=args.symbol,
    )

    signal.signal(signal.SIGINT, _handle_stop)
    signal.signal(signal.SIGTERM, _handle_stop)

    log_file = args.log or None
    interval_sec = args.interval * 60

    _log(
        f"Bot started: symbol={args.symbol} interval={args.interval}m timeframe={timeframe} "
        f"leverage={params.base_leverage}x data_source={args.data_source}",
        log_file,
    )
    _log(f"  bot_id={creds.get('bot_id','?')} long_bias={params.long_bias}", log_file)

    cycle = 0
    while RUNNING:
        cycle += 1
        try:
            run_cycle(client, params, timeframe, cycle, args.data_source,
                      symbol=args.symbol, log_file=log_file)
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
