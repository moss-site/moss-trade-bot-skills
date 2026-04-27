#!/usr/bin/env python3
"""Auto-running live trading bot. Executes trading decisions every N minutes.

Usage:
    # Run with credentials file + bot params:
    python live_runner.py --creds ~/.moss-trade-bot/agent_creds.json --params-file bot_params.json --interval 15
    # --platform-url should be site origin only, e.g. https://ai.moss.site

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


def _to_float(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def compute_current_signal(df: pd.DataFrame, params: DecisionParams) -> int:
    """Compute signal for the latest bar using full history for indicator warmup."""
    regime = classify_regime(df, version="v1")
    signals = compute_signals(df, params, regime)
    return int(signals.iloc[-1])


def _latest_regime(df: pd.DataFrame) -> str:
    try:
        regime = classify_regime(df, version="v1")
        if len(regime) > 0:
            return str(regime.iloc[-1])
    except Exception:
        pass
    return "UNKNOWN"


def _recent_change_pct(df: pd.DataFrame, bars: int) -> Optional[float]:
    if len(df) <= bars:
        return None
    start = _to_float(df["close"].iloc[-bars - 1], 0.0)
    end = _to_float(df["close"].iloc[-1], 0.0)
    if start <= 0:
        return None
    return (end - start) / start


def _bars_for_24h(timeframe: str) -> int:
    raw = str(timeframe or "").strip().lower()
    try:
        if raw.endswith("m"):
            minutes = max(1, int(raw[:-1]))
            return max(1, int(round(24 * 60 / minutes)))
        if raw.endswith("h"):
            hours = max(1, int(raw[:-1]))
            return max(1, int(round(24 / hours)))
    except ValueError:
        pass
    return 96


def _position_pnl_pct(position: dict, mark_price: float) -> Optional[float]:
    entry_price = _to_float(position.get("entry_price"), 0.0)
    if entry_price <= 0 or mark_price <= 0:
        return None
    leverage = _to_float(position.get("leverage"), 1.0)
    side = str(position.get("position_side") or position.get("side") or "").upper()
    if side in ("LONG", "BUY"):
        return (mark_price - entry_price) / entry_price * leverage
    if side in ("SHORT", "SELL"):
        return (entry_price - mark_price) / entry_price * leverage
    return None


def _pct_text(value: Optional[float]) -> str:
    if value is None:
        return "未知"
    return f"{value * 100:+.2f}%"


def _pct_text_en(value: Optional[float]) -> str:
    if value is None:
        return "unknown"
    return f"{value * 100:+.2f}%"


def build_open_reasoning_pair(
    *,
    direction: str,
    symbol: str,
    timeframe: str,
    data_source: str,
    df: pd.DataFrame,
    params: DecisionParams,
    mark_price: float,
    free_margin: float,
    notional: float,
    leverage: int,
) -> dict:
    side_text = "开多" if direction == "LONG" else "开空"
    signal_text = "多头" if direction == "LONG" else "空头"
    regime = _latest_regime(df)
    change_24h = _recent_change_pct(df, _bars_for_24h(timeframe))
    en_side_text = "open a long position" if direction == "LONG" else "open a short position"
    en_signal_text = "bullish" if direction == "LONG" else "bearish"
    return {
        "zh": (
            f"自动交易在 {data_source} {symbol} {timeframe} K线检测到{signal_text}信号，"
            f"当前标记价格约 {mark_price:.2f}，近24小时涨跌为 {_pct_text(change_24h)}，regime={regime}。"
            f"本次按 risk_per_trade={params.risk_per_trade:.2f}、可用保证金约 {free_margin:.2f} 控制仓位，"
            f"以 {leverage}x、名义金额约 {notional:.2f} {side_text}；后续由止损、止盈或反向信号管理退出。"
        ),
        "en": (
            f"The auto runner detected a {en_signal_text} signal on {data_source} {symbol} {timeframe} candles. "
            f"The current mark price is about {mark_price:.2f}, the 24h change is {_pct_text_en(change_24h)}, and regime={regime}. "
            f"Position size is controlled with risk_per_trade={params.risk_per_trade:.2f} and free margin around {free_margin:.2f}; "
            f"it will {en_side_text} with {leverage}x leverage and about {notional:.2f} notional. Exits remain managed by stop loss, take profit, or a reverse signal."
        ),
    }


def build_close_reasoning_pair(
    *,
    position: dict,
    exit_reason: str,
    symbol: str,
    timeframe: str,
    df: pd.DataFrame,
    mark_price: float,
) -> dict:
    pos_side = str(position.get("position_side") or position.get("side") or "").upper()
    entry_price = position.get("entry_price", "?")
    pnl_pct = _position_pnl_pct(position, mark_price)
    regime = _latest_regime(df)
    reason_text = {
        "stop_loss": "触发止损条件，优先控制回撤和杠杆风险",
        "take_profit": "触发止盈条件，优先锁定已实现收益",
        "signal_reverse": "当前信号与持仓方向相反，按反向信号退出",
    }.get(exit_reason, f"触发退出条件 {exit_reason}")
    en_reason_text = {
        "stop_loss": "The stop-loss condition was triggered, so drawdown and leverage risk take priority.",
        "take_profit": "The take-profit condition was triggered, so the runner is locking in realized gains.",
        "signal_reverse": "The current signal is opposite to the held position, so the runner exits on reversal.",
    }.get(exit_reason, f"The exit condition {exit_reason} was triggered.")
    return {
        "zh": (
            f"自动交易对 {symbol} {timeframe} 的 {pos_side} 仓位执行平仓："
            f"入场价 {entry_price}，当前标记价格约 {mark_price:.2f}，杠杆后浮动收益约 {_pct_text(pnl_pct)}，regime={regime}。"
            f"{reason_text}，因此通过 reduce-only 市价单退出该仓位。"
        ),
        "en": (
            f"The auto runner is closing the {pos_side} position on {symbol} {timeframe}: "
            f"entry price {entry_price}, current mark price about {mark_price:.2f}, leveraged unrealized return about {_pct_text_en(pnl_pct)}, and regime={regime}. "
            f"{en_reason_text} The position is exited with a reduce-only market order."
        ),
    }


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
            reasoning_pair = build_close_reasoning_pair(
                position=pos,
                exit_reason=exit_reason,
                symbol=symbol,
                timeframe=timeframe,
                df=df,
                mark_price=mark_price,
            )
            reasoning = reasoning_pair["zh"]
            reasoning_en = reasoning_pair["en"]
            result = client.close_position(pos_side, "", close_order_id, reasoning, reasoning_en)
            _log(f"  Closed: pnl={result.get('realized_pnl', '?')}", log_file)
            return {"action": "close", "reason": exit_reason, "reasoning": reasoning, "reasoning_en": reasoning_en, "result": result}
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
    reasoning_pair = build_open_reasoning_pair(
        direction=direction,
        symbol=symbol,
        timeframe=timeframe,
        data_source=data_source,
        df=df,
        params=params,
        mark_price=mark_price,
        free_margin=free_margin,
        notional=notional,
        leverage=leverage,
    )
    reasoning = reasoning_pair["zh"]
    reasoning_en = reasoning_pair["en"]

    if direction == "LONG":
        result = client.open_long(f"{notional:.2f}", leverage, order_id, reasoning, reasoning_en)
    else:
        result = client.open_short(f"{notional:.2f}", leverage, order_id, reasoning, reasoning_en)

    if "order_id" in result:
        _log(f"  FILLED: price={result.get('fill_price', '?')} qty={result.get('fill_qty', '?')}", log_file)
    else:
        _log(f"  ORDER FAILED: {result}", log_file)

    return {"action": "open", "direction": direction, "reasoning": reasoning, "reasoning_en": reasoning_en, "result": result}


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
