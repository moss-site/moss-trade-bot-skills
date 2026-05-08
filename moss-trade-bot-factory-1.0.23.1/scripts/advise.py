#!/usr/bin/env python3
"""One-shot decision advise CLI.

Replaces the long-running advisor runner. Every invocation:
    1. Pulls fresh market data + server-side positions/account
    2. Computes signal / regime / exit conditions
    3. Emits a single advice JSON to stdout (or --out path)
    4. Exits

The host LLM is expected to call this each decision cycle (cron / loop /
manual), read the advice, keep or enrich the bilingual reasoning draft, and
dispatch live_trade.py — all in the same context, no clock drift, no advice
expiry window.

Usage:
    python3 advise.py \
        --creds ~/.moss-trade-bot/agent_creds.json \
        --params-file /tmp/bot_params.json \
        --symbol BTC/USDT \
        --interval 5

    # write to file instead of stdout
    python3 advise.py ... --out /tmp/advice.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timedelta, timezone
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
from core.decision import DecisionParams, compute_signals
from core.fetcher import fetch_live_ohlcv
from core.indicators import atr as compute_atr
from core.regime import classify_regime
from trading_client import TradingClient


ADVICE_SCHEMA_VERSION = 1
SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
LIVE_TRADE_PATH = os.path.join(SCRIPTS_DIR, "live_trade.py")
_REASONING_TODO_ZH = "<TODO: 由载体 LLM 基于 advice.context 写的中文决策说明>"
_REASONING_TODO_EN = "<TODO: matching English version, no Chinese characters>"


# ---------------------------------------------------------------------------
# Helpers (lifted from the retired live_runner)
# ---------------------------------------------------------------------------

def _to_float(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _iso(ts: datetime) -> str:
    return ts.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


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


def _recent_change_pct(df: pd.DataFrame, bars: int) -> Optional[float]:
    if len(df) <= bars:
        return None
    start = _to_float(df["close"].iloc[-bars - 1], 0.0)
    end = _to_float(df["close"].iloc[-1], 0.0)
    if start <= 0:
        return None
    return (end - start) / start


def _latest_regime(df: pd.DataFrame) -> str:
    try:
        regime = classify_regime(df, version="v1")
        if len(regime) > 0:
            return str(regime.iloc[-1])
    except Exception:
        pass
    return "UNKNOWN"


def _compute_signal(df: pd.DataFrame, params: DecisionParams) -> int:
    regime = classify_regime(df, version="v1")
    signals = compute_signals(df, params, regime)
    return int(signals.iloc[-1])


def _check_exit_conditions(
    position: dict, params: DecisionParams, df: pd.DataFrame, mark_price: float
) -> Optional[str]:
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

    signal = _compute_signal(df, params)
    if side == "LONG" and signal == -1:
        return "signal_reverse"
    if side == "SHORT" and signal == 1:
        return "signal_reverse"
    return None


def _serialize_position(position):
    if not position:
        return None
    side = str(position.get("position_side") or position.get("side") or "").upper()
    return {
        "side": side or None,
        "qty": str(position.get("qty") or position.get("net_qty") or "0"),
        "entry_price": _to_float(position.get("entry_price"), 0.0),
        "leverage": _to_float(position.get("leverage"), 1.0),
        "unrealized_pnl": _to_float(position.get("unrealized_pnl"), 0.0),
    }


def _params_snapshot(params: DecisionParams) -> dict:
    return {
        "long_bias":          params.long_bias,
        "base_leverage":      params.base_leverage,
        "max_leverage":       params.max_leverage,
        "risk_per_trade":     params.risk_per_trade,
        "max_position_pct":   params.max_position_pct,
        "sl_atr_mult":        params.sl_atr_mult,
        "tp_rr_ratio":        params.tp_rr_ratio,
        "entry_threshold":    params.entry_threshold,
        "exit_threshold":     params.exit_threshold,
        "regime_sensitivity": params.regime_sensitivity,
    }


def _fmt_money(value) -> str:
    return f"{_to_float(value, 0.0):.2f}"


def _fmt_price(value) -> str:
    raw = _to_float(value, 0.0)
    if raw <= 0:
        return "unknown"
    return f"{raw:.2f}"


def _fmt_ratio(value) -> str:
    return f"{_to_float(value, 0.0) * 100:.2f}%"


def _fmt_multiplier(value) -> str:
    return f"{_to_float(value, 0.0):.2f}x"


def _change_phrase_zh(change_24h_pct: Optional[float]) -> str:
    if change_24h_pct is None:
        return "暂无完整涨跌幅数据"
    pct = abs(_to_float(change_24h_pct, 0.0)) * 100
    if pct < 0.05:
        return f"基本持平{pct:.2f}%"
    if change_24h_pct > 0:
        return f"上涨{pct:.2f}%"
    return f"下跌{pct:.2f}%"


def _change_phrase_en(change_24h_pct: Optional[float]) -> str:
    if change_24h_pct is None:
        return "no complete 24h change data"
    pct = abs(_to_float(change_24h_pct, 0.0)) * 100
    if pct < 0.05:
        return f"flat 24h move of {pct:.2f}%"
    if change_24h_pct > 0:
        return f"24h gain of {pct:.2f}%"
    return f"24h decline of {pct:.2f}%"


def _direction_zh(direction: Optional[str]) -> str:
    return "开空" if str(direction or "").upper() == "SHORT" else "开多"


def _direction_en(direction: Optional[str]) -> str:
    return "short" if str(direction or "").upper() == "SHORT" else "long"


def _position_side_zh(position: Optional[dict]) -> str:
    side = str((position or {}).get("side") or "").upper()
    return "空头" if side == "SHORT" else "多头"


def _position_side_en(position: Optional[dict]) -> str:
    side = str((position or {}).get("side") or "").upper()
    return "short" if side == "SHORT" else "long"


def _exit_reason_zh(exit_reason: Optional[str]) -> str:
    return {
        "stop_loss": "触发止损",
        "take_profit": "达到止盈目标",
        "signal_reverse": "信号反转",
    }.get(str(exit_reason or ""), str(exit_reason or "风控退出"))


def _exit_reason_en(exit_reason: Optional[str]) -> str:
    return {
        "stop_loss": "stop loss",
        "take_profit": "take profit",
        "signal_reverse": "signal reversal",
    }.get(str(exit_reason or ""), str(exit_reason or "risk exit"))


def _build_reasoning_draft(
    *,
    action: str,
    direction: Optional[str],
    exit_reason: Optional[str],
    symbol: str,
    timeframe: str,
    mark_price: float,
    change_24h_pct: Optional[float],
    regime: str,
    signal_value: int,
    free_margin: float,
    position: Optional[dict],
    params: DecisionParams,
    suggestion,
) -> Optional[dict]:
    if action not in ("open", "close"):
        return None

    price = _fmt_price(mark_price)
    change_zh = _change_phrase_zh(change_24h_pct)
    change_en = _change_phrase_en(change_24h_pct)
    risk = _fmt_ratio(params.risk_per_trade)
    max_pct = _fmt_ratio(params.max_position_pct)
    sl = _fmt_multiplier(params.sl_atr_mult)
    tp = f"{_to_float(params.tp_rr_ratio, 0.0):.2f}R"

    if action == "open":
        leverage = int((suggestion or {}).get("leverage") or min(params.base_leverage, params.max_leverage) or 1)
        notional = _fmt_money((suggestion or {}).get("notional_usdt") or 0)
        zh = (
            f"本轮按{timeframe}周期评估{symbol}，标记价{price}，24小时{change_zh}，"
            f"当前市场状态为{regime}，signal_value={int(signal_value)}，方向指向{_direction_zh(direction)}。"
            f"账户可用保证金约{_fmt_money(free_margin)} USDT，策略按 risk_per_trade={risk}、"
            f"max_position_pct={max_pct} 控制敞口，建议使用{leverage}x、名义金额{notional} USDT。"
            f"执行后以 ATR 止损倍数{sl}和盈亏比{tp}管理风险；若下一周期信号反转、"
            "趋势失效或触及止损/止盈，优先减仓或平仓，避免在波动里扩大亏损。"
        )
        en = (
            f"This {timeframe} review for {symbol} uses mark {price}, a {change_en}, "
            f"regime {regime}, and signal_value={int(signal_value)}. These inputs support "
            f"opening a {_direction_en(direction)} position now. Available margin is about "
            f"{_fmt_money(free_margin)} USDT, so sizing follows risk_per_trade={risk} "
            f"and max_position_pct={max_pct}, with suggested {leverage}x leverage and "
            f"{notional} USDT notional. Risk is managed with ATR stop {sl} and reward/risk "
            f"{tp}; if the next cycle shows reversal, regime failure, or stop/take-profit "
            "pressure, exposure should be reduced or closed."
        )
    else:
        pos = position or {}
        zh = (
            f"本轮按{timeframe}周期复核{symbol}现有{_position_side_zh(pos)}仓位，标记价{price}，"
            f"入场价{_fmt_price(pos.get('entry_price'))}，持仓数量{pos.get('qty') or '0'}，"
            f"未实现盈亏约{_fmt_money(pos.get('unrealized_pnl'))} USDT。24小时{change_zh}，"
            f"市场状态为{regime}，本次退出原因为{_exit_reason_zh(exit_reason)}。"
            "当前优先释放风险敞口并锁定或止住本轮结果，避免信号反转、止损压力或趋势失效继续扩大损失；"
            "平仓后下一周期再根据最新 signal_value、趋势结构和可用保证金重新评估是否开新仓。"
        )
        en = (
            f"This {timeframe} review for {symbol} checks the existing {_position_side_en(pos)} "
            f"position at mark {price}, entry {_fmt_price(pos.get('entry_price'))}, quantity "
            f"{pos.get('qty') or '0'}, and unrealized PnL about {_fmt_money(pos.get('unrealized_pnl'))} USDT. "
            f"The 24h context is {change_en}, regime is {regime}, and the exit trigger is "
            f"{_exit_reason_en(exit_reason)}. Closing now prioritizes releasing risk and preserving "
            "the current result before reversal or stop pressure widens. After the close, the next "
            "cycle should reassess signal_value, trend structure, and available margin before any new entry."
        )

    zh, en = TradingClient._require_bilingual_reasoning(zh, en)
    return {"zh": zh, "en": en}


def _build_dispatch_command(
    *,
    action: str,
    direction: Optional[str],
    creds_path: str,
    symbol: str,
    notional_usdt: Optional[str] = None,
    leverage: Optional[int] = None,
    position_side: Optional[str] = None,
    platform_url: str = "",
    reasoning_zh: str = "",
    reasoning_en: str = "",
):
    if action == "open":
        sub = "open-long" if direction == "LONG" else "open-short"
        cmd = [
            "python3", LIVE_TRADE_PATH, sub,
            "--creds", creds_path,
            "--symbol", symbol,
            "--amount", str(notional_usdt or ""),
            "--leverage", str(int(leverage or 1)),
        ]
    elif action == "close":
        cmd = [
            "python3", LIVE_TRADE_PATH, "close",
            "--creds", creds_path,
            "--symbol", symbol,
            "--side", (position_side or "").upper(),
        ]
    else:
        return None
    if platform_url:
        cmd.extend(["--platform-url", platform_url])
    cmd.extend([
        "--reasoning-zh", reasoning_zh or _REASONING_TODO_ZH,
        "--reasoning-en", reasoning_en or _REASONING_TODO_EN,
    ])
    return cmd


def build_advice(
    *,
    client: TradingClient,
    params: DecisionParams,
    symbol: str,
    timeframe: str,
    data_source: str,
    interval_min: int,
    creds_path: str,
    platform_url: str = "",
) -> dict:
    """Run one decision cycle and return an advice dict (no IO)."""
    df = fetch_live_ohlcv(symbol, timeframe, days=14, data_source=data_source, use_cache=False)

    price_data = client.get_price()
    mark_price = _to_float(price_data.get("mark_price") if isinstance(price_data, dict) else 0.0, 0.0)

    positions = client.get_positions()
    if not isinstance(positions, list):
        positions = []
    account = client.get_account() or {}
    free_margin = _to_float(account.get("free_margin", account.get("available_equity")), 0.0)
    wallet_balance = _to_float(account.get("wallet_balance", account.get("account_value")), 0.0)

    regime = _latest_regime(df)
    change_24h_pct = _recent_change_pct(df, _bars_for_24h(timeframe))

    now = datetime.now(timezone.utc)
    valid_until = now + timedelta(minutes=int(interval_min))
    cycle = int(now.timestamp())

    pos = positions[0] if positions else None
    if pos:
        exit_reason = _check_exit_conditions(pos, params, df, mark_price)
        if exit_reason:
            position_side = str(pos.get("position_side") or pos.get("side") or "").upper() or None
            return _wrap(
                cycle=cycle, now=now, valid_until=valid_until,
                symbol=symbol, timeframe=timeframe, data_source=data_source,
                action="close", direction=None, exit_reason=exit_reason,
                mark_price=mark_price, change_24h_pct=change_24h_pct,
                regime=regime, signal_value=0,
                free_margin=free_margin, wallet_balance=wallet_balance,
                position=pos, params=params, suggestion=None,
                creds_path=creds_path, platform_url=platform_url,
                position_side=position_side,
            )
        return _wrap(
            cycle=cycle, now=now, valid_until=valid_until,
            symbol=symbol, timeframe=timeframe, data_source=data_source,
            action="hold", direction=None, exit_reason=None,
            mark_price=mark_price, change_24h_pct=change_24h_pct,
            regime=regime, signal_value=0,
            free_margin=free_margin, wallet_balance=wallet_balance,
            position=pos, params=params, suggestion=None,
            creds_path=creds_path, platform_url=platform_url,
            position_side=None,
        )

    signal_value = _compute_signal(df, params)
    if signal_value == 0:
        return _wrap(
            cycle=cycle, now=now, valid_until=valid_until,
            symbol=symbol, timeframe=timeframe, data_source=data_source,
            action="wait", direction=None, exit_reason=None,
            mark_price=mark_price, change_24h_pct=change_24h_pct,
            regime=regime, signal_value=0,
            free_margin=free_margin, wallet_balance=wallet_balance,
            position=None, params=params, suggestion=None,
            creds_path=creds_path, platform_url=platform_url,
            position_side=None,
        )

    direction = "LONG" if signal_value == 1 else "SHORT"
    leverage = int(min(params.base_leverage, params.max_leverage))
    notional = free_margin * params.risk_per_trade * leverage
    notional = min(notional, free_margin * params.max_position_pct * leverage)
    notional = max(notional, 10)
    suggestion = {
        "leverage": int(leverage),
        "notional_usdt": f"{notional:.2f}",
        "client_order_id_prefix": f"advise-{cycle}",
    }
    return _wrap(
        cycle=cycle, now=now, valid_until=valid_until,
        symbol=symbol, timeframe=timeframe, data_source=data_source,
        action="open", direction=direction, exit_reason=None,
        mark_price=mark_price, change_24h_pct=change_24h_pct,
        regime=regime, signal_value=int(signal_value),
        free_margin=free_margin, wallet_balance=wallet_balance,
        position=None, params=params, suggestion=suggestion,
        creds_path=creds_path, platform_url=platform_url,
        position_side=None,
    )


def _wrap(
    *, cycle, now, valid_until,
    symbol, timeframe, data_source,
    action, direction, exit_reason,
    mark_price, change_24h_pct, regime, signal_value,
    free_margin, wallet_balance, position,
    params, suggestion, creds_path, platform_url, position_side,
) -> dict:
    serialized_position = _serialize_position(position)
    reasoning_draft = _build_reasoning_draft(
        action=action,
        direction=direction,
        exit_reason=exit_reason,
        symbol=symbol,
        timeframe=timeframe,
        mark_price=mark_price,
        change_24h_pct=change_24h_pct,
        regime=regime,
        signal_value=signal_value,
        free_margin=free_margin,
        position=serialized_position,
        params=params,
        suggestion=suggestion,
    )
    dispatch = _build_dispatch_command(
        action=action,
        direction=direction,
        creds_path=creds_path,
        symbol=symbol,
        notional_usdt=(suggestion or {}).get("notional_usdt"),
        leverage=(suggestion or {}).get("leverage"),
        position_side=position_side,
        platform_url=platform_url,
        reasoning_zh=(reasoning_draft or {}).get("zh", ""),
        reasoning_en=(reasoning_draft or {}).get("en", ""),
    )
    return {
        "version":     ADVICE_SCHEMA_VERSION,
        "cycle":       int(cycle),
        "issued_at":   _iso(now),
        "valid_until": _iso(valid_until),
        "symbol":      symbol,
        "timeframe":   timeframe,
        "data_source": data_source,
        "action":      action,
        "direction":   direction,
        "exit_reason": exit_reason,
        "context": {
            "mark_price":     float(mark_price),
            "change_24h_pct": change_24h_pct,
            "regime":         regime,
            "signal_value":   int(signal_value),
            "free_margin":    float(free_margin),
            "wallet_balance": float(wallet_balance),
            "position":       serialized_position,
        },
        "params_snapshot":  _params_snapshot(params),
        "suggestion":       suggestion,
        "reasoning_draft":  reasoning_draft,
        "dispatch_command": dispatch,
    }


def _build_advice_payload(
    *,
    cycle: int,
    interval_min: int,
    symbol: str,
    timeframe: str,
    data_source: str,
    action: str,
    direction: Optional[str],
    exit_reason: Optional[str],
    mark_price: float,
    change_24h_pct: Optional[float],
    regime: str,
    signal_value: int,
    free_margin: float,
    wallet_balance: float,
    position,
    params: DecisionParams,
    suggestion,
    creds_path: str,
    platform_url: str,
) -> dict:
    """Test-friendly adapter that derives now/valid_until from interval_min and delegates to _wrap."""
    now = datetime.now(timezone.utc)
    valid_until = now + timedelta(minutes=int(interval_min))
    position_side = None
    if action == "close" and position:
        position_side = str(position.get("position_side") or position.get("side") or "").upper() or None
    return _wrap(
        cycle=cycle, now=now, valid_until=valid_until,
        symbol=symbol, timeframe=timeframe, data_source=data_source,
        action=action, direction=direction, exit_reason=exit_reason,
        mark_price=mark_price, change_24h_pct=change_24h_pct,
        regime=regime, signal_value=signal_value,
        free_margin=free_margin, wallet_balance=wallet_balance,
        position=position, params=params, suggestion=suggestion,
        creds_path=creds_path, platform_url=platform_url,
        position_side=position_side,
    )


def _atomic_write(path: str, payload: dict) -> None:
    tmp = f"{path}.tmp"
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)


def main():
    parser = argparse.ArgumentParser(description="One-shot decision advise CLI.")
    parser.add_argument("--creds", required=True, help="Agent credentials JSON path")
    parser.add_argument("--params", default=None, help="Bot params as JSON string")
    parser.add_argument("--params-file", default=None, help="Bot params JSON file")
    parser.add_argument("--symbol", default="BTC/USDT", help="Trading pair (default: BTC/USDT)")
    parser.add_argument("--timeframe", default=None, help="K-line timeframe (default: derived from --interval)")
    parser.add_argument("--interval", type=int, default=15,
                        help="Decision cadence in minutes; sets advice valid_until window (default 15)")
    parser.add_argument("--data-source", choices=["hyperliquid"], default="hyperliquid")
    parser.add_argument("--platform-url", default="", help="Platform site origin; otherwise uses creds.base_url")
    parser.add_argument("--out", default="", help="Optional path to write the advice JSON; default stdout")
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
        sys.exit(2)
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

    advice = build_advice(
        client=client,
        params=params,
        symbol=args.symbol,
        timeframe=timeframe,
        data_source=args.data_source,
        interval_min=int(args.interval),
        creds_path=os.path.abspath(args.creds),
        platform_url=args.platform_url,
    )

    if args.out:
        _atomic_write(args.out, advice)
        print(f"Advice written to {args.out}", file=sys.stderr)
    else:
        json.dump(advice, sys.stdout, indent=2, ensure_ascii=False)
        sys.stdout.write("\n")


if __name__ == "__main__":
    main()
