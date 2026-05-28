#!/usr/bin/env python3
"""
Ambush bot 事件级本地回测器。

输入：
  /tmp/ambush_params.json      — 双通道参数（来自 propose.py）
  data_cache/ambush/events.csv     — 216 历史异动事件元数据
  data_cache/ambush/features.csv   — 19 项预算特征（balanced 规则用 RSI/surge/chg）
  data_cache/ambush/klines/<base>.csv — 87 个币种触发窗 K 线

输出（JSON）：
  summary       — 总触发数 / 胜率 / 总收益 / 最大回撤 / Sharpe
  per_direction — long / short / skip 各自的笔数 / 胜率 / 收益 / skip 原因细分
  trades        — 每笔模拟交易的明细（可用 --dump-trades N 打印最差/最好 N 笔）

调用：
  python3 backtest.py --params /tmp/ambush_params.json \
                      --output /tmp/ambush_backtest_result.json \
                      [--dump-trades N] [--data-dir <path>]
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass, asdict
from decimal import Decimal, getcontext
from pathlib import Path
from typing import Optional

import pandas as pd

getcontext().prec = 28

# ===== 路径 =====
SCRIPTS_DIR = Path(__file__).resolve().parent.parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from ambush import features, indicators  # noqa: E402


def _default_data_dir() -> Path:
    """Resolve the ambush data dir, hydrating from GitHub Release on first run.

    See ``core.data_cache_archive.resolve_data_root()`` for the precedence
    rules (env override → in-skill dev cache → user cache hydrate).
    """
    from core.data_cache_archive import resolve_data_root  # noqa: E402 — lazy
    return resolve_data_root() / "ambush"


# Back-compat alias for any caller that imported the module-level constant.
# Evaluated lazily on first attribute access via __getattr__ below.
def __getattr__(name: str):  # PEP 562
    if name == "DEFAULT_DATA_DIR":
        return _default_data_dir()
    raise AttributeError(name)


def _D(v) -> Decimal:
    if isinstance(v, Decimal):
        return v
    if isinstance(v, float):
        return Decimal(str(v))
    return Decimal(v)


# ===== 决策规则（与 internal/ambush/strategy/rule_decision.go 对齐）=====

def balanced_decide(surge_15m: float, rsi_14: float, chg_before_24h: float) -> tuple[str, str]:
    """与 BalancedDecideV0 同口径的 5 条规则。"""
    s, r, c = surge_15m, rsi_14, chg_before_24h
    # SHORT 信号
    if s > 0.20 and c > 20:
        return ("short", "rule_short_compound_overstretch")
    if c > 100 and r > 70:
        return ("short", "rule_short_extreme_pullback")
    if s > 0.25:
        return ("short", "rule_short_spike_extreme")
    # LONG 信号
    if 0.10 < s < 0.15 and r < 60 and c < 10:
        return ("long", "rule_long_momentum_init")
    if 30 <= c <= 80 and r > 75:
        return ("long", "rule_long_momentum_extend")
    return ("skip", "rule_no_match")


def decide_for_event(event: dict, params: dict) -> tuple[str, str]:
    """
    阈值过滤 + 规则判向 + direction 方向过滤。
    返回 (decision, reason)。
    """
    direction = params["direction"]
    trig = params["trigger"]

    # 阈值过滤
    if event["oi_mc"] is not None and event["oi_mc"] < trig["oi_mc_threshold"]:
        return ("skip", "below_oi_mc_threshold")
    if event["surge_15m"] < trig["surge_15m_threshold"]:
        return ("skip", "below_surge_threshold")
    # z_score 阈值在历史回测里 features.csv 没存（detector 端运行时算）；
    # 这里 best-effort 跳过 — 如果未来 features.csv 含 z_score，加上检查即可
    # if event.get("z_score") is not None and event["z_score"] < trig["z_score_threshold"]:
    #     return ("skip", "below_z_score_threshold")

    signal, reason = balanced_decide(
        event["surge_15m"],
        event.get("rsi_14") or 50,
        event.get("change_before_24h_pct") or 0,
    )
    if direction == "balanced" or signal == "skip" or signal == direction:
        return (signal, reason)
    if direction in ("short", "long"):
        return ("skip", "direction_mismatch")
    return ("skip", "invalid_direction")


def decide_for_features(
    surge_15m: float, rsi_14: float, chg_before_24h: float, direction: str
) -> tuple[str, str]:
    """Rule decision with only the configured direction filter, no trigger gate."""
    signal, reason = balanced_decide(surge_15m, rsi_14, chg_before_24h)
    if direction == "balanced" or signal == "skip" or signal == direction:
        return (signal, reason)
    if direction in ("short", "long"):
        return ("skip", "direction_mismatch")
    return ("skip", "invalid_direction")


# ===== 仓位演化 =====

AMBUSH_BACKTEST_INITIAL_CAPITAL = _D("1000")
AMBUSH_BACKTEST_TAKER_FEE_RATE = _D("0.00045")
AMBUSH_BACKTEST_FUNDING_RATE = _D("0.0000125")
AMBUSH_SHARED_DEPTH_MID_PRICE = _D("80876.5")

AMBUSH_SHARED_BID_DEPTH = [
    ("80876.0", "12.19958"),
    ("80875.0", "0.63913"),
    ("80874.0", "1.49278"),
    ("80872.0", "1.8746"),
    ("80871.0", "2.22314"),
    ("80870.0", "0.23029"),
    ("80869.0", "1.39707"),
    ("80868.0", "8.50507"),
    ("80867.0", "1.69148"),
    ("80866.0", "0.12391"),
    ("80865.0", "0.35311"),
    ("80864.0", "5.81773"),
    ("80863.0", "4.03869"),
    ("80862.0", "3.18947"),
    ("80861.0", "3.54351"),
    ("80860.0", "5.79071"),
    ("80859.0", "4.4586"),
    ("80858.0", "2.22688"),
    ("80857.0", "4.35442"),
    ("80856.0", "14.17119"),
]

AMBUSH_SHARED_ASK_DEPTH = [
    ("80877.0", "5.49057"),
    ("80878.0", "3.00013"),
    ("80879.0", "0.61845"),
    ("80880.0", "0.37237"),
    ("80881.0", "0.01794"),
    ("80882.0", "1.27627"),
    ("80883.0", "0.0518"),
    ("80884.0", "0.32118"),
    ("80885.0", "0.49772"),
    ("80886.0", "0.21498"),
    ("80887.0", "2.11463"),
    ("80888.0", "2.73305"),
    ("80889.0", "0.395"),
    ("80890.0", "0.06"),
    ("80891.0", "2.23133"),
    ("80892.0", "6.02488"),
    ("80893.0", "3.0444"),
    ("80894.0", "7.79161"),
    ("80895.0", "0.52808"),
    ("80896.0", "2.4172"),
]


def _anchored_depth_price(snapshot_price: str, mark_price: Decimal) -> Decimal:
    return mark_price * _D(snapshot_price) / AMBUSH_SHARED_DEPTH_MID_PRICE


def _anchored_depth_qty(snapshot_qty: str, mark_price: Decimal) -> Decimal:
    if mark_price <= 0:
        return _D(0)
    return _D(snapshot_qty) * AMBUSH_SHARED_DEPTH_MID_PRICE / mark_price


def _simulate_shared_depth_fill_by_notional(
    direction: int, target_notional: Decimal, mark_price: Decimal
) -> tuple[Decimal, Decimal, Decimal]:
    if target_notional <= 0 or mark_price <= 0:
        return _D(0), _D(0), _D(0)
    levels = AMBUSH_SHARED_ASK_DEPTH if direction > 0 else AMBUSH_SHARED_BID_DEPTH
    remaining = target_notional
    fill_qty = _D(0)
    fill_notional = _D(0)
    for snapshot_price, snapshot_qty in levels:
        if remaining <= 0:
            break
        price = _anchored_depth_price(snapshot_price, mark_price)
        qty = _anchored_depth_qty(snapshot_qty, mark_price)
        level_notional = price * qty
        take_notional = min(remaining, level_notional)
        if price <= 0 or take_notional <= 0:
            continue
        fill_qty += take_notional / price
        fill_notional += take_notional
        remaining -= take_notional
    if fill_qty <= 0:
        return _D(0), _D(0), _D(0)
    return fill_notional / fill_qty, fill_qty, fill_notional


def _simulate_shared_depth_fill_by_qty(
    direction: int, requested_qty: Decimal, mark_price: Decimal
) -> tuple[Decimal, Decimal, Decimal]:
    if requested_qty <= 0 or mark_price <= 0:
        return _D(0), _D(0), _D(0)
    levels = AMBUSH_SHARED_ASK_DEPTH if direction > 0 else AMBUSH_SHARED_BID_DEPTH
    remaining = requested_qty
    fill_qty = _D(0)
    fill_notional = _D(0)
    for snapshot_price, snapshot_qty in levels:
        if remaining <= 0:
            break
        price = _anchored_depth_price(snapshot_price, mark_price)
        qty = _anchored_depth_qty(snapshot_qty, mark_price)
        take_qty = min(remaining, qty)
        if price <= 0 or take_qty <= 0:
            continue
        fill_qty += take_qty
        fill_notional += take_qty * price
        remaining -= take_qty
    if fill_qty <= 0:
        return _D(0), _D(0), _D(0)
    return fill_notional / fill_qty, fill_qty, fill_notional


def _estimate_entry_execution(
    side: str, raw_entry: float, position_pct: float, leverage: int
) -> dict[str, Decimal]:
    mark = _D(raw_entry)
    margin = _D(position_pct) * AMBUSH_BACKTEST_INITIAL_CAPITAL
    lev = _D(leverage)
    if mark <= 0 or margin <= 0 or lev <= 0:
        return {"price": _D(0), "qty": _D(0), "notional": _D(0), "margin": margin, "filled_ratio": _D(0)}
    target_notional = margin * lev
    direction = -1 if side == "short" else 1
    price, qty, notional = _simulate_shared_depth_fill_by_notional(direction, target_notional, mark)
    if price <= 0 or qty <= 0:
        price = mark
        qty = target_notional / mark
        notional = target_notional
    filled_ratio = notional / target_notional if target_notional > 0 else _D(0)
    return {
        "price": price,
        "qty": qty,
        "notional": notional,
        "margin": margin,
        "filled_ratio": filled_ratio,
    }


def _next_funding_settlement(ts: pd.Timestamp) -> pd.Timestamp:
    cursor = pd.Timestamp(ts).floor("h")
    if cursor <= ts:
        cursor += pd.Timedelta(hours=1)
    return cursor


def _funding_mark_at(bars: list[dict], settlement: pd.Timestamp, fallback: Decimal) -> Decimal:
    mark = fallback
    for bar in bars:
        bar_ts = pd.Timestamp(bar["ts"])
        if bar_ts > settlement:
            break
        close = float(bar["close"])
        if close > 0:
            mark = _D(close)
    return mark


def _estimate_funding_fee(
    side: str,
    qty: Decimal,
    entry_time: pd.Timestamp,
    exit_time: pd.Timestamp,
    bars: list[dict],
    fallback_mark: Decimal,
) -> Decimal:
    if qty <= 0 or exit_time <= entry_time:
        return _D(0)
    signed_qty = -qty if side == "short" else qty
    total = _D(0)
    settlement = _next_funding_settlement(entry_time)
    while settlement <= exit_time:
        mark = _funding_mark_at(bars, settlement, fallback_mark)
        if mark > 0:
            total += -signed_qty * mark * AMBUSH_BACKTEST_FUNDING_RATE
        settlement += pd.Timedelta(hours=1)
    return total


def _apply_trade_costs(
    side: str,
    raw_entry: float,
    raw_exit: float,
    entry_time: pd.Timestamp,
    exit_time: pd.Timestamp,
    position_pct: float,
    leverage: int,
    bars: list[dict],
) -> dict[str, float]:
    raw_entry_d = _D(raw_entry)
    raw_exit_d = _D(raw_exit)
    entry_exec = _estimate_entry_execution(side, raw_entry, position_pct, leverage)
    if raw_entry_d <= 0 or raw_exit_d <= 0 or entry_exec["margin"] <= 0 or entry_exec["qty"] <= 0:
        raw = _signed_return(side, raw_entry, raw_exit) * leverage
        return {
            "entry_price": raw_entry,
            "exit_price": raw_exit,
            "raw_pnl_pct": raw,
            "gross_pnl_pct": raw,
            "pnl_pct": raw,
            "depth_cost_pct": 0.0,
            "trading_fee_pct": 0.0,
            "funding_fee_pct": 0.0,
            "entry_fee_pct": 0.0,
            "exit_fee_pct": 0.0,
            "depth_filled_ratio": 1.0,
        }

    close_direction = 1 if side == "short" else -1
    exit_price, exit_qty, exit_notional = _simulate_shared_depth_fill_by_qty(
        close_direction, entry_exec["qty"], raw_exit_d
    )
    if exit_price <= 0 or exit_qty <= 0:
        exit_price = raw_exit_d
        exit_qty = entry_exec["qty"]
        exit_notional = entry_exec["qty"] * raw_exit_d

    raw_pnl = _D(_signed_return(side, raw_entry, raw_exit) * leverage)
    if side == "short":
        gross_abs = (entry_exec["price"] - exit_price) * exit_qty
    else:
        gross_abs = (exit_price - entry_exec["price"]) * exit_qty
    gross_pnl = gross_abs / entry_exec["margin"]

    entry_fee = -(entry_exec["notional"] * AMBUSH_BACKTEST_TAKER_FEE_RATE)
    exit_fee = -(exit_notional * AMBUSH_BACKTEST_TAKER_FEE_RATE)
    trading_fee = entry_fee + exit_fee
    funding_fee = _estimate_funding_fee(
        side, entry_exec["qty"], entry_time, exit_time, bars, raw_entry_d
    )
    funding_pct = funding_fee / entry_exec["margin"]
    entry_fee_pct = entry_fee / entry_exec["margin"]
    exit_fee_pct = exit_fee / entry_exec["margin"]
    trading_fee_pct = trading_fee / entry_exec["margin"]
    net_pnl = gross_pnl + trading_fee_pct + funding_pct

    filled_ratio = min(entry_exec["filled_ratio"], _D(1))
    if entry_exec["qty"] > 0:
        filled_ratio = min(filled_ratio, exit_qty / entry_exec["qty"])

    return {
        "entry_price": float(entry_exec["price"]),
        "exit_price": float(exit_price),
        "raw_pnl_pct": float(raw_pnl),
        "gross_pnl_pct": float(gross_pnl),
        "pnl_pct": float(net_pnl),
        "depth_cost_pct": float(gross_pnl - raw_pnl),
        "trading_fee_pct": float(trading_fee_pct),
        "funding_fee_pct": float(funding_pct),
        "entry_fee_pct": float(entry_fee_pct),
        "exit_fee_pct": float(exit_fee_pct),
        "depth_filled_ratio": float(filled_ratio),
    }


@dataclass
class TradeOutcome:
    side: str  # "long" / "short"
    entry: float
    exit_price: float
    pnl_pct: float           # 杠杆后净 PnL（小数：0.50 = +50%）
    exit_reason: str         # 'atr_stop_loss' / 'trailing' / 'signal_reverse' / 'max_hold'
    bars_held: int
    entry_mark: float = 0.0
    exit_mark: float = 0.0
    raw_pnl_pct: float = 0.0
    gross_pnl_pct: float = 0.0
    depth_cost_pct: float = 0.0
    trading_fee_pct: float = 0.0
    funding_fee_pct: float = 0.0
    entry_fee_pct: float = 0.0
    exit_fee_pct: float = 0.0
    depth_filled_ratio: float = 1.0
    entry_time: str = ""
    exit_time: str = ""


def _sl_atr_mult_or_default(v, side: str) -> float:
    val = float(v or 0)
    if val > 0:
        return val
    return 2.5 if side == "short" else 2.0


def _signed_return(side: str, entry: float, exit_price: float) -> float:
    if entry <= 0:
        return 0.0
    if side == "short":
        return (entry - exit_price) / entry
    return (exit_price - entry) / entry


def simulate_position(
    klines_after: list[dict],
    raw_entry: float,
    entry_time: pd.Timestamp,
    side: str,
    leverage: int,
    position_pct: float,
    trailing_pct: float,
    max_hold_hours: int,
    max_hold_bars: int,
    sl_atr_mult: float,
    direction: str,
) -> TradeOutcome:
    """
    用 backend/live executor 同口径 close cascade 模拟一次仓位演化。

    平仓优先级（每根 K 线内）:
      1. ATR stop_loss（按 K 线 close 判断）
      2. max_hold
      3. K-line-close trailing
      4. signal_reverse

    pnl_pct 是杠杆后的 net % return。
    """
    bars = klines_after
    if not bars:
        return TradeOutcome(side, raw_entry, raw_entry, 0.0, "no_klines", 0)

    entry_exec = _estimate_entry_execution(side, raw_entry, position_pct, leverage)
    entry = float(entry_exec["price"]) if entry_exec["price"] > 0 else raw_entry

    peak = 0.0
    have_peak = False
    limit = len(bars)
    if max_hold_bars > 0 and limit > max_hold_bars + 1:
        limit = max_hold_bars + 1

    for i in range(limit):
        lo = i - 95
        if lo < 0:
            lo = 0
        window = bars[lo : i + 1]
        if len(window) < 15:
            continue
        last_close = float(bars[i]["close"])
        opened_ago_h = i * 0.25

        # 1. ATR stop_loss
        try:
            atr = indicators.compute_atr(window, period=14)
        except ValueError:
            atr = 0.0
        if atr > 0 and entry > 0:
            sl_dist = sl_atr_mult * atr / entry
            raw_pnl = _signed_return(side, entry, last_close)
            if raw_pnl <= -sl_dist:
                cost = _apply_trade_costs(
                    side, raw_entry, last_close, entry_time, pd.Timestamp(bars[i]["ts"]),
                    position_pct, leverage, bars[: i + 1],
                )
                return TradeOutcome(
                    side, cost["entry_price"], cost["exit_price"], cost["pnl_pct"],
                    "atr_stop_loss", i + 1, raw_entry, last_close,
                    cost["raw_pnl_pct"], cost["gross_pnl_pct"], cost["depth_cost_pct"],
                    cost["trading_fee_pct"], cost["funding_fee_pct"],
                    cost["entry_fee_pct"], cost["exit_fee_pct"], cost["depth_filled_ratio"],
                    str(pd.Timestamp(entry_time)), str(pd.Timestamp(bars[i]["ts"])),
                )

        # 2. max_hold
        if max_hold_hours > 0 and opened_ago_h >= max_hold_hours:
            cost = _apply_trade_costs(
                side, raw_entry, last_close, entry_time, pd.Timestamp(bars[i]["ts"]),
                position_pct, leverage, bars[: i + 1],
            )
            return TradeOutcome(
                side, cost["entry_price"], cost["exit_price"], cost["pnl_pct"],
                "max_hold", i + 1, raw_entry, last_close,
                cost["raw_pnl_pct"], cost["gross_pnl_pct"], cost["depth_cost_pct"],
                cost["trading_fee_pct"], cost["funding_fee_pct"],
                cost["entry_fee_pct"], cost["exit_fee_pct"], cost["depth_filled_ratio"],
                str(pd.Timestamp(entry_time)), str(pd.Timestamp(bars[i]["ts"])),
            )

        # 3. K-line-close trailing
        if not have_peak:
            peak = last_close
            have_peak = True
            continue
        new_peak = peak
        if side == "long":
            if last_close > peak or not (peak > 0):
                new_peak = last_close
            drift = (new_peak - last_close) / new_peak if new_peak > 0 else 0.0
        else:
            if (last_close < peak and peak > 0) or not (peak > 0):
                new_peak = last_close
            drift = (last_close - new_peak) / new_peak if new_peak > 0 else 0.0
        peak = new_peak
        if trailing_pct > 0 and drift >= trailing_pct:
            cost = _apply_trade_costs(
                side, raw_entry, last_close, entry_time, pd.Timestamp(bars[i]["ts"]),
                position_pct, leverage, bars[: i + 1],
            )
            return TradeOutcome(
                side, cost["entry_price"], cost["exit_price"], cost["pnl_pct"],
                "trailing", i + 1, raw_entry, last_close,
                cost["raw_pnl_pct"], cost["gross_pnl_pct"], cost["depth_cost_pct"],
                cost["trading_fee_pct"], cost["funding_fee_pct"],
                cost["entry_fee_pct"], cost["exit_fee_pct"], cost["depth_filled_ratio"],
                str(pd.Timestamp(entry_time)), str(pd.Timestamp(bars[i]["ts"])),
            )

        # 4. signal_reverse
        try:
            fresh = features.compute_features(window)
            reverse, _ = decide_for_features(
                fresh["surge_15m"], fresh["rsi_14"], fresh["chg_24h_pct"], direction
            )
        except ValueError:
            reverse = "skip"
        if (side == "long" and reverse == "short") or (
            side == "short" and reverse == "long"
        ):
            cost = _apply_trade_costs(
                side, raw_entry, last_close, entry_time, pd.Timestamp(bars[i]["ts"]),
                position_pct, leverage, bars[: i + 1],
            )
            return TradeOutcome(
                side, cost["entry_price"], cost["exit_price"], cost["pnl_pct"],
                "signal_reverse", i + 1, raw_entry, last_close,
                cost["raw_pnl_pct"], cost["gross_pnl_pct"], cost["depth_cost_pct"],
                cost["trading_fee_pct"], cost["funding_fee_pct"],
                cost["entry_fee_pct"], cost["exit_fee_pct"], cost["depth_filled_ratio"],
                str(pd.Timestamp(entry_time)), str(pd.Timestamp(bars[i]["ts"])),
            )

    last_close = float(bars[-1]["close"])
    cost = _apply_trade_costs(
        side, raw_entry, last_close, entry_time, pd.Timestamp(bars[-1]["ts"]),
        position_pct, leverage, bars,
    )
    return TradeOutcome(
        side, cost["entry_price"], cost["exit_price"], cost["pnl_pct"],
        "max_hold", len(bars), raw_entry, last_close,
        cost["raw_pnl_pct"], cost["gross_pnl_pct"], cost["depth_cost_pct"],
        cost["trading_fee_pct"], cost["funding_fee_pct"],
        cost["entry_fee_pct"], cost["exit_fee_pct"], cost["depth_filled_ratio"],
        str(pd.Timestamp(entry_time)), str(pd.Timestamp(bars[-1]["ts"])),
    )


# ===== 加载数据 =====

def load_event_dataset(data_dir: Path) -> pd.DataFrame:
    """events.csv + features.csv merge by (symbol, trigger_date)。"""
    events = pd.read_csv(data_dir / "events.csv")
    features = pd.read_csv(data_dir / "features.csv")
    merged = events.merge(
        features[["symbol", "trigger_date", "rsi_14", "label", "label_long", "label_short"]],
        on=["symbol", "trigger_date"], how="left",
    )
    merged["base"] = merged["symbol"].str.split("/").str[0]
    return merged


def load_klines_for_base(data_dir: Path, base: str) -> Optional[pd.DataFrame]:
    p = data_dir / "klines" / f"{base}.csv"
    if not p.exists():
        return None
    df = pd.read_csv(p)
    df["ts"] = pd.to_datetime(df["timestamp"], utc=True)
    if df["ts"].dt.tz is not None:
        df["ts"] = df["ts"].dt.tz_convert(None)
    return df


def klines_after_trigger(klines: pd.DataFrame, trigger_ts: pd.Timestamp, max_bars: int) -> list[dict]:
    after = klines[klines["ts"] > trigger_ts].head(max_bars)
    return after[["ts", "open", "high", "low", "close"]].to_dict("records")


# ===== 主回测循环 =====

def run_backtest(
    events_df: pd.DataFrame, params: dict, data_dir: Path,
) -> dict:
    direction = params["direction"]
    long_p = params["long_params"]
    short_p = params["short_params"]

    # 单事件仓位演化最长 max_hold_bars = max(long.max_hold, short.max_hold) × 4 (15m bars/h)
    max_hold_bars = max(long_p["max_hold_hours"], short_p["max_hold_hours"]) * 4

    events_df = events_df.copy()
    events_df["_trigger_ts_sort"] = pd.to_datetime(events_df["trigger_ts"], utc=True)
    events_df = events_df.sort_values(["_trigger_ts_sort", "base"], kind="mergesort")

    decisions = []
    trades = []
    skip_reasons: dict[str, int] = {}

    klines_cache: dict[str, pd.DataFrame] = {}

    for _, ev_row in events_df.iterrows():
        ev = {
            "symbol":                  ev_row["symbol"],
            "base":                    ev_row["base"],
            "trigger_ts":              pd.to_datetime(ev_row["trigger_ts"], utc=True).tz_convert(None),
            "trigger_price":           float(ev_row["price_at_trigger"]),
            "oi_mc":                   float(ev_row["oi_mc"]),
            "surge_15m":               float(ev_row["surge_15m"]),
            "rsi_14":                  ev_row.get("rsi_14"),
            "change_before_24h_pct":   float(ev_row["change_before_24h_pct"]),
            "actual_24h":              float(ev_row["return_24h_pct"]),
        }
        if pd.notna(ev["rsi_14"]):
            ev["rsi_14"] = float(ev["rsi_14"])
        else:
            ev["rsi_14"] = 50.0

        decision, reason = decide_for_event(ev, params)
        decisions.append({"base": ev["base"], "decision": decision, "reason": reason})

        if decision == "skip":
            skip_reasons[reason] = skip_reasons.get(reason, 0) + 1
            continue

        # 演化仓位
        if ev["base"] not in klines_cache:
            klines_cache[ev["base"]] = load_klines_for_base(data_dir, ev["base"])
        klines = klines_cache[ev["base"]]
        if klines is None or klines.empty:
            skip_reasons["missing_klines"] = skip_reasons.get("missing_klines", 0) + 1
            continue

        # entry_delay_bars (short) / momentum_bars (long) — both shift entry
        # by N × 15min after the trigger to let initial squeeze tail run /
        # confirm momentum. Skill production-side already implements this
        # via daemon-thread deferred opens (event_handler._run_deferred_open).
        # Backtest now mirrors it: fetch max_hold + delay bars, then enter
        # at bars_after[delay].open instead of bars_after[0] price.
        if decision == "long":
            delay_bars = int(long_p.get("momentum_bars", 0) or 0)
            tr = float(long_p["trailing_pct"])
            max_hold_hours = int(long_p["max_hold_hours"])
            sl_atr_mult = _sl_atr_mult_or_default(long_p.get("sl_atr_mult", 0), "long")
            lev = int(long_p["leverage"])
            position_pct = float(long_p["position_pct"])
        else:
            delay_bars = int(short_p.get("entry_delay_bars", 0) or 0)
            tr = float(short_p["trailing_pct"])
            max_hold_hours = int(short_p["max_hold_hours"])
            sl_atr_mult = _sl_atr_mult_or_default(short_p.get("sl_atr_mult", 0), "short")
            lev = int(short_p["leverage"])
            position_pct = float(short_p["position_pct"])

        bars_after = klines_after_trigger(klines, ev["trigger_ts"], max_hold_bars + delay_bars + 1)
        if not bars_after:
            skip_reasons["no_klines_after_trigger"] = skip_reasons.get("no_klines_after_trigger", 0) + 1
            continue
        if len(bars_after) <= delay_bars:
            skip_reasons["insufficient_klines_after_delay"] = skip_reasons.get("insufficient_klines_after_delay", 0) + 1
            continue

        # Shift entry by delay_bars: use the OPEN of bars_after[delay_bars]
        # as new entry price (mirrors skill behavior — order fires at market
        # after delay completes).
        delayed_bars = bars_after[delay_bars:]
        entry_price = delayed_bars[0]["open"] if delay_bars > 0 else ev["trigger_price"]
        entry_time = pd.Timestamp(delayed_bars[0]["ts"]) if delay_bars > 0 else ev["trigger_ts"]
        outcome = simulate_position(
            delayed_bars,
            entry_price,
            entry_time,
            decision,
            lev,
            position_pct,
            tr,
            max_hold_hours,
            max_hold_bars,
            sl_atr_mult,
            direction,
        )
        leverage = lev
        trades.append({
            "base": ev["base"],
            "symbol": ev["symbol"],
            "trigger_date": ev_row["trigger_date"],
            "decision": decision,
            "reason": reason,
            "position_pct": position_pct,
            "leverage": leverage,
            **asdict(outcome),
        })

    out = _aggregate(decisions, trades, skip_reasons)
    out["trades"] = trades
    return out


def _aggregate(decisions: list, trades: list, skip_reasons: dict) -> dict:
    n_total = len(decisions)
    n_triggered = len(trades)
    triggered_pct = n_triggered / n_total if n_total else 0.0

    if not trades:
        return {
            "summary": {
                "total_events": n_total,
                "triggered_count": 0,
                "triggered_pct": 0.0,
                "win_count": 0, "win_rate": 0.0,
                "total_return_pct": 0.0,
                "avg_pnl_pct": 0.0,
                "gross_return_pct": 0.0,
                "depth_cost_pct": 0.0,
                "trading_fee_pct": 0.0,
                "funding_fee_pct": 0.0,
                "entry_fee_pct": 0.0,
                "exit_fee_pct": 0.0,
                "avg_gross_pnl_pct": 0.0,
                "avg_cost_pnl_pct": 0.0,
                "max_drawdown_pct": 0.0,
                "sharpe": 0.0,
            },
            "per_direction": {
                "skip": {"count": n_total, "reason_breakdown": skip_reasons},
            },
        }

    # pnl_pct 是 simulate_position 返回的"保证金相对收益率"（杠杆后）。
    # 实际每笔只押 position_pct × equity 作保证金，所以：
    #   net_return_per_trade = position_pct × pnl_pct
    #   equity_after          = equity × (1 + position_pct × pnl_pct)
    #
    # 之前公式 `equity *= 1 + pnl_pct` 假设每笔押 100% equity 当保证金 — 严重高估
    # 回撤 + 高估收益。修复后回撤反映真实资金动态。
    net_pnl_list = [t["pnl_pct"] * t.get("position_pct", 1.0) for t in trades]
    gross_pnl_list = [t.get("gross_pnl_pct", 0.0) * t.get("position_pct", 1.0) for t in trades]
    depth_cost_list = [t.get("depth_cost_pct", 0.0) * t.get("position_pct", 1.0) for t in trades]
    trading_fee_list = [t.get("trading_fee_pct", 0.0) * t.get("position_pct", 1.0) for t in trades]
    funding_fee_list = [t.get("funding_fee_pct", 0.0) * t.get("position_pct", 1.0) for t in trades]
    entry_fee_list = [t.get("entry_fee_pct", 0.0) * t.get("position_pct", 1.0) for t in trades]
    exit_fee_list = [t.get("exit_fee_pct", 0.0) * t.get("position_pct", 1.0) for t in trades]
    wins = [p for p in net_pnl_list if p > 0]
    total_ret = sum(net_pnl_list) * 100
    gross_ret = sum(gross_pnl_list) * 100
    depth_cost = sum(depth_cost_list) * 100
    trading_fee = sum(trading_fee_list) * 100
    funding_fee = sum(funding_fee_list) * 100
    entry_fee = sum(entry_fee_list) * 100
    exit_fee = sum(exit_fee_list) * 100
    avg_pnl = total_ret / len(net_pnl_list)
    avg_gross_pnl = gross_ret / len(net_pnl_list)
    avg_cost_pnl = (depth_cost + trading_fee + funding_fee) / len(net_pnl_list)
    win_rate = len(wins) / len(net_pnl_list)
    sd = pd.Series(net_pnl_list).std(ddof=1) if len(net_pnl_list) > 1 else 0
    sharpe = (sum(net_pnl_list) / len(net_pnl_list)) / sd if sd > 0 else 0.0

    # 最大回撤（按 trade 累计权益曲线，net 后口径）
    equity = 1.0
    peak = 1.0
    max_dd = 0.0
    for p in net_pnl_list:
        equity *= 1 + p
        peak = max(peak, equity)
        dd = (equity - peak) / peak
        max_dd = min(max_dd, dd)

    summary = {
        "total_events":     n_total,
        "triggered_count":  n_triggered,
        "triggered_pct":    round(triggered_pct, 4),
        "win_count":        len(wins),
        "win_rate":         round(win_rate, 4),
        "total_return_pct": round(total_ret, 2),
        "avg_pnl_pct":      round(avg_pnl, 2),
        "gross_return_pct": round(gross_ret, 2),
        "depth_cost_pct":   round(depth_cost, 2),
        "trading_fee_pct":  round(trading_fee, 2),
        "funding_fee_pct":  round(funding_fee, 2),
        "entry_fee_pct":    round(entry_fee, 2),
        "exit_fee_pct":     round(exit_fee, 2),
        "avg_gross_pnl_pct": round(avg_gross_pnl, 2),
        "avg_cost_pnl_pct": round(avg_cost_pnl, 2),
        "max_drawdown_pct": round(max_dd * 100, 2),
        "sharpe":           round(sharpe, 4),
    }

    by_side: dict[str, list] = {"long": [], "short": []}
    for t in trades:
        by_side.setdefault(t["decision"], []).append(t)

    per_direction = {}
    for side, ts in by_side.items():
        if not ts:
            continue
        ps = [t["pnl_pct"] * t.get("position_pct", 1.0) for t in ts]
        ws = [p for p in ps if p > 0]
        per_direction[side] = {
            "count": len(ts),
            "win_rate": round(len(ws) / len(ps), 4),
            "total_return_pct": round(sum(ps) * 100, 2),
            "avg_pnl_pct": round(sum(ps) / len(ps) * 100, 2),
        }
    per_direction["skip"] = {
        "count": n_total - n_triggered,
        "reason_breakdown": skip_reasons,
    }

    return {"summary": summary, "per_direction": per_direction}


# ===== main =====

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--params", required=True, help="ambush bot 参数 JSON 路径")
    p.add_argument("--output", required=True, help="回测结果输出 JSON 路径")
    p.add_argument("--data-dir", default=None,
                   help="data_cache/ambush 路径；省略时自动定位(MOSS_TRADE_BOT_DATA_DIR / 本地 dev cache / GitHub Release 自动 hydrate)")
    p.add_argument("--dump-trades", type=int, default=0, metavar="N",
                   help="额外打印最差 N 笔 + 最好 N 笔交易明细（用于查 outlier）。"
                        "trades 列表本身始终在 result JSON 里")
    args = p.parse_args()

    data_dir = Path(args.data_dir) if args.data_dir else _default_data_dir()
    if not data_dir.exists():
        print(f"[backtest] ERROR: data dir {data_dir} not found", file=sys.stderr)
        sys.exit(1)

    params = json.loads(Path(args.params).read_text())

    print(f"[backtest] loading data from {data_dir} ...")
    events_df = load_event_dataset(data_dir)
    print(f"[backtest] {len(events_df)} events loaded")

    print(f"[backtest] running event-level backtest ...")
    result = run_backtest(events_df, params, data_dir)

    Path(args.output).write_text(json.dumps(result, indent=2, ensure_ascii=False))

    s = result["summary"]
    print(f"\n=== 总结 ===")
    print(f"触发: {s['triggered_count']}/{s['total_events']} ({s['triggered_pct']*100:.0f}%)")
    print(f"胜率: {s['win_rate']*100:.1f}%  总收益: {s['total_return_pct']:+.1f}%  "
          f"avg/笔: {s['avg_pnl_pct']:+.2f}%")
    print(f"成本: depth {s.get('depth_cost_pct', 0):+.2f}%  "
          f"fee {s.get('trading_fee_pct', 0):+.2f}%  "
          f"funding {s.get('funding_fee_pct', 0):+.2f}%")
    print(f"最大回撤: {s['max_drawdown_pct']:.1f}%  Sharpe: {s['sharpe']:.3f}")

    if args.dump_trades > 0 and result.get("trades"):
        N = args.dump_trades
        trades_sorted = sorted(result["trades"], key=lambda t: t["pnl_pct"])
        print(f"\n=== 最差 {N} 笔（outlier 排查）===")
        print(f"{'symbol':<18}{'trigger_date':<14}{'side':<7}{'rule':<32}"
              f"{'lev':>5}{'pnl_pct':>10}{'exit_reason':>15}{'bars_held':>10}")
        for t in trades_sorted[:N]:
            print(f"{t['symbol']:<18}{t['trigger_date']:<14}{t['decision']:<7}"
                  f"{t['reason']:<32}{t.get('leverage',0):>5}"
                  f"{t['pnl_pct']*100:>+9.1f}%{t.get('exit_reason','?'):>15}"
                  f"{t.get('bars_held','?'):>10}")
        print(f"\n=== 最好 {N} 笔 ===")
        for t in reversed(trades_sorted[-N:]):
            print(f"{t['symbol']:<18}{t['trigger_date']:<14}{t['decision']:<7}"
                  f"{t['reason']:<32}{t.get('leverage',0):>5}"
                  f"{t['pnl_pct']*100:>+9.1f}%{t.get('exit_reason','?'):>15}"
                  f"{t.get('bars_held','?'):>10}")

    print(f"\n[backtest] wrote {args.output}")


if __name__ == "__main__":
    main()
