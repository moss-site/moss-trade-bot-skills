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
from pathlib import Path
from typing import Optional

import pandas as pd

# ===== 路径 =====
DEFAULT_DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data_cache" / "ambush"


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


# ===== 仓位演化 =====

@dataclass
class TradeOutcome:
    side: str  # "long" / "short"
    entry: float
    exit_price: float
    pnl_pct: float           # 杠杆后净 PnL（小数：0.50 = +50%）
    exit_reason: str         # 'stop_loss' / 'trailing' / 'max_hold'
    bars_held: int


def simulate_position(
    klines_after: list[dict],
    entry: float,
    side: str,
    leverage: int,
    stop_loss_pct: float,
    trailing_pct: float,
    max_hold_bars: int,
) -> TradeOutcome:
    """
    用触发后 K 线序列模拟一次仓位演化。

    平仓优先级（每根 K 线内）:
      1. 硬止损（保证金亏损 ≥ stop_loss_pct）
      2. 移动止盈（从 peak 回撤 ≥ trailing_pct）
      3. 超时（max_hold_bars 内未平 → close at last close）

    pnl_pct 是杠杆后的 net % return。
    """
    bars = klines_after[:max_hold_bars]
    if not bars:
        return TradeOutcome(side, entry, entry, 0.0, "no_klines", 0)

    if side == "long":
        peak = entry
        for i, k in enumerate(bars):
            high, low = k["high"], k["low"]
            peak = max(peak, high)
            # stop_loss: low ≤ entry × (1 - stop_loss / leverage)
            stop_price = entry * (1 - stop_loss_pct / leverage)
            if low <= stop_price:
                pnl = (stop_price - entry) / entry * leverage
                return TradeOutcome(side, entry, stop_price, pnl, "stop_loss", i + 1)
            # trailing: 从 peak 回撤
            if peak > entry:
                drawdown = (peak - low) / peak
                if drawdown >= trailing_pct:
                    exit_price = peak * (1 - trailing_pct)
                    pnl = (exit_price - entry) / entry * leverage
                    return TradeOutcome(side, entry, exit_price, pnl, "trailing", i + 1)
        # max_hold
        last_close = bars[-1]["close"]
        pnl = (last_close - entry) / entry * leverage
        return TradeOutcome(side, entry, last_close, pnl, "max_hold", len(bars))

    else:  # short
        trough = entry
        for i, k in enumerate(bars):
            high, low = k["high"], k["low"]
            trough = min(trough, low)
            stop_price = entry * (1 + stop_loss_pct / leverage)
            if high >= stop_price:
                pnl = (entry - stop_price) / entry * leverage
                return TradeOutcome(side, entry, stop_price, pnl, "stop_loss", i + 1)
            if trough < entry:
                drawup = (high - trough) / trough
                if drawup >= trailing_pct:
                    exit_price = trough * (1 + trailing_pct)
                    pnl = (entry - exit_price) / entry * leverage
                    return TradeOutcome(side, entry, exit_price, pnl, "trailing", i + 1)
        last_close = bars[-1]["close"]
        pnl = (entry - last_close) / entry * leverage
        return TradeOutcome(side, entry, last_close, pnl, "max_hold", len(bars))


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
    return after[["open", "high", "low", "close"]].to_dict("records")


# ===== 主回测循环 =====

def run_backtest(
    events_df: pd.DataFrame, params: dict, data_dir: Path,
) -> dict:
    direction = params["direction"]
    long_p = params["long_params"]
    short_p = params["short_params"]

    # 单事件仓位演化最长 max_hold_bars = max(long.max_hold, short.max_hold) × 4 (15m bars/h)
    max_hold_bars = max(long_p["max_hold_hours"], short_p["max_hold_hours"]) * 4

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
            sl, tr, mh_bars = long_p["stop_loss_pct"], long_p["trailing_pct"], long_p["max_hold_hours"] * 4
            lev = int(long_p["leverage"])
        else:
            delay_bars = int(short_p.get("entry_delay_bars", 0) or 0)
            sl, tr, mh_bars = short_p["stop_loss_pct"], short_p["trailing_pct"], short_p["max_hold_hours"] * 4
            lev = int(short_p["leverage"])

        bars_after = klines_after_trigger(klines, ev["trigger_ts"], mh_bars + delay_bars)
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
        outcome = simulate_position(delayed_bars, entry_price, decision, lev, sl, tr, mh_bars)
        if decision == "long":
            position_pct = float(long_p["position_pct"])
        else:
            position_pct = float(short_p["position_pct"])
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
    pnl_list = [t["pnl_pct"] for t in trades]  # 保证金相对收益（不含 position_pct）
    net_pnl_list = [t["pnl_pct"] * t.get("position_pct", 1.0) for t in trades]
    wins = [p for p in net_pnl_list if p > 0]
    total_ret = sum(net_pnl_list) * 100
    avg_pnl = total_ret / len(net_pnl_list)
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
        ps = [t["pnl_pct"] for t in ts]
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
    p.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR), help="data_cache/ambush 路径")
    p.add_argument("--dump-trades", type=int, default=0, metavar="N",
                   help="额外打印最差 N 笔 + 最好 N 笔交易明细（用于查 outlier）。"
                        "trades 列表本身始终在 result JSON 里")
    args = p.parse_args()

    data_dir = Path(args.data_dir)
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
