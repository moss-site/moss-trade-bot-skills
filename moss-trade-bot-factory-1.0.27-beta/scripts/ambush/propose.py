#!/usr/bin/env python3
"""
Ambush bot 参数推断器。

根据用户描述的「方向偏好」+「激进度」，从预设表生成 Ambush 参数 JSON。
trigger 只作为本地回测的后端默认阈值假设保留，不会提交给后端。

参数语义详见：
  knowledge/ambush_params_reference.md

调用方（SKILL.md Ambush Step 2）：
  python3 propose.py --direction balanced --aggressiveness default \
                     --output /tmp/ambush_params.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

# ---- 本地回测触发阈值（固定使用后端 env 默认值）----
# 实盘触发阈值由 server env 统一控制，不是 per-bot 参数：
#   AMBUSH_OI_MC_THRESHOLD=0.20
#   AMBUSH_Z_SCORE_THRESHOLD=2.5
#   AMBUSH_SURGE_15M_THRESHOLD=0.08
BACKTEST_TRIGGER_DEFAULTS = {
    "oi_mc_threshold":     0.20,
    "z_score_threshold":   2.5,
    "surge_15m_threshold": 0.08,
}

# ---- Long 仓位预设（仓位随激进度变化；杠杆封顶 3x；其他字段固定默认）----
# 2026-05-18 update：lev=3 hardcap 下的 576-combo 真实 K 线回放 sweep
# (pp × sl × tr × mh × momentum_bars) 显示：
#   - 紧 stop (sl=0.08) + 宽 trail (tr=0.30) + mb=0 在 216 事件上 sharpe 0.188，
#     aggressive (pp=0.45) → +850% ret / dd -39%；
#   - momentum_bars > 0 显著降低收益 (mb=2 -220%, mb=4 -730%)，long 信号已经
#     过初步确认，再加延迟反而错过 fast move。
#
# leverage 不再分档：Hyperliquid 把异动币目标人群（127/230 永续）全部 cap 在 3x，
# server `ValidateAmbushBotParams` 拒绝 leverage > 3 的参数，`validateSourceLeverage`
# 在下单时再校验一次。所以「激进」只通过 `position_pct` 表达。
LONG_LEVERAGE_POSITION = {
    "conservative": {"leverage": 3, "position_pct": 0.15},
    "default":      {"leverage": 3, "position_pct": 0.20},
    "aggressive":   {"leverage": 3, "position_pct": 0.30},
}
LONG_FIXED = {
    "stop_loss_pct":  0.08,   # 旧 0.20；sweep top 全部使用 0.08，失败入场快速止损
    "sl_atr_mult":    2.0,    # NEW 2026-05-20: ATR-based stop when --kline-driven-close on
    "trailing_pct":   0.30,   # 旧 0.25；让赢家跑得更远
    "max_hold_hours": 30,
    "momentum_bars":  0,      # 旧 2；sweep 证明 0 收益最高
    "cooldown_bars":  1,
}

# ---- Short 仓位预设（同样：杠杆封顶 3，激进度走 position_pct）----
# 2026-05-18 update：lev=3 hardcap 下的 1600-combo 真实 K 线回放 sweep
# (pp × sl × tr × mh × entry_delay) 显示：
#   - entry_delay_bars=16 (即 4h) 是决定性改动 — ed=0/4 共 640 组合 0 个赚钱，
#     ed=16 zone 几乎全部赚钱；异动初始 squeeze tail 必须躲过去；
#   - 宽 stop (sl=0.40) + 中 trail (tr=0.25) + mh=168h (7d) 是最优结构；
#   - aggressive (pp=0.30) → +127% ret / dd -70%；继续放大 pp 到 0.45 外推
#     dd 接近 -100%（单事件即爆仓），故 aggressive 上限下调到 0.30。
SHORT_LEVERAGE_POSITION = {
    "conservative": {"leverage": 3, "position_pct": 0.15},   # 旧 0.20
    "default":      {"leverage": 3, "position_pct": 0.20},   # 旧 0.30
    "aggressive":   {"leverage": 3, "position_pct": 0.30},   # 旧 0.45
}
SHORT_FIXED = {
    "stop_loss_pct":    0.40,   # 旧 0.28；窄 stop 在异动初始 squeeze 必出局
    "sl_atr_mult":      2.5,    # NEW 2026-05-20: wider for short squeeze tails
    "trailing_pct":     0.25,   # 旧 0.28；top 区域统一 0.25
    "max_hold_hours":   168,    # 旧 132；7d 让回归走完
    "cooldown_bars":    15,
    "entry_delay_bars": 16,     # 旧 1；最关键改动 — 4h 延迟让 squeeze 走完
}

# ---- Rhythm 共享默认（不随激进度变）----
RHYTHM_DEFAULTS = {
    "max_trades_per_event": 1,
    "same_coin_dedup_days": 7,
}


def build_params(direction: str, aggressiveness: str) -> dict:
    # NOTE: `trigger` is local-backtest only. Phase 2 moved trigger thresholds
    # to global server-side config (AmbushOIMCThreshold / AmbushZScoreThreshold /
    # AmbushSurge15mThreshold). `_ambush_params_for_wire` strips this block
    # before POSTing, so do not present it as generated bot config.
    return {
        "strategy_type": "ambush",
        "direction": direction,
        "trigger": dict(BACKTEST_TRIGGER_DEFAULTS),
        "long_params": {
            **LONG_FIXED,
            **LONG_LEVERAGE_POSITION[aggressiveness],
        },
        "short_params": {
            **SHORT_FIXED,
            **SHORT_LEVERAGE_POSITION[aggressiveness],
        },
        "rhythm": dict(RHYTHM_DEFAULTS),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--direction", required=True,
        choices=["long", "short", "balanced"],
        help="触发后的开仓方向偏好",
    )
    p.add_argument(
        "--aggressiveness", default="conservative",
        choices=["conservative", "default", "aggressive"],
        help="激进度，只影响 long/short 仓位预设；本地回测 trigger 固定使用后端默认 "
             "(oi_mc=0.20, z=2.5, surge=0.08)。默认 conservative。",
    )
    p.add_argument(
        "--output", required=True,
        help="输出 JSON 路径（典型 /tmp/ambush_params.json）",
    )
    args = p.parse_args()

    cfg = build_params(args.direction, args.aggressiveness)
    Path(args.output).write_text(json.dumps(cfg, indent=2, ensure_ascii=False))

    # 给 SKILL.md AI 助手回显的摘要 — 方便用户一眼看到关键参数
    print(f"[propose] wrote {args.output}")
    print(f"  direction={cfg['direction']}, aggressiveness={args.aggressiveness}")
    t = cfg["trigger"]
    print(f"  backtest trigger defaults: oi_mc={t['oi_mc_threshold']:.2f}  "
          f"z_score={t['z_score_threshold']:.1f}  "
          f"surge={t['surge_15m_threshold']:.2f}")
    lp = cfg["long_params"]
    sp = cfg["short_params"]
    print(f"  long:       leverage={lp['leverage']:>2d}  "
          f"position_pct={lp['position_pct']:.2f}  "
          f"stop_loss={lp['stop_loss_pct']:.2f}  "
          f"max_hold={lp['max_hold_hours']}h")
    print(f"  short:      leverage={sp['leverage']:>2d}  "
          f"position_pct={sp['position_pct']:.2f}  "
          f"stop_loss={sp['stop_loss_pct']:.2f}  "
          f"max_hold={sp['max_hold_hours']}h")


if __name__ == "__main__":
    main()
