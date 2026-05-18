# Ambush 本地回测命令模板

> Step 3 用。Ambush 回测**和 majors 完全不一样** — 不是给 BTC 喂 148 天 K 线 tick，而是在 216 个历史异动事件上点状评估 bot 阈值。**单次回测 < 5 秒**。

## 数据资产前置

确认 `{baseDir}/data_cache/ambush/` 下有：

```
data_cache/ambush/
├── events.csv                   # 216 历史异动事件 + 后续涨跌
├── features.csv                 # 19 项预算特征
├── klines/<base>.csv × 87       # 触发窗口 K 线（剪枝后 ~7MB）
├── supply.json
└── market_cap_snapshot.json
```

如果缺，跑根目录的一次性迁移脚本：
```bash
python3 scripts/migrate_ambush_data.py
```

## 标准回测命令

```bash
python3 {baseDir}/scripts/ambush/backtest.py \
  --params /tmp/ambush_params.json \
  --output /tmp/ambush_backtest_result.json \
  --dump-trades 3
```

`--dump-trades 3` 让 backtest 额外打印最差 3 笔 + 最好 3 笔的明细（供用户感性感受 "这种 case 会亏 / 那种 case 会赚"，比单纯看胜率直观）。

`/tmp/ambush_params.json` 由 `propose.py` 上一步产出，**双通道结构**（long_params 和 short_params 各自独立）：

```json
{
  "strategy_type": "ambush",
  "direction": "balanced",
  "trigger": {
    "oi_mc_threshold": 0.35,
    "z_score_threshold": 2.0,
    "surge_15m_threshold": 0.10
  },
  "long_params": {
    "leverage": 3,
    "position_pct": 0.20,
    "stop_loss_pct": 0.20,
    "trailing_pct": 0.25,
    "max_hold_hours": 30,
    "momentum_bars": 2,
    "cooldown_bars": 1
  },
  "short_params": {
    "leverage": 3,
    "position_pct": 0.30,
    "stop_loss_pct": 0.28,
    "trailing_pct": 0.28,
    "max_hold_hours": 132,
    "cooldown_bars": 15,
    "entry_delay_bars": 1
  },
  "rhythm": {
    "max_trades_per_event": 1,
    "same_coin_dedup_days": 7
  }
}
```

> ⚠️ 即便用户选 `direction=long`，propose.py 仍要填 `short_params` 默认值（避免后悔改 direction 后丢参数）。回测时只生效对应方向的 params。

## 输出格式

`/tmp/ambush_backtest_result.json`：

```json
{
  "summary": {
    "total_events": 216,
    "triggered_count": 87,
    "triggered_pct": 0.40,
    "win_count": 46,
    "win_rate": 0.529,
    "total_return_pct": 312.4,
    "avg_pnl_pct": 3.59,
    "max_drawdown_pct": -38.2,
    "sharpe": 0.45
  },
  "per_direction": {
    "short": {"count": 62, "win_rate": 0.55, "total_return_pct": 245.0},
    "long":  {"count": 25, "win_rate": 0.48, "total_return_pct":  67.4},
    "skip":  {"count": 129, "reason_breakdown": {"momentum_failed": 41, "rule_no_match": 88}}
  },
  "trades": [
    {"symbol": "PEPE/USDC", "trigger_date": "2025-04-12", "decision": "short", ...},
    ...
  ]
}
```

## 给用户看的标准展示

回测完成后展示**三块**：

### 1. 总结表

```
触发次数:   87 / 216 (40%)
胜率:      52.9%
总收益:    +312%
最大回撤:  -38%
Sharpe:    0.45
```

### 2. 方向分布

```
做空: 62 笔, 胜率 55%, 收益 +245%
做多: 25 笔, 胜率 48%, 收益 +67%
跳过: 129 笔
  ├─ 动量失败: 41
  └─ 规则未匹配: 88
```

### 3. 最好 / 最差 N 笔（用 --dump-trades N 自动打印）

```
=== 最差 3 笔（outlier 排查）===
symbol           trigger_date  side   rule                             lev   pnl_pct  exit_reason  bars_held
SOLO/USDC        2025-04-15    short  rule_short_spike_extreme           3    -38.2%  stop_loss            4
HIPPO/USDC       2025-05-22    short  rule_short_compound_overstretch    3    -28.1%  stop_loss            6
ZEREBRO/USDC     2025-06-03    long   rule_long_momentum_init            3    -19.5%  trailing             8

=== 最好 3 笔 ===
symbol           trigger_date  side   rule                             lev   pnl_pct  exit_reason  bars_held
PEPE/USDC        2025-04-12    short  rule_short_compound_overstretch    3    +52.7%  trailing            22
TST/USDC         2025-05-01    long   rule_long_momentum_extend          3    +41.3%  max_hold            120
BONK/USDC        2025-06-14    short  rule_short_spike_extreme           3    +33.8%  trailing            18
```

让用户感性看清楚 "什么 case bot 抓得到 / 什么 case 会亏"。比单纯的胜率数字直观。

## 调参循环

用户对结果不满意时的常见调整：

| 用户反馈 | 推荐改动 |
|---------|---------|
| "触发太少了，没几笔" | `trigger.oi_mc_threshold ↓` 或 `trigger.z_score_threshold ↓` 或 `trigger.surge_15m_threshold ↓` |
| "触发太多噪声" | 反向，trigger 三个阈值任一调高 |
| "做多老亏" | `direction=short` 或 `long_params.momentum_bars ↑`（要求更强动量确认） |
| "做空回撤太大" | `short_params.stop_loss_pct ↓` 或 `short_params.position_pct ↓`（**只调 short 通道**） |
| "做多回撤太大" | `long_params.stop_loss_pct ↓` 或 `long_params.position_pct ↓`（**只调 long 通道**） |
| "做空持仓太久" | `short_params.max_hold_hours ↓` |
| "做多持仓太久" | `long_params.max_hold_hours ↓` |
| "做空冷却期太短，反复被轧" | `short_params.cooldown_bars ↑` |
| "最差几笔都是 long 那边" | balanced 规则的 long 条件偏严，可手动改 direction=short |

> ⚠️ 调参时**注意是 long_params 还是 short_params**。同样字段名在两个通道里都有，调错通道会调成"反方向"参数。

每次调整重跑一次，对比前后总结表 + 最差几笔。

## 失败排查

| 错误 | 原因 | 解决 |
|------|------|------|
| `FileNotFoundError: data_cache/ambush/events.csv` | 数据未迁移 | 跑 `scripts/migrate_ambush_data.py` |
| `triggered_count: 0` | 阈值太严 | 三个触发阈值任一调低 |
| `WARN: no kline for XYZ` | 87 币种里缺某币 K 线 | 不影响整体结果，但该币事件被跳过；可补 K 线后重跑 |
| 回测时间 > 30 秒 | 数据加载/计算异常 | 检查 K 线文件是否完好（pd.read_csv 报错） |

## 不做的事

- ❌ Ambush bot **不需要 fingerprint**（majors 是单币 + K 线时段唯一标识；ambush 是事件集合 + 阈值组合，无固定 K 线流）
- ❌ Ambush bot **不需要 evolve**（参数固化）
- ❌ Ambush bot **不调用平台 verify**（path 是平台 walk-forward；ambush 不兼容 — 没有 K 线连续回放语义，server 端用同一份事件 dataset 也无法等价重放）
- ❌ Ambush bot 上传时 `data_fingerprint.symbol` 写 `null` 或 `"*"`（不绑币）
