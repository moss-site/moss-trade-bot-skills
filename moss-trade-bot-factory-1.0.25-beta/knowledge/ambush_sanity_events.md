# Ambush 8 标杆事件清单

> Sanity check 用。用户跑回测时，bot 必须对这 8 个**有代表性的历史异动事件**给出明确判决；判决符合预期 → 一目了然知道 bot 在典型场景下是否合理。
>
> ⚠️ 本文件初版列出**挑选方法论**和**字段格式**；具体 8 个事件清单 Week 0b 完成后填实。当前是**占位**。

## 为什么是 8 个 + 4 类

ambush 历史事件 216 条，对用户人话表示是不可能的。挑 8 个**有代表性**的：

- 让用户看完能形成对 bot "倾向" 的直观判断
- 4 类各 2 个，覆盖三路决策（short / long / skip）+ 边缘案例
- 不太多（避免用户读不完），不太少（覆盖各种典型）

## 4 类标杆

| 类别 | 选法 | 数量 | 期望 bot 判决 |
|------|------|------|-------------|
| **大涨翻车** | `label_short=short` 且 `return_24h < -30%` 的典型 | 2 | `short` ✓ |
| **持续大涨** | `label_long=long` 且 `return_48h > +50%` 的典型 | 2 | `long` ✓（balanced 命中 long 规则）/ direction=short 的 bot 判 `skip` 也算合理 |
| **假突破回落** | `label=skip` 且触发后 24h 内涨幅 < 5% 且 7d 涨幅 < 0 | 2 | `skip` ✓ |
| **边缘案例** | label 模糊（小涨小跌）/ 触发条件擦边球 | 2 | 任意决策都合理（用户**重点观察**这两个的判决） |

## 字段格式（sanity_events.json）

```json
{
  "version": 1,
  "events": [
    {
      "event_id": "event_017",
      "symbol":          "PEPE/USDT",
      "base":            "PEPE",
      "trigger_ts":      "2025-04-12 08:30:00",
      "trigger_price":   0.0000142,
      "oi_mc":           0.42,
      "surge_15m":       0.18,
      "z_score":         2.71,
      "rsi_14":          78.3,
      "change_before_24h_pct": 32.1,
      "actual_return_24h_pct": -47.2,
      "actual_return_48h_pct": -52.0,
      "expected_direction":    "short",
      "category":              "overstretch_short",
      "rationale": "触发涨幅 18% + RSI 78 + 24h 已涨 32%，典型过度拉伸 → 高概率回调（实际 24h -47%）"
    },
    {
      "event_id": "event_032",
      "symbol":   "BONK/USDT",
      ...
      "category": "momentum_long",
      "rationale": "..."
    },
    ...
  ]
}
```

## 字段含义

| 字段 | 来源 |
|------|------|
| `event_id` | `event_NNN` 自增（按 events.csv 行号或自定义） |
| `symbol` / `base` / `trigger_ts` / `trigger_price` | events.csv 直接复制 |
| `oi_mc` / `surge_15m` / `z_score` / `rsi_14` / `change_before_24h_pct` | features.csv 对应列 |
| `actual_return_24h_pct` / `actual_return_48h_pct` | events.csv `change_after_24h(%)` / `change_after_48h(%)` |
| `expected_direction` | `short` / `long` / `skip`（人工标注） |
| `category` | `overstretch_short` / `momentum_long` / `false_breakout_skip` / `edge_case` |
| `rationale` | 一两句话说明为什么挑这个事件作标杆 + 为什么期望这个方向 |

## Week 0b 挑选 SOP

1. 加载 `data_cache/ambush/events.csv` + `features.csv`
2. 按 4 类各筛候选：

```python
import pandas as pd
df = pd.read_csv("data_cache/ambush/events.csv")
ft = pd.read_csv("data_cache/ambush/features.csv")
m = df.merge(ft, on=["symbol", "trigger_ts"])

# 大涨翻车：label_short = "short" + return_24h < -30%
overstretch = m[(m["label_short"] == "short") & (m["return_24h"] < -30)]\
    .sort_values("return_24h").head(10)

# 持续大涨：label_long = "long" + return_48h > 50%
momentum_long = m[(m["label_long"] == "long") & (m["return_48h"] > 50)]\
    .sort_values("return_48h", ascending=False).head(10)

# 假突破：label = "skip" + abs(return_24h) < 5
false_breakout = m[(m["label"] == "skip") & (m["return_24h"].abs() < 5)]\
    .head(10)

# 边缘：触发涨幅在 surge_15m_threshold 边缘（0.08~0.12 区间）
edge = m[(m["surge"].between(0.08, 0.12))].head(10)
```

3. 每类候选里挑 2 个**币种不同**、**时间分散**的（避免都集中在同一币 / 同一周）
4. 写 rationale（一两句人话说明）
5. 汇总成 `data_cache/ambush/sanity_events.json`

## 用法

`backtest.py --include-sanity` 时，对每个标杆事件：
1. 加载该事件的特征（trigger 之前的 K 线）
2. 跑 bot 决策逻辑（双门 + 三路 + 规则）
3. 输出表格：事件 / 触发涨幅 / 实际 24h / 期望方向 / bot 判决 / 规则触发原因 / ✓/✗

```
event_017  PEPE     +18%    -47%   short    short    rule_overstretch     ✓
event_032  BONK     +12%    +73%   long     skip     momentum_failed      ✗
event_055  FLOKI    +14%    +5%    skip     skip     rule_no_match        ✓
event_077  ZEREBRO  +14%    -39%   short    short    rule_overstretch     ✓
event_103  TST      +25%    +145%  long     long     rule_momentum_init   ✓
event_118  HIPPO    +35%    -78%   short    short    rule_overstretch     ✓
event_142  FHE      +9%     +12%   skip     skip     rule_no_match        ✓
event_188  STO      +16%    -8%    边缘     skip     rule_no_match        ✓
                                                     通过: 7/8
```

## 评分标准

| 通过数 | 评级 | 含义 |
|--------|------|------|
| 8/8 | ⭐⭐⭐ | 在标杆上完全符合预期，bot 阈值组合很稳健 |
| 7/8 | ⭐⭐ | 单个失败可接受，看是否在边缘案例上失败（边缘失败 OK） |
| 5–6/8 | ⭐ | bot 倾向有偏差，建议调阈值后重跑 |
| ≤ 4/8 | ✗ | bot 不符合常识，**强烈建议**用户重新选 direction 或调阈值 |

> ⚠️ 标杆通过率不是越高越好。**全 8/8 在小样本上反而可能是过拟合**。理想是 6~8/8 + 失败的 1~2 个能讲清楚为什么（边缘 / 异常事件）。

## 后续

- 上线一段时间后，用户实盘触发的事件可以**逐步加入标杆库**（保持 8 个总数，淘汰不再典型的）
- 边缘案例的判决是**调参信号源**，多次回测中边缘案例判决变化大说明 bot 对这类事件不稳定
