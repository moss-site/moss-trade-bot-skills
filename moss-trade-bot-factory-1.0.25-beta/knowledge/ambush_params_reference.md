# Ambush Bot 参数完整说明

> 异动小市值币策略专用参数。和 majors / BTC bot 的 5 维信号权重 + entry_threshold 那一套**完全不一样** — ambush 走「双门触发 + 规则决策 + 双通道仓位演化」。
>
> ⚠️ Ambush bot 参数**创建后不能改**。要调整只能新建一个 bot；旧 bot 可手动停止。

## 决策骨架

```
每小时全市场扫描 → OI/MC ≥ trigger.oi_mc_threshold        ⇒ 加观察名单
观察名单 OI 监控 → Z-Score > trigger.z_score_threshold     ⇒ 双门 ① 通过
15m K线        → 涨幅 ≥ trigger.surge_15m_threshold       ⇒ 双门 ② 通过

两门齐 → 按 direction 决策:
  direction=short    → 立即做空（用 short_params）
  direction=long     → 等 momentum_bars 根 K线 + 双重确认 → 做多（用 long_params）
  direction=balanced → 规则判向（涨幅/RSI/24h前涨幅）
                       命中 short 规则 → 做空（用 short_params）
                       命中 long 规则  → 等动量确认 → 做多（用 long_params）

决策方向 → 用对应 params（leverage/position_pct/stop_loss_pct/trailing_pct/max_hold_hours）走 source-core 下单
```

> ⚠️ 关键点：**做多和做空是两组独立参数**（`long_params` / `short_params`），不是一组。
> 因为做多和做空在异动币上的风险结构本质不同：做空容易瞬间被轧爆（高杠杆 + 严止损 + 长持仓等翻车）；做多容易回吐（低杠杆 + 移动止盈 + 短持仓）。
> 一组参数在两个方向上都次优。

## 参数结构

```json
{
  "strategy_type": "ambush",
  "direction": "balanced",
  "trigger":      { ... 共享触发阈值 ... },
  "long_params":  { ... 做多专用：仓位 + 止损 + 动量确认 + 冷却 ... },
  "short_params": { ... 做空专用：仓位 + 止损 + 入场延迟 + 冷却 ... },
  "rhythm":       { ... 共享节奏：去重、单事件多笔 ... }
}
```

## 触发参数（trigger，共享）

| 字段 | 范围 | 默认 | 含义 |
|------|------|------|------|
| `oi_mc_threshold` | 0.20 ~ 0.50 | 0.35 | OI/MC ≥ 此值进观察名单。低=更多币入观察、触发频繁但精度降；高=更严苛 |
| `z_score_threshold` | 1.5 ~ 3.0 | 2.0 | OI/MC 30 天 Z-Score 超此值算 OI 异动（双门 ①）。冷启动期（日线 OI < 14 天）退化为纯绝对值触发 |
| `surge_15m_threshold` | 0.05 ~ 0.20 | 0.10 | 15m K 线涨幅 ≥ 此值算价格异动（双门 ②）|

## 方向偏好

| 字段 | 取值 | 含义 |
|------|------|------|
| `direction` | `long` / `short` / `balanced` | 触发后开仓方向偏好。**balanced 是默认推荐** |

### balanced 模式的判向规则 v0（不可调）

5 条规则按顺序匹配，**任一 short 规则命中**就 short；**任一 long 规则命中**就 long；都不命中 skip。

```
SHORT 信号（任一满足）:
  rule_short_compound_overstretch:
    触发涨幅 > 20% AND 24h 前已涨 > 20%
    （叠加狂涨：单根大阳线 + 多日已涨，崩盘风险高）

  rule_short_extreme_pullback:
    24h 前已涨 > 100% AND RSI > 70
    （极端拉伸高位：异常涨幅 + 高位 RSI，回调概率大）

  rule_short_spike_extreme:
    触发涨幅 > 25%
    （单根 K 线极端：通常是顶部 spike，反转概率高）

LONG 信号（任一满足）:
  rule_long_momentum_init:
    10% < 触发涨幅 < 15% AND RSI < 60 AND 24h 前涨 < 10%
    （早期动量：刚启动，未到极端，有持续空间）

  rule_long_momentum_extend:
    30 ≤ 24h 前涨幅 ≤ 80 AND RSI > 75
    （中段动量持续：已经在涨势中，触发是再次发力）
```

**8 标杆 sanity 验证**：v0 规则通过 7/8。失败案例 ALPINE（数据和 TNSR 几乎一样但实际假突破）反映规则化方案固有局限 — 仅靠 surge/RSI/change_before_24h 三维度无法区分"假突破"和"暴涨"。**靠仓位层 stop_loss + trailing 兜底**。

> ⚠️ 这套规则在 v1 实施完成、积累实盘事件 ≥ 50 后会基于实测数据 review，可能微调阈值。当前是 v0 起步版本。

## 做多参数（long_params，仅 direction=long 或 balanced 命中 long 规则时生效）

| 字段 | 范围 | 默认 | 含义 |
|------|------|------|------|
| `leverage` | 3 ~ 15 | 4 | 杠杆。做多偏保守，避免冲高回吐时高杠杆爆仓 |
| `position_pct` | 0.10 ~ 0.50 | 0.20 | 单笔保证金占可用资金比例 |
| `stop_loss_pct` | 0.05 ~ 0.40 | 0.20 | 硬止损 |
| `trailing_pct` | 0.05 ~ 0.30 | 0.25 | 移动止盈回撤（做多重在吃趋势，回撤稍宽） |
| `max_hold_hours` | 6 ~ 168 | 30 | 最长持仓时长。做多典型 1~2 天 |
| `momentum_bars` | 1 ~ 8 | 2 | 动量确认窗口：等几根 15m K 线（Hyperliquid candleSnapshot），收盘 close 持续 > 触发价才进场 |
| `cooldown_bars` | 1 ~ 16 | 1 | 平仓后冷却 K 线数。做多冷却短，可立即接力下一波 |

> 做多专用：**momentum_bars** 只在 long 通道有，做空不需要。

## 做空参数（short_params，仅 direction=short 或 balanced 命中 short 规则时生效）

| 字段 | 范围 | 默认 | 含义 |
|------|------|------|------|
| `leverage` | 3 ~ 15 | 8 | 杠杆。做空 odds 较高（统计显示约 60% 异动 24h 内回调），允许更高杠杆 |
| `position_pct` | 0.10 ~ 0.50 | 0.30 | 单笔保证金占可用资金比例 |
| `stop_loss_pct` | 0.05 ~ 0.40 | 0.28 | 硬止损。做空容易瞬间被轧（异动币高位还能再继续轧 30~50%），止损要更宽 |
| `trailing_pct` | 0.05 ~ 0.30 | 0.28 | 移动止盈回撤 |
| `max_hold_hours` | 6 ~ 168 | 132 | 最长持仓时长。做空可拿很久（5+ 天）等翻车确认 |
| `cooldown_bars` | 1 ~ 16 | 15 | 平仓后冷却 K 线数。做空冷却长，避免反复被轧 + 反弹消耗 |
| `entry_delay_bars` | 0 ~ 4 | 1 | 入场延迟：触发后先静默 N 根 K 线再开始评估，避免初始 spike 顶部入场 |

> 做空专用：**entry_delay_bars** 只在 short 通道有，做多不需要（做多要快、不需要等顶部）。

## 节奏参数（rhythm，共享）

| 字段 | 范围 | 默认 | 含义 |
|------|------|------|------|
| `max_trades_per_event` | 1 ~ 5 | 1 | 同一异动事件内最多串行交易次数（平仓 → 冷却 → 再开仓）|
| `same_coin_dedup_days` | 1 ~ 14 | 7 | 同币触发后多少天内不再重复触发 |

## 不同 direction 下的参数生效范围

| direction | long_params | short_params | 备注 |
|---|---|---|---|
| `long` | ✓ 用 | — 不生效 | 即便填了也不会用 |
| `short` | — 不生效 | ✓ 用 | 即便填了也不会用 |
| `balanced` | 命中 long 规则用 | 命中 short 规则用 | 规则判向后按方向取参 |

> ⚠️ **propose.py 默认两组都填**。即便用户说"只做多"，short_params 也写全（避免用户后悔了改 direction 时丢参数）。

## ⚠️ 风控前提

ambush bot **同时只允许 1 个持仓**（"单持仓锁"）。已持有 A 币时 B 币触发会被自动拒绝，决策日志写 `single_position_lock`。该约束由 moss server 强制，不可调。

---

## 自然语言意图 → 参数推断规则

> Step 1 用，**优先从用户描述推断，不反复追问**。

### 风格映射

| 用户描述关键词 | direction | 触发阈值倾向 |
|--------------|-----------|------------|
| "做空为主" / "抓回调" / "妖币翻车" | `short` | 偏激进（低 surge 阈值，多触发） |
| "动量启动" / "抓上涨" / "追涨" | `long` | 中性 |
| "双向" / "balanced" / 未明说 | `balanced` | 中性 |

### 激进度映射（影响 trigger + 仓位）

| 用户描述 | trigger.* | leverage（双向都缩放） | position_pct（双向都缩放） |
|---------|-----------|----------------------|----------------------------|
| "保守" / "小试" / "稳健" | oi_mc=0.40 / z=2.5 / surge=0.12 | long=3 / short=5 | long=0.15 / short=0.20 |
| 默认 / 未说 | oi_mc=0.35 / z=2.0 / surge=0.10 | long=4 / short=8 | long=0.20 / short=0.30 |
| "激进" / "梭哈" / "抓极端" | oi_mc=0.30 / z=1.8 / surge=0.08 | long=8 / short=12 | long=0.30 / short=0.45 |

### 默认值（用户描述里没出现的参数）

```python
defaults = {
    "long_params": {
        "stop_loss_pct": 0.20,
        "trailing_pct":  0.25,
        "max_hold_hours": 30,
        "momentum_bars": 2,
        "cooldown_bars": 1,
    },
    "short_params": {
        "stop_loss_pct": 0.28,
        "trailing_pct":  0.28,
        "max_hold_hours": 132,
        "cooldown_bars": 15,
        "entry_delay_bars": 1,
    },
    "rhythm": {
        "max_trades_per_event": 1,
        "same_coin_dedup_days": 7,
    },
}
```

### 反追问示例

❌ 不要这样问："你想保守还是激进？" "做多还是做空？" "杠杆要多少？"
✅ 直接按推断结果跑回测，sanity check 表给用户看，不满意他自己说要调哪个。

---

## 硬约束（不可调）

- `direction` 三选一：`long` / `short` / `balanced`
- `oi_mc_threshold` 不能 < 0.20（< 0.20 触发量爆炸 + 数据噪声）
- `position_pct × leverage` 实际敞口建议 ≤ 5.0；超过会被回测拒绝并提示"风险过高"。**双通道分别校验**：long_params 和 short_params 各自的 `position_pct × leverage` 都要满足
- `max_hold_hours` ≤ 168（更长无意义）
- 单 bot **同时只允许 1 个持仓**（不可改）
- **`long_params.cooldown_bars` 和 `short_params.cooldown_bars` 独立**：做空建议 ≥ 10，做多建议 ≤ 4

---

## 对 majors / BTC bot 参数的差异

| | majors | ambush |
|---|--------|--------|
| symbol | 创建时绑死 | **占位 `"*"`**（DB 列 NOT NULL；运行时实际持仓 symbol 由 `active_symbol` 列承载）|
| 5 维信号权重 | 有 | **没有**（用规则替代） |
| entry_threshold | 有 | **没有**（用 surge_15m_threshold 替代） |
| sl_atr_mult | 有 | **没有**（用 long/short_params 各自的 stop_loss_pct 替代） |
| 双方向参数 | 单组（long_bias 一个数控制方向） | **双通道**（long_params + short_params 完全独立） |
| 进化 | 周级 cron 自动 | **不进化**（参数固化）|
| 上线后改参数 | 支持（重新上传） | **不支持**（只能新建 bot）|
