# Ambush Bot 参数完整说明

> 异动小市值币策略专用参数。和 majors / BTC bot 的 5 维信号权重 + entry_threshold 那一套**完全不一样** — ambush 走「双门触发 + 规则决策 + 双通道仓位演化」。
>
> ⚠️ Ambush bot 参数**创建后不能改**。要调整只能新建一个 bot；旧 bot 可手动停止。

## 决策骨架

```
每小时全市场扫描 → OI/MC ≥ 后端全局 AMBUSH_OI_MC_THRESHOLD        ⇒ 加观察名单
观察名单 OI 监控 → Z-Score > 后端全局 AMBUSH_Z_SCORE_THRESHOLD     ⇒ 双门 ① 通过
15m K线        → 涨幅 ≥ 后端全局 AMBUSH_SURGE_15M_THRESHOLD       ⇒ 双门 ② 通过

两门齐 → 先规则判向，再按 direction 过滤:
  规则判向（涨幅/RSI/24h前涨幅）→ signal=short/long/skip
  direction=short    → 仅 signal=short 时做空（用 short_params）；signal=long/skip 则 skip
  direction=long     → 仅 signal=long 时等 momentum_bars + 双重确认后做多（用 long_params）；signal=short/skip 则 skip
  direction=balanced → signal=short 做空；signal=long 等动量确认后做多；signal=skip 则 skip

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
  "trigger":      { ... 本地回测用的后端默认触发阈值；创建 bot 时不提交 ... },
  "long_params":  { ... 做多专用：仓位 + 止损 + 动量确认 + 冷却 ... },
  "short_params": { ... 做空专用：仓位 + 止损 + 入场延迟 + 冷却 ... },
  "rhythm":       { ... 共享节奏：去重、单事件多笔 ... }
}
```

## 本地回测触发阈值（trigger）

`trigger` 只用于本地回测，并固定使用后端 env 默认值；创建 bot 时不会提交给后端。实盘触发阈值由后端统一控制。

| 字段 | 后端 env | skill 本地回测默认 | 含义 |
|------|------|------|------|
| `oi_mc_threshold` | `AMBUSH_OI_MC_THRESHOLD` | 0.20 | OI/MC ≥ 此值进观察名单 |
| `z_score_threshold` | `AMBUSH_Z_SCORE_THRESHOLD` | 2.5 | OI/MC 30 天 Z-Score 超此值算 OI 异动（双门 ①）。冷启动期（日线 OI < 14 天）退化为纯绝对值触发 |
| `surge_15m_threshold` | `AMBUSH_SURGE_15M_THRESHOLD` | 0.08 | 15m K 线涨幅 ≥ 此值算价格异动（双门 ②）|

## 方向控制

| 字段 | 取值 | 含义 |
|------|------|------|
| `direction` | `long` / `short` / `balanced` | 触发后的方向过滤。所有方向都会先按规则信号判 `long` / `short` / `skip`；`long` / `short` 只放行同方向信号，反方向记为 `direction_mismatch` 并 skip；`balanced` 放行两边。**balanced 是默认推荐** |

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

**规则化方案的固有局限**：仅靠 surge/RSI/change_before_24h 三个维度无法在事件触发瞬间区分"假突破"和"真暴涨"（同样的输入特征可能对应完全不同的后续走势）。**靠仓位层 stop_loss + trailing 兜底** — 万一判断错，损失被 stop_loss_pct 封顶。

> ⚠️ 这套规则在 v1 实施完成、积累实盘事件 ≥ 50 后会基于实测数据 review，可能微调阈值。当前是 v0 起步版本。

## 做多参数（long_params，仅 direction=long 或 balanced 命中 long 规则时生效）

> **K-line-driven close (2026-05-20)**：当 live_runner 启动时带 `--kline-driven-close` 旗标，
> close_monitor 切换到 4 优先级 cascade（ATR 止损 / max_hold / K 线收盘 trailing / signal_reverse）。
> 默认 off — 不带旗标时 `sl_atr_mult` 字段不被读取，仅占位。当前 sl_atr_mult 默认值 2.0 / 2.5
> 是工程师初值，未经回测校准（follow-up: 扩展 calibrate_thresholds.py）。
>
> **Backtest cost model (2026-05-22)**：本地 Ambush 回测与后端 hosted Ambush backtest 使用同一成本口径：
> 共享固定深度曲线修正 entry/exit 成交价，taker fee 固定 4.5bps，funding 固定 +0.00125% 并按整点结算。
> 这些是回测/hosted 成本假设，不属于用户提交给后端的 bot 参数。

| 字段 | 范围 | 默认 | 含义 |
|------|------|------|------|
| `leverage` | 1 ~ 3 | 3 | 杠杆。**Hyperliquid 把异动币目标人群（127 / 230 个永续）全部 cap 在 3x**，server 端按 HL meta 强制——无论用户多激进，写超过 3 会被 `validateSourceLeverage` 拒绝。激进度通过 `position_pct` 表达 |
| `position_pct` | 0.10 ~ 0.50 | 0.20 | 单笔保证金占可用资金比例 |
| `stop_loss_pct` | 0.05 ~ 0.40 | 0.08 | 硬止损。**2026-05-18 lev=3 sweep 调整**：旧 0.20 在 long 信号失败时拖时间，0.08 让失败入场迅速止损 |
| `sl_atr_mult` | 1.0 ~ 4.0 | 2.0 | 配合 `--kline-driven-close` 使用：实际止损距离 = `sl_atr_mult × ATR(14) / entry`。低于此点 × 杠杆 → 平仓。原 `stop_loss_pct` 作为 HL K 线 API 不可用时的 fallback floor 保留 |
| `trailing_pct` | 0.05 ~ 0.30 | 0.30 | 移动止盈回撤。**调整**：旧 0.25 → 0.30，让赢家跑得更远（lev=3 sweep top 全部 0.30）|
| `max_hold_hours` | 6 ~ 168 | 30 | 最长持仓时长。做多典型 1~2 天 |
| `momentum_bars` | 0 ~ 8 | 0 | 动量确认窗口：等几根 15m K 线后再开仓。**调整**：旧 2 → 0；sweep 显示 long 信号（`momentum_init` / `momentum_extend`）本身已含确认，再加延迟反而错过 fast move |
| `cooldown_bars` | 1 ~ 16 | 1 | 平仓后冷却 K 线数。做多冷却短，可立即接力下一波 |

> 做多专用：**momentum_bars** 只在 long 通道有，做空不需要。默认 0 = 不延迟；保留接口给保守用户手动开启。

## 做空参数（short_params，仅 direction=short 或 balanced 命中 short 规则时生效）

| 字段 | 范围 | 默认 | 含义 |
|------|------|------|------|
| `leverage` | 1 ~ 3 | 3 | 杠杆。和做多用同一个 HL cap=3x；做空 odds 高（统计 60% 异动 24h 内回调）但杠杆上限被交易所封死。激进度靠 `position_pct` |
| `position_pct` | 0.10 ~ 0.50 | 0.20 | 单笔保证金占可用资金比例。**2026-05-18 lev=3 sweep 下调**：旧 0.30，sweep 显示 pp=0.30 下 dd ≈ -70%，下调到 0.20 控 dd 在 -55% 内 |
| `stop_loss_pct` | 0.05 ~ 0.40 | 0.40 | 硬止损。**调整**：旧 0.28 在异动初始 squeeze 必出局；sweep 1600 top 15 全部用 0.40 |
| `sl_atr_mult` | 1.0 ~ 4.0 | 2.5 | 配合 `--kline-driven-close` 使用：实际止损距离 = `sl_atr_mult × ATR(14) / entry`。低于此点 × 杠杆 → 平仓。原 `stop_loss_pct` 作为 HL K 线 API 不可用时的 fallback floor 保留。做空容忍 squeeze tail 多一点，所以默认值高一档 |
| `trailing_pct` | 0.05 ~ 0.30 | 0.25 | 移动止盈回撤。**调整**：旧 0.28 → 0.25，sweep top 区域统一 0.25 |
| `max_hold_hours` | 6 ~ 168 | 168 | 最长持仓时长。**调整**：旧 132 → 168 (7d)，让 OI 回归走完 |
| `cooldown_bars` | 1 ~ 16 | 15 | 平仓后冷却 K 线数。做空冷却长，避免反复被轧 + 反弹消耗 |
| `entry_delay_bars` | 0 ~ 32 | 16 | 入场延迟：触发后先静默 N 根 K 线再开始评估。**关键调整**：旧 1 → 16 (4h)；sweep 显示 ed=0/4 共 640 组合 0 个赚钱，ed=16 区域几乎全部赚钱 — 异动初始 squeeze tail 必须躲过去 |

> 做空专用：**entry_delay_bars** 只在 short 通道有，做多不需要（做多要快、不需要等顶部）。

## 节奏参数（rhythm，共享）

| 字段 | 范围 | 默认 | 含义 |
|------|------|------|------|
| `max_trades_per_event` | 1 ~ 5 | 1 | 同一异动事件内最多串行交易次数（平仓 → 冷却 → 再开仓）|
| `same_coin_dedup_days` | 1 ~ 14 | 7 | 同币触发后多少天内不再重复触发 |

> 💾 **节奏 gate 的实现位置**：以上 3 个 rhythm 字段（`max_trades_per_event` / `same_coin_dedup_days` / 各方向的 `cooldown_bars`）由 **skill 侧** `event_handler.py` 在决策后 / 下单前强制执行，状态存在 skill 本地 SQLite `~/.moss-trade-bot/ambush_state.db` 的 `symbol_action_history` 表（schema 见 `scripts/ambush/live_database.py`）。
>
> server 端**不**拦这 3 项 — 因此这些字段：
> 1. **不在 server `/api/v2` 端点暴露**：要审计「这个 bot 在 SAGA 上最近 7 天开过几次」，可启动 skill 本地只读 REST：`python -m ambush.live_runner ... --action-history-port 8765`，然后查 `GET http://127.0.0.1:8765/symbol-action-history?symbol=SAGA&limit=20`；也可直接 `sqlite3 ~/.moss-trade-bot/ambush_state.db 'SELECT * FROM symbol_action_history WHERE hl_symbol="SAGA" ORDER BY id DESC LIMIT 20'`，或者退回看 server 侧的 `/api/v2/.../orders` 全量订单流。
> 2. **skill 重启后历史保留**：SQLite 路径默认在 `~/.moss-trade-bot/`，跨 live_runner 进程重启不丢。
> 3. **server 端的兜底是 `enforceAmbushPositionLock`**（仓位锁）：即便 skill 本地节奏 gate 全部失效，也不会出现「同 bot 多持仓」。但「同币 7 天内不要再开」这种纯节奏约束是 skill-only。
> 4. **deferred open 可恢复**：`momentum_bars` / `entry_delay_bars` 产生的延迟开仓写入 SQLite `deferred_opens` 表；live_runner 重启会重新挂起未完成的延迟开仓，若已超过 due time 15 分钟则标记 `deferred_expired`，避免很晚才追单。

## 不同 direction 下的参数生效范围

| direction | long_params | short_params | 备注 |
|---|---|---|---|
| `long` | 命中 long 规则用 | — 不生效 | 只放行 long 信号；short 信号 skip |
| `short` | — 不生效 | 命中 short 规则用 | 只放行 short 信号；long 信号 skip |
| `balanced` | 命中 long 规则用 | 命中 short 规则用 | 规则判向后按方向取参 |

> ⚠️ **propose.py 默认两组都填**。即便用户说"只做多"，short_params 也写全（避免用户后悔了改 direction 时丢参数）。

## ⚠️ 风控前提

ambush bot **同时只允许 1 个持仓**（"单持仓锁"）。已持有 A 币时 B 币触发会被自动拒绝，决策日志写 `single_position_lock`。该约束由 moss server 强制，不可调。

---

## 自然语言意图 → 参数推断规则

> Step 1 用，**优先从用户描述推断，不反复追问**。

### 风格映射

| 用户描述关键词 | direction | 仓位倾向 |
|--------------|-----------|------------|
| "做空为主" / "抓回调" / "妖币翻车" | `short` | 由 aggressiveness 决定 |
| "动量启动" / "抓上涨" / "追涨" | `long` | 中性 |
| "双向" / "balanced" / 未明说 | `balanced` | 中性 |

### 激进度映射（影响仓位 — leverage 永远封顶 3x）

> ⚠️ trigger 阈值 (`oi_mc` / `z_score` / `surge_15m`) 已经移到 server config，**per-bot 不可调**。表里只保留风格 → 仓位的映射。
>
> ⚠️ leverage 不再分档：HL 把异动币目标人群（127 / 230 永续）全部 cap 在 3x，无论用户口语多激进都得 ≤3。skill 推断时直接用 3，「激进」靠 `position_pct` 拉高仓位规模。

| 用户描述 | leverage（双向都 3）| position_pct（long / short） |
|---------|--------------------|-----------------------------|
| "保守" / "小试" / "稳健" | 3 | 0.15 / 0.15 |
| 默认 / 未说 | 3 | 0.20 / 0.20 |
| "激进" / "梭哈" / "抓极端" | 3 | 0.30 / 0.30 |

> ⚠️ 2026-05-18 下调：sweep 显示 short pp≥0.45 在 216 事件上 dd 接近 -100%（单事件即爆仓），aggressive 上限收紧到 0.30；short conservative/default 同步下调一档。long 三档保持不变（dd 可控 ≤ -45%）。

> 如果用户口语描述里出现"几倍杠杆"之类具体数字（哪怕只是"5 倍"），skill 必须主动说明：HL 已经把这些币 cap 在 3 倍，写超过会下不出单。然后落到 3。

### 默认值（用户描述里没出现的参数）

```python
defaults = {
    "long_params": {
        "leverage":      3,          # HL cap，强制
        "stop_loss_pct": 0.08,       # 2026-05-18 sweep: 旧 0.20 → 0.08（紧 stop 失败快撤）
        "sl_atr_mult":   2.0,        # 2026-05-20: ATR-based stop when --kline-driven-close on
        "trailing_pct":  0.30,       # 2026-05-18 sweep: 旧 0.25 → 0.30（让赢家跑）
        "max_hold_hours": 30,
        "momentum_bars": 0,          # 2026-05-18 sweep: 旧 2 → 0（long 信号已含确认）
        "cooldown_bars": 1,
    },
    "short_params": {
        "leverage":         3,       # HL cap，强制
        "stop_loss_pct":    0.40,    # 2026-05-18 sweep: 旧 0.28 → 0.40（躲初始 squeeze）
        "sl_atr_mult":      2.5,     # 2026-05-20: wider for short squeeze tails
        "trailing_pct":     0.25,    # 2026-05-18 sweep: 旧 0.28 → 0.25
        "max_hold_hours":   168,     # 2026-05-18 sweep: 旧 132 → 168 (7d)
        "cooldown_bars":    15,
        "entry_delay_bars": 16,      # 2026-05-18 sweep: 旧 1 → 16（4h 关键改动）
    },
    "rhythm": {
        "max_trades_per_event": 1,
        "same_coin_dedup_days": 7,
    },
}
```

### 反追问示例

❌ 不要这样问："你想保守还是激进？" "做多还是做空？" "杠杆要多少？"
✅ 直接按推断结果跑回测，把总结 + 最差/最好 N 笔展示给用户，不满意他自己说要调哪个。

---

## 硬约束（不可调）

- `direction` 三选一：`long` / `short` / `balanced`
- **`leverage` ∈ [1, 3]**（Hyperliquid 已经把异动币目标人群全部 cap 在 3x；server `ValidateAmbushBotParams` 直接拒绝超过 3 的参数，`validateSourceLeverage` 在下单时再校验一次。skill 推断阶段也应该硬夹）
- `oi_mc_threshold` / `z_score_threshold` / `surge_15m_threshold` 已经移到 server config，per-bot 不再可设；skill 本地回测固定使用后端默认值
- `position_pct × leverage` 实际敞口建议 ≤ 5.0；超过会被回测拒绝并提示"风险过高"。**双通道分别校验**：long_params 和 short_params 各自的 `position_pct × leverage` 都要满足。在 leverage=3 的现实下，position_pct 上限自然落到约 0.50（×3 = 1.5，远小于 5.0），所以这条约束在异动币场景下基本不会触发，主要由 `position_pct` 自身的 0.50 上限管住
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
| sl_atr_mult | 有 | **有**（long_params 默认 2.0 / short_params 默认 2.5；仅 `--kline-driven-close` 时生效） |
| 双方向参数 | 单组（long_bias 一个数控制方向） | **双通道**（long_params + short_params 完全独立） |
| 进化 | 周级 cron 自动 | **不进化**（参数固化）|
| 上线后改参数 | 支持（重新上传） | **不支持**（只能新建 bot）|
