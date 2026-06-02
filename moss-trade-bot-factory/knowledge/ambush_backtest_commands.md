# Ambush 本地回测命令模板

> Step 3 用。Ambush 回测**和 majors 完全不一样** — 不是给 BTC 喂连续 K 线 tick，而是在 216 个历史异动事件上点状评估 bot 阈值。**单次回测 < 5 秒**。

## 数据资产前置

**v1.0.27 起走轻量模式 — 首次运行 `backtest.py` / `upload.py` 时自动从 GitHub
Release 下载数据集到用户 cache**,无需手动准备。默认路径由
`scripts/core/data_cache_archive.resolve_data_root()` 决定:

| 优先级 | 路径 | 用途 |
|---|---|---|
| 1 | `$MOSS_TRADE_BOT_DATA_DIR` (env) | 显式 dev 覆盖 |
| 2 | `{baseDir}/scripts/data_cache/` | 本地 dev cache(`.gitignore` 默认空) |
| 3 | `~/.cache/moss-trade-bot-factory/v<version>/data_cache/` | **生产路径**:首次运行自动 hydrate + sha256 校验 |

数据集结构(解压后):

```
<data_cache>/ambush/
├── events.csv                   # 216 历史异动事件 + 后续涨跌
├── features.csv                 # 30 列特征（symbol/trigger_ts + 28 项指标）
├── klines/<base>.csv × 87       # 触发窗口 K 线（剪枝后 ~7MB）
├── supply.json
└── market_cap_snapshot.json
```

如果自动下载失败,常见原因:网络不通 / GitHub Release 资产未发布 /
`archive_sha256` 不匹配。详见 `scripts/core/data_cache_archive.py`
的错误信息。

## 标准回测命令

```bash
python3 {baseDir}/scripts/ambush/backtest.py \
  --params /tmp/ambush_params.json \
  --output /tmp/ambush_backtest_result.json \
  --dump-trades 3
```

`--dump-trades 3` 让 backtest 额外打印最差 3 笔 + 最好 3 笔的明细（供用户感性感受 "这种 case 会亏 / 那种 case 会赚"，比单纯看胜率直观）。

`/tmp/ambush_params.json` 由 `propose.py` 上一步产出，**双通道结构**（long_params 和 short_params 各自独立）。其中 `trigger` 只给本地回测用，固定使用后端 env 默认值；创建 bot 时不会提交给后端：

```json
{
  "strategy_type": "ambush",
  "direction": "balanced",
  "trigger": {
    "oi_mc_threshold": 0.20,
    "z_score_threshold": 2.5,
    "surge_15m_threshold": 0.08
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

> ⚠️ 即便用户选 `direction=long`，propose.py 仍要填 `short_params` 默认值（避免后悔改 direction 后丢参数）。回测方向语义和实盘一致：每个事件先按规则信号判 `long` / `short` / `skip`；`direction=long` / `short` 只放行同方向信号，反方向记为 `direction_mismatch` 并 skip；`direction=balanced` 放行 long 和 short，并使用命中方向对应 params。
>
> ⚠️ `trigger` 不是生成的 bot 参数。实盘触发阈值由后端统一 env 控制：`AMBUSH_OI_MC_THRESHOLD=0.20`、`AMBUSH_Z_SCORE_THRESHOLD=2.5`、`AMBUSH_SURGE_15M_THRESHOLD=0.08`（除非部署环境显式覆盖）。

成本口径与后端 Ambush backtest 对齐：
- 平仓路径使用后端/live 4 优先级 cascade：ATR 止损 → max_hold → K 线收盘 trailing → signal_reverse。
- PnL 先按共享固定深度簿修正 entry/exit 成交价，再扣 taker fee，并按整点 funding 结算。
- 深度簿所有 Ambush 代币共用一条 USD 深度曲线；这是回测成本模型，不是用户可调参数。

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
	    "gross_return_pct": 320.1,
	    "depth_cost_pct": -0.1,
	    "trading_fee_pct": -5.2,
	    "funding_fee_pct": -2.4,
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
成本:      depth -0.1% / fee -5.2% / funding -2.4%
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
| `FileNotFoundError: data_cache/ambush/events.csv` | 自动 hydrate 失败(网络 / sha256 不匹配) | 重跑命令(会重新下载);或手动 `python3 -c "import sys; sys.path.insert(0,'scripts'); from core.data_cache_archive import ensure_data_cache; print(ensure_data_cache())"` 查具体报错 |
| `triggered_count: 0` | 阈值太严 | 三个触发阈值任一调低 |
| `WARN: no kline for XYZ` | 87 币种里缺某币 K 线 | 不影响整体结果，但该币事件被跳过；可补 K 线后重跑 |
| 回测时间 > 30 秒 | 数据加载/计算异常 | 检查 K 线文件是否完好（pd.read_csv 报错） |

## 不做的事

- ❌ Ambush bot **不需要 evolve**（参数固化）
- ❌ Ambush bot 上传时 `data_fingerprint.symbol` 写 `null` 或 `"*"`（不绑币）

## 平台 verify（可选，2026-05-27 之后）

Ambush 现在**支持**平台 verify，与主流币 verify 是**完全不同的契约**：

- 主流币 verify：走 K 线 walk-forward，按 (symbol, timeframe, date_range) 对账。
- Ambush verify：走**事件 dataset + 链式仓位回放**。skill 端用 `ChainedHarness` 跑出 `local_result`，
  连同 `fingerprint`（params + initial_capital + harness_version + dataset_sha256 的 SHA-256）一起
  POST 到 `/api/v1/moss/agent/backtest/verify-job`。server 端用同一份 dataset 重新跑链式 harness，
  对账 fingerprint；一致则签名通过，server 端持久化 verify job + 完整 trades/equity 工件，用户能拿到
  "平台已验证此参数集" 的凭证（详情接口、leaderboard、follower 都会读这个）。

何时**应该**调 verify：
- 用户想把这个 ambush bot 公开到 leaderboard（必须有 server-side verified backtest）
- 用户拿不准本地结果是否能在平台复现，想跑一遍对账
- 上线前的最后一道双方算法一致性检查

何时**可以跳过**：
- 用户只是想 propose → create-bot → 直接看实盘表现，对历史不在乎
- 已经 verify 过一遍且参数没改

命令模板（Step 3 后的可选 Step 3.5）：

```bash
python3 {baseDir}/scripts/ambush/upload.py \
  --params /tmp/ambush_params.json \
  --backtest-result /tmp/ambush_backtest_result.json \
  --creds ~/.moss-trade-bot/agent_creds.json \
  --ambush-verify
# 输出:fingerprint 一致性结论 + server-side trades/equity 工件链接
```
