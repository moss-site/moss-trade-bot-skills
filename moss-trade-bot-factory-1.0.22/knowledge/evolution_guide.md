# 进化机制完整指南

## 核心概念：进化 = 边回测边反思

进化不是回测之后的独立步骤，而是嵌入在回测过程中：

```
第1周数据 → 用初始参数回测 → 分析结果 → 调整战术参数
    ↓
第2周数据 → 用调整后参数回测 → 分析结果 → 再调整
    ↓
第3周数据 → ... 以此类推，直到数据用完
```

## 分段回测上下文（你必须看这些再做调参）

evolution_log 每段包含：

- `segment_result.exit_reasons` — 出场原因统计（stop_loss / trailing_stop / take_profit / signal_reverse 各多少次）
- `segment_result.avg_win_pct / avg_loss_pct` — 平均盈利/亏损百分比
- `segment_result.longs / shorts` — 多空方向分布
- `market_context` — 本段BTC价格走势和regime
- `cumulative_context` — 累计收益、峰值、峰值回撤、累计胜率、近3段表现
- `recent_trades` — 最近8笔交易明细（方向/价格/盈亏/出场原因）

**不能只看 total_return 一个数字。** 典型判断：
- 止损次数占80%+ → 考虑加宽 sl_atr_mult
- 全是同方向止损 → long_bias 方向和行情不符（但不能改，只能调 entry_threshold）
- 均盈 > 均亏但胜率低 → 结构健康，不要大改

## 反思 7 原则

1. **先看大局再看细节** — 累计收益是正的，说明核心方向没错，本周期亏钱可能只是短期波动，不要过度反应
2. **分析哪些交易赚了、为什么** — 趋势判断对了？止损设得好？滚仓放大了？
3. **分析哪些交易亏了、为什么** — 止损太紧被扫？方向判错？入场阈值太低信号太多？
4. **找出参数的具体问题** — 不要泛泛而谈，要指出 "sl_atr_mult=1.5太紧，应该加宽到2.0" 这样的具体建议
5. **微调而非重设** — 你是在优化，不是重新设计。单个参数单次调整不超过初始值的 10%
6. **保持惯性** — 上一轮调参后效果还没充分体现（<2段），本轮应保持不变或仅微调
7. **不能连续3轮以上不调整** — 市场在变，连续3段没调任何参数，必须至少调1个参数做微调（哪怕只调1-2%），保持策略"活性"

## 硬性约束（代码层面已强制执行）

### 性格参数永远不改
long_bias, base_leverage, max_leverage, risk_per_trade, max_position_pct, rolling_*, 所有 signal_weight — 即使你在进化计划里改了，代码也会强制回滚。

### 战术参数有漂移上限
每个战术参数不能偏离初始值超过 ±30%。例如初始 entry_threshold=0.32，最多调到 0.22~0.42。代码会自动钳制。

### 允许调整的参数
entry_threshold, exit_threshold, sl_atr_mult, tp_rr_ratio, trailing_activation_pct, trailing_distance_atr, regime_sensitivity, exit_on_regime_change, supertrend_mult, trend_strength_min, fast_ma_period, slow_ma_period, rsi_period, rsi_overbought, rsi_oversold

## segment-bars 计算

| 时间级别 | 进化频率 | segment-bars |
|----------|----------|-------------|
| 15m | 每周 | 672 |
| 15m | 每天 | 96 |
| 1h | 每周 | 168 |
| 1h | 每天 | 24 |
