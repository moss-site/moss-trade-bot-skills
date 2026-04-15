# 参数完整说明

## 决策函数

```
交易信号 = trend_weight×趋势 + momentum_weight×动量 + mean_revert_weight×均值回归 + volume_weight×量能 + volatility_weight×波动率
信号 > entry_threshold → 开多 ｜ 信号 < -entry_threshold → 开空
```

## 信号权重（5个，自动归一化到和=1）

| 参数 | 含义 |
|------|------|
| trend_weight | 趋势跟踪。高=重视EMA交叉和Supertrend方向 |
| momentum_weight | 动量。高=重视RSI和MACD信号 |
| mean_revert_weight | 均值回归。高=重视布林带回归 |
| volume_weight | 量能。高=重视OBV和量价配合 |
| volatility_weight | 波动率。高=重视ATR突破/收缩 |

## 交易阈值（综合信号实际范围约 -0.5 ~ +0.5）

| 参数 | 范围 | 含义 |
|------|------|------|
| entry_threshold | 0.05~0.55 | 越低越容易触发。0.15=激进, 0.25=中性, 0.40=保守, >0.5几乎不触发 |
| exit_threshold | 0.03~0.30 | 持仓时反向信号超此值平仓 |

## 方向偏好

| 参数 | 范围 | 含义 |
|------|------|------|
| long_bias | 0~1 | 0=只做空, 0.5=双向, 1=只做多 |

## 技术参数

| 参数 | 范围 | 含义 |
|------|------|------|
| fast_ma_period | 5~50 | 快速均线周期 |
| slow_ma_period | 20~200 | 慢速均线周期（必须 > fast_ma_period） |
| trend_strength_min | 10~50 | ADX趋势强度阈值 |
| supertrend_mult | 1~5 | Supertrend 倍数 |
| rsi_period | 7~28 | RSI 周期 |
| rsi_overbought | 60~85 | RSI 超买线 |
| rsi_oversold | 15~40 | RSI 超卖线 |
| bb_period | 10~50 | 布林带周期 |
| bb_std | 1.0~3.0 | 布林带标准差倍数 |

## 杠杆与仓位

| 参数 | 范围 | 含义 |
|------|------|------|
| base_leverage | 1~150 | 基础杠杆倍数 |
| max_leverage | 1~150 | 最大杠杆 |
| risk_per_trade | 0.01~0.50 | 每笔交易使用资金比例 |
| max_position_pct | 0.05~1.0 | 单笔最大资金占比 |

## 止损/止盈

**核心公式**：杠杆 × ATR百分比 ≈ 单笔最大亏损%

| 杠杆 | 建议 sl_atr_mult |
|------|------------------|
| 5x | 1.0 |
| 10x | 2.0 |
| 20x | 2.5~3.0 |
| 50x | 3.0~5.0 |

| 参数 | 范围 | 含义 |
|------|------|------|
| sl_atr_mult | 0.5~5.0 | 止损距离 = X × ATR。**高杠杆必须配宽止损** |
| tp_rr_ratio | 1.0~10.0 | 止盈/止损距离比（风险回报比） |
| trailing_enabled | bool | 是否启用移动止损 |
| trailing_activation_pct | 0.01~0.10 | 浮盈X%后激活移动止损 |
| trailing_distance_atr | 0.5~3.0 | 移动止损距最高点 X × ATR |

## 滚仓（趋势策略利润放大器）

没有滚仓时盈亏对称（赢25%/亏25%），50%胜率=不赚钱。开启后赢单可达+100%以上。**趋势策略强烈建议开启。**

| 参数 | 范围 | 含义 |
|------|------|------|
| rolling_enabled | bool | 是否启用滚仓（用浮盈加仓） |
| rolling_trigger_pct | 0.10~0.80 | 浮盈X%时触发 |
| rolling_reinvest_pct | 0.30~1.0 | 用浮盈的X%作为新仓保证金 |
| rolling_max_times | 1~5 | 最多滚仓次数 |
| rolling_move_stop | bool | 滚仓后老仓止损移到成本价 |

## Regime 敏感度

| 参数 | 范围 | 含义 |
|------|------|------|
| regime_sensitivity | 0~1 | 0=忽略行情阶段, 1=严格只在匹配行情交易 |
| exit_on_regime_change | bool | 行情切换时是否立即平仓 |

## 参数分类（进化时的约束）

- **性格参数（不可调）**：long_bias, base/max_leverage, risk_per_trade, max_position_pct, rolling_*, 所有 signal weight
- **战术参数（可微调）**：entry/exit_threshold, sl_atr_mult, tp_rr_ratio, trailing_*, regime_*, 技术参数

## 调参速查表

| 用户说 | 调的参数 |
|--------|----------|
| "止损太紧/被扫太多" | sl_atr_mult ↑（如 2.0→2.8） |
| "止损太松/亏损太大" | sl_atr_mult ↓ |
| "交易太频繁" | entry_threshold ↑（如 0.2→0.35） |
| "交易太少" | entry_threshold ↓ |
| "杠杆太高/低" | base_leverage, max_leverage |
| "更激进" | base_leverage↑, risk_per_trade↑ |
| "更保守" | base_leverage↓, entry_threshold↑ |
| "多做多/空" | long_bias ↑/↓ |
| "开/关滚仓" | rolling_enabled |
| "让利润奔跑" | tp_rr_ratio↑, trailing_enabled=true |
