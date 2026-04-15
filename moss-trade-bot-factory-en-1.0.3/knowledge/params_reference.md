# Complete Parameter Reference

## Decision Function

```text
Trading signal = trend_weight×trend + momentum_weight×momentum + mean_revert_weight×mean reversion + volume_weight×volume + volatility_weight×volatility
signal > entry_threshold -> open long | signal < -entry_threshold -> open short
```

## Signal Weights (5 total, auto-normalized so the sum = 1)

| Parameter | Meaning |
|-----------|---------|
| trend_weight | Trend-following. Higher = more emphasis on EMA crossover and Supertrend direction |
| momentum_weight | Momentum. Higher = more emphasis on RSI and MACD signals |
| mean_revert_weight | Mean reversion. Higher = more emphasis on Bollinger-band reversion |
| volume_weight | Volume. Higher = more emphasis on OBV and price-volume confirmation |
| volatility_weight | Volatility. Higher = more emphasis on ATR expansion / contraction |

## Trading Thresholds (composite signal usually ranges around `-0.5 ~ +0.5`)

| Parameter | Range | Meaning |
|-----------|-------|---------|
| entry_threshold | `0.05~0.55` | Lower = easier to trigger. `0.15` aggressive, `0.25` neutral, `0.40` conservative, `>0.5` almost never triggers |
| exit_threshold | `0.03~0.30` | Close when the reverse signal exceeds this value while a position is open |

## Direction Bias

| Parameter | Range | Meaning |
|-----------|-------|---------|
| long_bias | `0~1` | `0` = short only, `0.5` = bi-directional, `1` = long only |

## Technical Parameters

| Parameter | Range | Meaning |
|-----------|-------|---------|
| fast_ma_period | `5~50` | Fast moving-average period |
| slow_ma_period | `20~200` | Slow moving-average period (must be `> fast_ma_period`) |
| trend_strength_min | `10~50` | ADX trend-strength threshold |
| supertrend_mult | `1~5` | Supertrend multiplier |
| rsi_period | `7~28` | RSI period |
| rsi_overbought | `60~85` | RSI overbought line |
| rsi_oversold | `15~40` | RSI oversold line |
| bb_period | `10~50` | Bollinger-band period |
| bb_std | `1.0~3.0` | Bollinger-band standard-deviation multiplier |

## Leverage And Position Sizing

| Parameter | Range | Meaning |
|-----------|-------|---------|
| base_leverage | `1~150` | Base leverage |
| max_leverage | `1~150` | Maximum leverage |
| risk_per_trade | `0.01~0.50` | Fraction of capital used per trade |
| max_position_pct | `0.05~1.0` | Maximum capital share for a single position |

## Stop Loss / Take Profit

**Core formula**: `leverage × ATR percentage ≈ max single-trade loss %`

| Leverage | Recommended `sl_atr_mult` |
|----------|---------------------------|
| 5x | 1.0 |
| 10x | 2.0 |
| 20x | 2.5~3.0 |
| 50x | 3.0~5.0 |

| Parameter | Range | Meaning |
|-----------|-------|---------|
| sl_atr_mult | `0.5~5.0` | Stop distance = `X × ATR`. **High leverage must use a wide stop** |
| tp_rr_ratio | `1.0~10.0` | Take-profit / stop-loss distance ratio (risk-reward ratio) |
| trailing_enabled | bool | Whether trailing stop is enabled |
| trailing_activation_pct | `0.01~0.10` | Activate trailing stop after floating profit reaches `X%` |
| trailing_distance_atr | `0.5~3.0` | Trailing stop distance from the high watermark = `X × ATR` |

## Rolling (Profit Amplifier For Trend Strategies)

Without rolling, PnL is roughly symmetric (`+25% / -25%`), so a 50% win rate does not make money. With rolling, winning trades can exceed `+100%`. **Strongly recommended for trend strategies.**

| Parameter | Range | Meaning |
|-----------|-------|---------|
| rolling_enabled | bool | Whether to enable rolling (adding with floating profit) |
| rolling_trigger_pct | `0.10~0.80` | Trigger when floating profit reaches `X%` |
| rolling_reinvest_pct | `0.30~1.0` | Use `X%` of floating profit as margin for the new layer |
| rolling_max_times | `1~5` | Maximum number of rolling additions |
| rolling_move_stop | bool | Move old stop to breakeven after rolling |

## Regime Sensitivity

| Parameter | Range | Meaning |
|-----------|-------|---------|
| regime_sensitivity | `0~1` | `0` = ignore regime, `1` = trade only when the regime matches strictly |
| exit_on_regime_change | bool | Whether to close immediately when the regime changes |

## Parameter Categories (Evolution Constraints)

- **Personality parameters (not adjustable)**: `long_bias`, `base/max_leverage`, `risk_per_trade`, `max_position_pct`, all `rolling_*`, all signal weights
- **Tactical parameters (micro-adjustable)**: `entry/exit_threshold`, `sl_atr_mult`, `tp_rr_ratio`, `trailing_*`, `regime_*`, and the technical parameters

## Quick Tuning Cheatsheet

| If the user says | Adjust these |
|------------------|--------------|
| "Stops are too tight / getting clipped too often" | `sl_atr_mult ↑` (for example `2.0 -> 2.8`) |
| "Stops are too loose / losses are too large" | `sl_atr_mult ↓` |
| "It trades too frequently" | `entry_threshold ↑` (for example `0.2 -> 0.35`) |
| "It trades too rarely" | `entry_threshold ↓` |
| "Leverage is too high / too low" | `base_leverage`, `max_leverage` |
| "Make it more aggressive" | `base_leverage ↑`, `risk_per_trade ↑` |
| "Make it more conservative" | `base_leverage ↓`, `entry_threshold ↑` |
| "Lean more long / more short" | `long_bias ↑ / ↓` |
| "Turn rolling on / off" | `rolling_enabled` |
| "Let profits run" | `tp_rr_ratio ↑`, `trailing_enabled=true` |
