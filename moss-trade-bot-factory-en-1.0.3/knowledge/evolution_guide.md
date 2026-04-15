# Complete Evolution Guide

## Core Idea: Evolution = Reflect While Backtesting

Evolution is not a separate step after backtesting. It is embedded inside the backtest process:

```text
Week 1 data -> backtest with initial parameters -> analyze result -> adjust tactical parameters
    ↓
Week 2 data -> backtest with adjusted parameters -> analyze result -> adjust again
    ↓
Week 3 data -> ... and so on until all data is consumed
```

## Segment-level context you must inspect before tuning

Each segment inside `evolution_log` contains:

- `segment_result.exit_reasons` -> counts of exit causes (`stop_loss / trailing_stop / take_profit / signal_reverse`)
- `segment_result.avg_win_pct / avg_loss_pct` -> average win / loss percentage
- `segment_result.longs / shorts` -> long / short direction distribution
- `market_context` -> BTC price trend and regime for this segment
- `cumulative_context` -> cumulative return, peak, peak drawdown, cumulative win rate, recent 3-segment performance
- `recent_trades` -> the latest 8 trade details (direction / price / PnL / exit reason)

**Do not look only at `total_return`.** Typical judgments:
- If stop losses account for more than 80% -> consider widening `sl_atr_mult`
- If all stop losses are on the same side -> `long_bias` may conflict with the market regime, but you still cannot change it; adjust `entry_threshold` instead
- If average win is larger than average loss but win rate is low -> the structure is healthy; do not overreact

## The 7 Reflection Principles

1. **See the big picture before the details** -> if cumulative return is still positive, the core direction may still be right, so a losing week may only be noise
2. **Analyze which trades made money and why** -> was the trend read correctly, was the stop placement good, did rolling amplify the winner
3. **Analyze which trades lost money and why** -> were stops too tight, was direction wrong, was the entry threshold too low and noisy
4. **Identify the exact parameter problem** -> do not stay vague; say something like `sl_atr_mult=1.5 is too tight, widen it to 2.0`
5. **Fine-tune, do not redesign** -> you are optimizing, not rebuilding. A single parameter adjustment should not exceed 10% of its initial value in one round
6. **Respect inertia** -> if a previous adjustment has not had enough time to show its effect (`<2` segments), keep it unchanged or only tweak it slightly
7. **Do not go more than 3 rounds without any adjustment** -> markets change. If nothing was changed for 3 consecutive segments, you must at least tweak one parameter slightly to keep the strategy adaptive

## Hard Constraints (Already Enforced In Code)

### Personality parameters never change
`long_bias`, `base_leverage`, `max_leverage`, `risk_per_trade`, `max_position_pct`, all `rolling_*`, and all signal weights. Even if you modify them in the evolution plan, code forces them back.

### Tactical parameters have a drift cap
Each tactical parameter may not drift more than `±30%` away from its initial value. For example, if initial `entry_threshold=0.32`, the maximum range is `0.22~0.42`. Code clamps automatically.

### Parameters that may be adjusted
`entry_threshold`, `exit_threshold`, `sl_atr_mult`, `tp_rr_ratio`, `trailing_activation_pct`, `trailing_distance_atr`, `regime_sensitivity`, `exit_on_regime_change`, `supertrend_mult`, `trend_strength_min`, `fast_ma_period`, `slow_ma_period`, `rsi_period`, `rsi_overbought`, `rsi_oversold`

## How to compute `segment-bars`

| Timeframe | Evolution Frequency | segment-bars |
|-----------|---------------------|--------------|
| 15m | weekly | 672 |
| 15m | daily | 96 |
| 1h | weekly | 168 |
| 1h | daily | 24 |
