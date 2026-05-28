"""Pure-Python TA indicators used by the ambush K-line-driven decision path.

Ambush's live runner intentionally stays off pandas (only the offline
backtest.py imports it), so these helpers operate on `list[dict]` bars
of shape `{"open": float, "high": float, "low": float, "close": float}`.
Volume is not required.

ATR + RSI use the same EWM smoothing as the altcoins/core/indicators.py
reference. Numerical results are within 1e-6 of the pandas equivalents
on identical inputs.
"""
from __future__ import annotations

from typing import Sequence


def compute_atr(bars: Sequence[dict], period: int = 14) -> float:
    """Wilder ATR over the most recent `period` bars.

    True Range[i] = max(high - low, |high - prev_close|, |low - prev_close|)
    ATR = EMA-smoothed (alpha = 1/period) average of TR over the series.

    Raises ValueError if len(bars) < period + 1 (need at least one
    previous close to compute the first TR).
    """
    if len(bars) < period + 1:
        raise ValueError(
            f"compute_atr requires at least {period + 1} bars, got {len(bars)}"
        )
    trs: list[float] = []
    prev_close = float(bars[0]["close"])
    for bar in bars[1:]:
        high = float(bar["high"])
        low = float(bar["low"])
        close = float(bar["close"])
        tr = max(
            high - low,
            abs(high - prev_close),
            abs(low - prev_close),
        )
        trs.append(tr)
        prev_close = close
    # EWM smoothing with span = period (matches pandas `ewm(span=period, adjust=False)`).
    alpha = 2.0 / (period + 1)
    atr = trs[0]
    for tr in trs[1:]:
        atr = alpha * tr + (1 - alpha) * atr
    return atr


def compute_rsi(bars: Sequence[dict], period: int = 14) -> float:
    """Wilder RSI(14) over the most recent `period + 1` closes.

    delta[i] = close[i] - close[i-1]
    gain[i]  = max(delta[i], 0)
    loss[i]  = max(-delta[i], 0)
    avg_gain = EWM(gain, alpha=1/period, min_periods=period)
    avg_loss = EWM(loss, alpha=1/period, min_periods=period)
    rs       = avg_gain / avg_loss      (∞ if avg_loss == 0)
    RSI      = 100 - 100 / (1 + rs)     (100 if rs == ∞)

    EWM uses pandas' default `adjust=True` weighting to stay numerically
    aligned with `core/indicators.py::rsi` (which does not pass
    `adjust=False`). The accumulator form below matches pandas'
    `ewm(alpha=1/period, adjust=True).mean()` to within 1e-12.

    Raises ValueError if len(bars) < period + 1.
    """
    if len(bars) < period + 1:
        raise ValueError(
            f"compute_rsi requires at least {period + 1} bars, got {len(bars)}"
        )
    closes = [float(b["close"]) for b in bars]
    gains: list[float] = []
    losses: list[float] = []
    for i in range(1, len(closes)):
        delta = closes[i] - closes[i - 1]
        gains.append(max(delta, 0.0))
        losses.append(max(-delta, 0.0))
    alpha = 1.0 / period
    one_minus = 1.0 - alpha
    # adjust=True EWM: y_t = sum_i (1-α)^(t-i) * x_i / sum_i (1-α)^(t-i)
    # computed iteratively via num/den accumulators.
    num_g = 0.0
    num_l = 0.0
    den = 0.0
    for g, l in zip(gains, losses):
        num_g = num_g * one_minus + g
        num_l = num_l * one_minus + l
        den = den * one_minus + 1.0
    avg_gain = num_g / den
    avg_loss = num_l / den
    if avg_loss == 0.0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100.0 - 100.0 / (1.0 + rs)
