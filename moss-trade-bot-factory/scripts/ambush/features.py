"""Convert a list of 15m K-line bars into the 3 floats balanced_decide_v0
expects: (surge_15m, rsi_14, chg_24h_pct).

Bar shape: dict {"open": float, "high": float, "low": float, "close": float}.
Bars must be ordered oldest-first; the last entry is "now".

96 bars × 15m = 24h, so the first bar in a 96-bar window is the 24h-ago
baseline used to compute chg_24h_pct.
"""
from __future__ import annotations

from typing import Sequence

from ambush import indicators


def compute_features(bars: Sequence[dict]) -> dict[str, float]:
    """Returns {"surge_15m": float, "rsi_14": float, "chg_24h_pct": float}.

    Raises ValueError if len(bars) is too small to compute any of the three.
    Minimum required: 15 bars (for RSI(14) + 14-bar TR window).
    """
    if len(bars) < 15:
        raise ValueError(
            f"compute_features requires at least 15 bars, got {len(bars)}"
        )

    last = bars[-1]
    last_open = float(last["open"])
    last_close = float(last["close"])
    # surge_15m = (close - open) / open. Single-bar percent move on the
    # bar that just closed (mirrors server detector dual_gate semantics).
    if last_open <= 0.0:
        surge_15m = 0.0
    else:
        surge_15m = (last_close - last_open) / last_open

    # chg_24h_pct uses bars[0] as the 24h-ago anchor when the window is
    # exactly 96 bars wide. For shorter windows we still use bars[0]
    # (caller responsibility to pass a full 96).
    first_close = float(bars[0]["close"])
    if first_close <= 0.0:
        chg_24h_pct = 0.0
    else:
        chg_24h_pct = (last_close - first_close) / first_close * 100.0

    rsi_14 = indicators.compute_rsi(bars, period=14)

    return {
        "surge_15m": surge_15m,
        "rsi_14": rsi_14,
        "chg_24h_pct": chg_24h_pct,
    }
