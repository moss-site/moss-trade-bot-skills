"""
行情Regime识别模块

支持3个版本的检测器:
- v1: ADX + MA（经典趋势识别）
- v2: ATR + 波动率（波动率驱动）
- v3: 多指标综合投票

输出标准化的3种regime: BULL / BEAR / SIDEWAYS
"""

import pandas as pd
import numpy as np
from core.indicators import ema, atr, adx


class Regime:
    BULL = "BULL"
    BEAR = "BEAR"
    SIDEWAYS = "SIDEWAYS"

    ALL = [BULL, BEAR, SIDEWAYS]


def classify_regime(
    df: pd.DataFrame,
    version: str = "v1",
    window: int = 48,
    min_duration: int = 0,
) -> pd.Series:
    """
    对每根K线标记其所处的行情regime。

    Args:
        df: OHLCV DataFrame
        version: 检测器版本 (v1/v2/v3)
        window: 评估窗口大小
        min_duration: 最小regime持续K线数（0=不平滑）

    Returns:
        Series of regime labels (BULL/BEAR/SIDEWAYS)
    """
    if version == "v1":
        raw = _classify_v1(df, window)
    elif version == "v2":
        raw = _classify_v2(df, window)
    elif version == "v3":
        raw = _classify_v3(df, window)
    else:
        raw = _classify_v1(df, window)

    if min_duration > 0:
        return _smooth_regime(raw, min_duration)
    return raw


def _smooth_regime(regime: pd.Series, min_duration: int) -> pd.Series:
    """Regime 平滑：新 regime 必须持续 min_duration 根K线才正式切换。"""
    smoothed = regime.copy()
    current = regime.iloc[0]
    pending = None
    pending_count = 0

    for i in range(1, len(regime)):
        raw = regime.iloc[i]
        if raw == current:
            pending = None
            pending_count = 0
            smoothed.iloc[i] = current
        elif raw == pending:
            pending_count += 1
            if pending_count >= min_duration:
                current = pending
                pending = None
                pending_count = 0
            smoothed.iloc[i] = current
        else:
            pending = raw
            pending_count = 1
            smoothed.iloc[i] = current

    return smoothed


def _classify_v1(df: pd.DataFrame, window: int = 48) -> pd.Series:
    """v1: ADX + MA 经典趋势识别"""
    df = df.copy()
    df["ema_50"] = ema(df, 50)
    df["ema_slope"] = df["ema_50"].pct_change(window)
    adx_val, _, _ = adx(df, 14)
    df["adx"] = adx_val
    df["return"] = df["close"].pct_change(window)

    regime = pd.Series("", index=df.index)
    for i in range(window, len(df)):
        slope = df["ema_slope"].iloc[i]
        adx_v = df["adx"].iloc[i]
        ret = df["return"].iloc[i]

        if adx_v is not None and adx_v > 25:
            if slope > 0.01:
                regime.iloc[i] = Regime.BULL
            elif slope < -0.01:
                regime.iloc[i] = Regime.BEAR
            else:
                regime.iloc[i] = Regime.SIDEWAYS
        else:
            if abs(ret) > 0.05:
                regime.iloc[i] = Regime.BULL if ret > 0 else Regime.BEAR
            else:
                regime.iloc[i] = Regime.SIDEWAYS

    return regime.replace("", np.nan).ffill().fillna(Regime.SIDEWAYS)


def _classify_v2(df: pd.DataFrame, window: int = 48) -> pd.Series:
    """v2: ATR + 波动率驱动"""
    df = df.copy()
    atr_val = atr(df, 14)
    df["atr_pct"] = atr_val / df["close"]
    df["atr_rank"] = df["atr_pct"].rolling(window * 4, min_periods=window).rank(pct=True)
    df["return"] = df["close"].pct_change(window)
    df["vol"] = df["close"].pct_change().rolling(window).std()
    df["vol_rank"] = df["vol"].rolling(window * 4, min_periods=window).rank(pct=True)

    regime = pd.Series("", index=df.index)
    for i in range(window, len(df)):
        vol_r = df["vol_rank"].iloc[i]
        ret = df["return"].iloc[i]

        if vol_r is not None and vol_r < 0.3:
            regime.iloc[i] = Regime.SIDEWAYS
        elif ret > 0.03:
            regime.iloc[i] = Regime.BULL
        elif ret < -0.03:
            regime.iloc[i] = Regime.BEAR
        else:
            regime.iloc[i] = Regime.SIDEWAYS

    return regime.replace("", np.nan).ffill().fillna(Regime.SIDEWAYS)


def _classify_v3(df: pd.DataFrame, window: int = 48) -> pd.Series:
    """v3: 多指标综合投票（ADX + ATR + MA + 动量）"""
    df = df.copy()
    df["ema_20"] = ema(df, 20)
    df["ema_50"] = ema(df, 50)
    adx_val, plus_di, minus_di = adx(df, 14)
    df["adx"] = adx_val
    df["plus_di"] = plus_di
    df["minus_di"] = minus_di
    atr_val = atr(df, 14)
    df["atr_pct"] = atr_val / df["close"]
    df["atr_rank"] = df["atr_pct"].rolling(window * 4, min_periods=window).rank(pct=True)
    df["momentum"] = df["close"].pct_change(window)

    regime = pd.Series("", index=df.index)
    for i in range(window, len(df)):
        votes = {"BULL": 0, "BEAR": 0, "SIDEWAYS": 0}

        if df["ema_20"].iloc[i] > df["ema_50"].iloc[i]:
            votes["BULL"] += 1
        elif df["ema_20"].iloc[i] < df["ema_50"].iloc[i]:
            votes["BEAR"] += 1
        else:
            votes["SIDEWAYS"] += 1

        adx_v = df["adx"].iloc[i]
        if adx_v is not None and adx_v > 25:
            if df["plus_di"].iloc[i] > df["minus_di"].iloc[i]:
                votes["BULL"] += 1
            else:
                votes["BEAR"] += 1
        else:
            votes["SIDEWAYS"] += 1

        atr_r = df["atr_rank"].iloc[i]
        if atr_r is not None and atr_r < 0.3:
            votes["SIDEWAYS"] += 1

        mom = df["momentum"].iloc[i]
        if mom > 0.05:
            votes["BULL"] += 1
        elif mom < -0.05:
            votes["BEAR"] += 1
        else:
            votes["SIDEWAYS"] += 1

        regime.iloc[i] = max(votes, key=votes.get)

    return regime.replace("", np.nan).ffill().fillna(Regime.SIDEWAYS)


def get_regime_segments(df: pd.DataFrame, regime: pd.Series) -> list[dict]:
    """将regime序列切割成连续段落。"""
    segments = []
    current_regime = regime.iloc[0]
    start_idx = 0

    for i in range(1, len(regime)):
        if regime.iloc[i] != current_regime:
            segments.append({
                "regime": current_regime,
                "start_idx": start_idx,
                "end_idx": i - 1,
                "duration": i - start_idx,
                "start_time": df["timestamp"].iloc[start_idx] if "timestamp" in df.columns else None,
                "end_time": df["timestamp"].iloc[i - 1] if "timestamp" in df.columns else None,
            })
            current_regime = regime.iloc[i]
            start_idx = i

    segments.append({
        "regime": current_regime,
        "start_idx": start_idx,
        "end_idx": len(regime) - 1,
        "duration": len(regime) - start_idx,
        "start_time": df["timestamp"].iloc[start_idx] if "timestamp" in df.columns else None,
        "end_time": df["timestamp"].iloc[-1] if "timestamp" in df.columns else None,
    })

    return segments


def regime_summary(df: pd.DataFrame, regime: pd.Series) -> dict:
    """统计各regime占比和特征。"""
    summary = {}
    for r in Regime.ALL:
        mask = regime == r
        count = mask.sum()
        if count > 0:
            sub = df[mask]
            summary[r] = {
                "count": int(count),
                "pct": count / len(regime),
                "avg_return": float(sub["close"].pct_change().mean()),
                "volatility": float(sub["close"].pct_change().std()),
            }
    return summary
