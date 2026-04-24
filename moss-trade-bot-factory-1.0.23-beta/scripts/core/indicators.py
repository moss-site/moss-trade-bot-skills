"""
技术指标计算模块

支持所有schema中定义的指标类型，统一接口。
每个指标函数接收DataFrame，返回添加了指标列的DataFrame。
"""

import pandas as pd
import numpy as np
from typing import Optional


def ema(df: pd.DataFrame, period: int, col: str = "close") -> pd.Series:
    return df[col].ewm(span=period, adjust=False).mean()


def sma(df: pd.DataFrame, period: int, col: str = "close") -> pd.Series:
    return df[col].rolling(window=period).mean()


def rsi(df: pd.DataFrame, period: int = 14, col: str = "close") -> pd.Series:
    delta = df[col].diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def macd(df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9,
         col: str = "close") -> tuple[pd.Series, pd.Series, pd.Series]:
    """返回 (macd_line, signal_line, histogram)"""
    ema_fast = df[col].ewm(span=fast, adjust=False).mean()
    ema_slow = df[col].ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def bollinger_bands(df: pd.DataFrame, period: int = 20, std_dev: float = 2.0,
                    col: str = "close") -> tuple[pd.Series, pd.Series, pd.Series]:
    """返回 (upper, middle, lower)"""
    middle = df[col].rolling(window=period).mean()
    std = df[col].rolling(window=period).std()
    upper = middle + std_dev * std
    lower = middle - std_dev * std
    return upper, middle, lower


def bollinger_width(df: pd.DataFrame, period: int = 20, std_dev: float = 2.0,
                    col: str = "close") -> pd.Series:
    """布林带宽度 - 用于判断squeeze"""
    upper, middle, lower = bollinger_bands(df, period, std_dev, col)
    return (upper - lower) / middle


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high = df["high"]
    low = df["low"]
    close = df["close"].shift(1)
    tr = pd.concat([
        high - low,
        (high - close).abs(),
        (low - close).abs()
    ], axis=1).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean()


def adx(df: pd.DataFrame, period: int = 14) -> tuple[pd.Series, pd.Series, pd.Series]:
    """返回 (adx, plus_di, minus_di)"""
    high = df["high"]
    low = df["low"]
    close = df["close"]

    plus_dm = high.diff()
    minus_dm = -low.diff()
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)

    atr_val = atr(df, period)
    plus_di = 100 * (plus_dm.ewm(span=period, adjust=False).mean() / atr_val)
    minus_di = 100 * (minus_dm.ewm(span=period, adjust=False).mean() / atr_val)

    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    adx_val = dx.ewm(span=period, adjust=False).mean()
    return adx_val, plus_di, minus_di


def stochastic(df: pd.DataFrame, k_period: int = 14, d_period: int = 3) -> tuple[pd.Series, pd.Series]:
    """返回 (%K, %D)"""
    low_min = df["low"].rolling(window=k_period).min()
    high_max = df["high"].rolling(window=k_period).max()
    k = 100 * (df["close"] - low_min) / (high_max - low_min).replace(0, np.nan)
    d = k.rolling(window=d_period).mean()
    return k, d


def supertrend(df: pd.DataFrame, period: int = 10, multiplier: float = 3.0) -> tuple[pd.Series, pd.Series]:
    """返回 (supertrend_line, direction) direction: 1=上升趋势, -1=下降趋势"""
    atr_val = atr(df, period)
    hl2 = (df["high"] + df["low"]) / 2
    upper_band = hl2 + multiplier * atr_val
    lower_band = hl2 - multiplier * atr_val

    st = pd.Series(0.0, index=df.index)
    direction = pd.Series(1, index=df.index)

    for i in range(1, len(df)):
        if df["close"].iloc[i] > upper_band.iloc[i - 1]:
            direction.iloc[i] = 1
        elif df["close"].iloc[i] < lower_band.iloc[i - 1]:
            direction.iloc[i] = -1
        else:
            direction.iloc[i] = direction.iloc[i - 1]

        if direction.iloc[i] == 1:
            st.iloc[i] = max(lower_band.iloc[i], st.iloc[i - 1]) if direction.iloc[i - 1] == 1 else lower_band.iloc[i]
        else:
            st.iloc[i] = min(upper_band.iloc[i], st.iloc[i - 1]) if direction.iloc[i - 1] == -1 else upper_band.iloc[i]

    return st, direction


def vwap(df: pd.DataFrame) -> pd.Series:
    """成交量加权平均价"""
    typical_price = (df["high"] + df["low"] + df["close"]) / 3
    cumvol = df["volume"].cumsum()
    cumtp = (typical_price * df["volume"]).cumsum()
    return cumtp / cumvol.replace(0, np.nan)


def obv(df: pd.DataFrame) -> pd.Series:
    """能量潮"""
    sign = np.sign(df["close"].diff())
    return (sign * df["volume"]).cumsum()


def volume_spike(df: pd.DataFrame, period: int = 20, threshold: float = 2.0) -> pd.Series:
    """成交量是否为均量的threshold倍 (布尔)"""
    avg_vol = df["volume"].rolling(window=period).mean()
    return df["volume"] / avg_vol.replace(0, np.nan)


def price_breakout(df: pd.DataFrame, period: int = 20) -> tuple[pd.Series, pd.Series]:
    """返回 (breakout_high, breakout_low) 布尔序列"""
    high_max = df["high"].rolling(window=period).max().shift(1)
    low_min = df["low"].rolling(window=period).min().shift(1)
    return df["close"] > high_max, df["close"] < low_min


def keltner_channels(df: pd.DataFrame, ema_period: int = 20, atr_period: int = 10,
                     multiplier: float = 2.0) -> tuple[pd.Series, pd.Series, pd.Series]:
    """返回 (upper, middle, lower)"""
    middle = ema(df, ema_period)
    atr_val = atr(df, atr_period)
    upper = middle + multiplier * atr_val
    lower = middle - multiplier * atr_val
    return upper, middle, lower


def ichimoku(df: pd.DataFrame, tenkan: int = 9, kijun: int = 26,
             senkou_b: int = 52) -> dict[str, pd.Series]:
    """返回一云图各组件"""
    high = df["high"]
    low = df["low"]

    tenkan_sen = (high.rolling(tenkan).max() + low.rolling(tenkan).min()) / 2
    kijun_sen = (high.rolling(kijun).max() + low.rolling(kijun).min()) / 2
    senkou_a = ((tenkan_sen + kijun_sen) / 2).shift(kijun)
    senkou_b_val = ((high.rolling(senkou_b).max() + low.rolling(senkou_b).min()) / 2).shift(kijun)
    chikou = df["close"].shift(-kijun)

    return {
        "tenkan_sen": tenkan_sen,
        "kijun_sen": kijun_sen,
        "senkou_a": senkou_a,
        "senkou_b": senkou_b_val,
        "chikou": chikou,
    }


def donchian_channel(df: pd.DataFrame, period: int = 20) -> tuple[pd.Series, pd.Series, pd.Series]:
    """唐奇安通道 (upper, middle, lower)"""
    upper = df["high"].rolling(period).max()
    lower = df["low"].rolling(period).min()
    middle = (upper + lower) / 2
    return upper, middle, lower


def williams_r(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """威廉指标"""
    high_max = df["high"].rolling(period).max()
    low_min = df["low"].rolling(period).min()
    return -100 * (high_max - df["close"]) / (high_max - low_min).replace(0, np.nan)


def cci(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """商品通道指数"""
    tp = (df["high"] + df["low"] + df["close"]) / 3
    sma_tp = tp.rolling(period).mean()
    mad = tp.rolling(period).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True)
    return (tp - sma_tp) / (0.015 * mad).replace(0, np.nan)


def mfi(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """资金流量指数"""
    tp = (df["high"] + df["low"] + df["close"]) / 3
    mf = tp * df["volume"]
    delta = tp.diff()
    pos_mf = mf.where(delta > 0, 0.0).rolling(period).sum()
    neg_mf = mf.where(delta < 0, 0.0).rolling(period).sum()
    ratio = pos_mf / neg_mf.replace(0, np.nan)
    return 100 - (100 / (1 + ratio))


# ============ 统一计算接口 ============

def compute_indicator(df: pd.DataFrame, indicator_type: str, params: dict) -> dict:
    """
    统一指标计算接口
    返回 dict[str, pd.Series]，key为指标列名
    """
    result = {}

    if indicator_type == "ema":
        period = params.get("period", 20)
        result[f"ema_{period}"] = ema(df, period)

    elif indicator_type == "sma":
        period = params.get("period", 20)
        result[f"sma_{period}"] = sma(df, period)

    elif indicator_type == "ema_cross":
        fast = params.get("fast", 12)
        slow = params.get("slow", 26)
        result[f"ema_{fast}"] = ema(df, fast)
        result[f"ema_{slow}"] = ema(df, slow)

    elif indicator_type == "sma_cross":
        fast = params.get("fast", 10)
        slow = params.get("slow", 30)
        result[f"sma_{fast}"] = sma(df, fast)
        result[f"sma_{slow}"] = sma(df, slow)

    elif indicator_type == "rsi":
        period = params.get("period", 14)
        result["rsi"] = rsi(df, period)

    elif indicator_type == "macd":
        fast = params.get("fast", 12)
        slow = params.get("slow", 26)
        signal = params.get("signal", 9)
        m, s, h = macd(df, fast, slow, signal)
        result["macd"] = m
        result["macd_signal"] = s
        result["macd_hist"] = h

    elif indicator_type == "bollinger":
        period = params.get("period", 20)
        std_dev = params.get("std_dev", 2.0)
        u, m, l = bollinger_bands(df, period, std_dev)
        result["bb_upper"] = u
        result["bb_middle"] = m
        result["bb_lower"] = l
        result["bb_width"] = bollinger_width(df, period, std_dev)

    elif indicator_type == "atr":
        period = params.get("period", 14)
        result["atr"] = atr(df, period)

    elif indicator_type == "adx":
        period = params.get("period", 14)
        a, p, m = adx(df, period)
        result["adx"] = a
        result["plus_di"] = p
        result["minus_di"] = m

    elif indicator_type == "stochastic":
        k_period = params.get("k_period", 14)
        d_period = params.get("d_period", 3)
        k, d = stochastic(df, k_period, d_period)
        result["stoch_k"] = k
        result["stoch_d"] = d

    elif indicator_type == "supertrend":
        period = params.get("period", 10)
        multiplier = params.get("multiplier", 3.0)
        st, direction = supertrend(df, period, multiplier)
        result["supertrend"] = st
        result["supertrend_dir"] = direction

    elif indicator_type == "vwap":
        result["vwap"] = vwap(df)

    elif indicator_type == "obv":
        result["obv"] = obv(df)

    elif indicator_type == "volume_spike":
        period = params.get("period", 20)
        result["vol_ratio"] = volume_spike(df, period)

    elif indicator_type == "price_breakout":
        period = params.get("period", 20)
        bh, bl = price_breakout(df, period)
        result["breakout_high"] = bh
        result["breakout_low"] = bl

    elif indicator_type == "keltner":
        ema_period = params.get("ema_period", 20)
        atr_period = params.get("atr_period", 10)
        multiplier = params.get("multiplier", 2.0)
        u, m, l = keltner_channels(df, ema_period, atr_period, multiplier)
        result["kc_upper"] = u
        result["kc_middle"] = m
        result["kc_lower"] = l

    elif indicator_type == "ichimoku":
        tenkan_p = params.get("tenkan", 9)
        kijun_p = params.get("kijun", 26)
        senkou_b_p = params.get("senkou_b", 52)
        ichi = ichimoku(df, tenkan_p, kijun_p, senkou_b_p)
        result.update(ichi)

    elif indicator_type == "donchian":
        period = params.get("period", 20)
        u, m, l = donchian_channel(df, period)
        result["dc_upper"] = u
        result["dc_middle"] = m
        result["dc_lower"] = l

    elif indicator_type == "williams_r":
        period = params.get("period", 14)
        result["williams_r"] = williams_r(df, period)

    elif indicator_type == "cci":
        period = params.get("period", 20)
        result["cci"] = cci(df, period)

    elif indicator_type == "mfi":
        period = params.get("period", 14)
        result["mfi"] = mfi(df, period)

    elif indicator_type == "support_resistance":
        period = params.get("period", 50)
        # 简单的支撑阻力：N周期高低点
        result["resistance"] = df["high"].rolling(period).max()
        result["support"] = df["low"].rolling(period).min()

    elif indicator_type == "candle_pattern":
        # 基础蜡烛图形态
        body = df["close"] - df["open"]
        body_abs = body.abs()
        upper_shadow = df["high"] - df[["open", "close"]].max(axis=1)
        lower_shadow = df[["open", "close"]].min(axis=1) - df["low"]
        # 锤子线：下影线>实体2倍, 上影线很小
        result["hammer"] = (lower_shadow > 2 * body_abs) & (upper_shadow < body_abs * 0.3)
        # 射击之星：上影线>实体2倍
        result["shooting_star"] = (upper_shadow > 2 * body_abs) & (lower_shadow < body_abs * 0.3)
        # 十字星：实体很小
        avg_body = body_abs.rolling(20).mean()
        result["doji"] = body_abs < avg_body * 0.1
        # 吞没形态
        result["bullish_engulf"] = (body > 0) & (body.shift(1) < 0) & (body_abs > body_abs.shift(1))
        result["bearish_engulf"] = (body < 0) & (body.shift(1) > 0) & (body_abs > body_abs.shift(1))

    return result
