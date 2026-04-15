"""
连续加权决策函数

替代原来的离散参数 + 硬编码规则路由。
所有参数都是连续值，LLM可以精细调节。

信号计算:
  composite = trend_w * trend_sig + momentum_w * momentum_sig
            + revert_w * revert_sig + volume_w * volume_sig
            + volatility_w * volatility_sig

  composite > entry_threshold → 开多
  composite < -entry_threshold → 开空
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field, asdict
from typing import Optional
import json

from core.indicators import (
    ema, sma, rsi, macd, bollinger_bands, atr, adx,
    stochastic, supertrend, obv, volume_spike,
)


@dataclass
class DecisionParams:
    """LLM 可调的连续参数集。每个字段都有明确语义。"""

    # --- 信号权重（5维，归一化到和=1）---
    trend_weight: float = 0.30
    momentum_weight: float = 0.25
    mean_revert_weight: float = 0.15
    volume_weight: float = 0.15
    volatility_weight: float = 0.15

    # --- 信号阈值 ---
    entry_threshold: float = 0.40    # |composite| > 此值才开仓
    exit_threshold: float = 0.10     # 持仓时 composite 反向超过此值平仓

    # --- 方向偏好 ---
    long_bias: float = 0.50          # 0=只做空  0.5=双向  1=只做多

    # --- 趋势参数 ---
    fast_ma_period: int = 10
    slow_ma_period: int = 50
    trend_strength_min: float = 25.0  # ADX 阈值
    supertrend_mult: float = 3.0

    # --- 动量参数 ---
    rsi_period: int = 14
    rsi_overbought: float = 70.0
    rsi_oversold: float = 30.0
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9

    # --- 均值回归参数 ---
    bb_period: int = 20
    bb_std: float = 2.0

    # --- 杠杆 & 仓位 ---
    base_leverage: float = 10.0
    max_leverage: float = 100.0
    risk_per_trade: float = 0.10      # 每笔用资金的比例
    max_position_pct: float = 0.50    # 单笔最大资金占比

    # --- 止损 / 止盈 ---
    sl_atr_mult: float = 2.0         # 止损 = entry ± sl_atr_mult * ATR
    tp_rr_ratio: float = 3.0         # 止盈 = 止损距离 * tp_rr_ratio
    trailing_enabled: bool = False
    trailing_activation_pct: float = 0.02  # 浮盈超过此比例激活移动止损
    trailing_distance_atr: float = 1.5     # 移动止损距高点 X * ATR

    # --- 滚仓 ---
    rolling_enabled: bool = False
    rolling_trigger_pct: float = 0.30      # 浮盈 30% 时触发滚仓
    rolling_reinvest_pct: float = 0.80     # 用 80% 浮盈作新仓保证金
    rolling_max_times: int = 3
    rolling_move_stop: bool = True         # 滚仓后老仓止损移到成本价

    # --- Regime 敏感度 ---
    regime_sensitivity: float = 0.5        # 0=忽略regime  1=严格按regime
    exit_on_regime_change: bool = True

    def normalize_weights(self):
        """归一化信号权重到和=1，并钳制所有参数到合理范围。"""
        total = (self.trend_weight + self.momentum_weight +
                 self.mean_revert_weight + self.volume_weight +
                 self.volatility_weight)
        if total > 0:
            self.trend_weight /= total
            self.momentum_weight /= total
            self.mean_revert_weight /= total
            self.volume_weight /= total
            self.volatility_weight /= total

        self.entry_threshold = np.clip(self.entry_threshold, 0.05, 0.60)
        self.exit_threshold = np.clip(self.exit_threshold, 0.02, 0.40)
        self.long_bias = np.clip(self.long_bias, 0.0, 1.0)
        self.regime_sensitivity = np.clip(self.regime_sensitivity, 0.0, 1.0)
        self.base_leverage = max(1.0, min(self.base_leverage, 150.0))
        self.max_leverage = max(self.base_leverage, min(self.max_leverage, 150.0))
        self.risk_per_trade = np.clip(self.risk_per_trade, 0.01, 1.0)
        self.sl_atr_mult = np.clip(self.sl_atr_mult, 0.3, 8.0)
        self.tp_rr_ratio = np.clip(self.tp_rr_ratio, 0.5, 10.0)

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self, indent=2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)

    @classmethod
    def from_dict(cls, d: dict) -> "DecisionParams":
        valid = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in d.items() if k in valid})


# ====== 子信号计算 (每个返回 -1.0 ~ +1.0) ======

def _trend_signal(df: pd.DataFrame, p: DecisionParams, i: int,
                  cache: dict) -> float:
    """趋势信号: EMA交叉 + ADX强度 + Supertrend方向 + DI方向"""
    if "ema_fast" not in cache:
        cache["ema_fast"] = ema(df, p.fast_ma_period)
        cache["ema_slow"] = ema(df, p.slow_ma_period)
        adx_val, plus_di, minus_di = adx(df, 14)
        cache["adx"] = adx_val
        cache["plus_di"] = plus_di
        cache["minus_di"] = minus_di
        _, st_dir = supertrend(df, period=max(p.fast_ma_period, 7),
                               multiplier=p.supertrend_mult)
        cache["st_dir"] = st_dir

    ef = cache["ema_fast"].iloc[i]
    es = cache["ema_slow"].iloc[i]
    adx_v = cache["adx"].iloc[i]
    st_d = cache["st_dir"].iloc[i]
    plus_di = cache["plus_di"].iloc[i]
    minus_di = cache["minus_di"].iloc[i]

    if np.isnan(ef) or np.isnan(es) or es == 0:
        return 0.0

    # EMA 方向：快线在慢线上方=多，下方=空
    ema_diff = (ef - es) / es
    ema_sig = np.clip(ema_diff * 80, -1, 1)

    # Supertrend 方向
    st_sig = 1.0 if st_d == 1 else -1.0

    # +DI/-DI 方向性强度
    di_sig = 0.0
    if not (np.isnan(plus_di) or np.isnan(minus_di)):
        di_diff = plus_di - minus_di
        di_sig = np.clip(di_diff / 30, -1, 1)

    # ADX 作为趋势可信度放大器（而非衰减器）
    # ADX > 25 = 有趋势，信号全额；ADX < 25 = 弱趋势，但仍保留 50%
    if not np.isnan(adx_v):
        confidence = 0.5 + 0.5 * min(adx_v / 30.0, 1.0)
    else:
        confidence = 0.5

    # 三票加权：EMA方向 40% + Supertrend 30% + DI方向 30%
    raw = ema_sig * 0.40 + st_sig * 0.30 + di_sig * 0.30
    return np.clip(raw * confidence, -1, 1)


def _momentum_signal(df: pd.DataFrame, p: DecisionParams, i: int,
                     cache: dict) -> float:
    """动量信号: RSI + MACD histogram"""
    if "rsi" not in cache:
        cache["rsi"] = rsi(df, p.rsi_period)
        ml, sl, hist = macd(df, p.macd_fast, p.macd_slow, p.macd_signal)
        cache["macd_hist"] = hist
        cache["macd_line"] = ml

    rsi_v = cache["rsi"].iloc[i]
    hist_v = cache["macd_hist"].iloc[i]

    if np.isnan(rsi_v):
        return 0.0

    # RSI: 50为中性，超买超卖区给出强信号
    rsi_sig = (rsi_v - 50) / 50  # -1 ~ +1
    rsi_sig = np.clip(rsi_sig * 1.5, -1, 1)

    # MACD histogram 归一化
    price = df["close"].iloc[i]
    if price > 0 and not np.isnan(hist_v):
        macd_sig = np.clip(hist_v / (price * 0.002), -1, 1)
    else:
        macd_sig = 0.0

    return np.clip(rsi_sig * 0.5 + macd_sig * 0.5, -1, 1)


def _mean_revert_signal(df: pd.DataFrame, p: DecisionParams, i: int,
                        cache: dict) -> float:
    """均值回归信号: 布林带位置"""
    if "bb_upper" not in cache:
        cache["bb_upper"], cache["bb_mid"], cache["bb_lower"] = \
            bollinger_bands(df, p.bb_period, p.bb_std)

    price = df["close"].iloc[i]
    upper = cache["bb_upper"].iloc[i]
    lower = cache["bb_lower"].iloc[i]
    mid = cache["bb_mid"].iloc[i]

    if np.isnan(upper) or upper == lower:
        return 0.0

    # 价格在 BB 中的位置: -1(下轨外) ~ 0(中轨) ~ +1(上轨外)
    position = (price - mid) / ((upper - lower) / 2)

    # 均值回归: 价格越偏离，反向信号越强
    return np.clip(-position * 0.8, -1, 1)


def _volume_signal(df: pd.DataFrame, p: DecisionParams, i: int,
                   cache: dict) -> float:
    """成交量信号: OBV 趋势 + 量价配合"""
    if "obv" not in cache:
        cache["obv"] = obv(df)
        cache["obv_ema"] = cache["obv"].ewm(span=20, adjust=False).mean()
        cache["vol_ma"] = df["volume"].rolling(20).mean()

    obv_v = cache["obv"].iloc[i]
    obv_e = cache["obv_ema"].iloc[i]
    vol = df["volume"].iloc[i]
    vol_ma = cache["vol_ma"].iloc[i]

    if np.isnan(obv_e) or obv_e == 0 or np.isnan(vol_ma) or vol_ma == 0:
        return 0.0

    # OBV 趋势 (OBV > 其EMA = 多头量能)
    obv_sig = np.clip((obv_v - obv_e) / abs(obv_e) * 10, -1, 1)

    # 量能放大系数
    vol_ratio = vol / vol_ma
    vol_boost = np.clip((vol_ratio - 1) * 0.5, 0, 0.5)

    price_dir = 1 if df["close"].iloc[i] > df["close"].iloc[max(0, i-1)] else -1
    return np.clip(obv_sig * 0.7 + price_dir * vol_boost * 0.3, -1, 1)


def _volatility_signal(df: pd.DataFrame, p: DecisionParams, i: int,
                       cache: dict) -> float:
    """波动率信号: ATR变化判断突破/收缩"""
    if "atr" not in cache:
        cache["atr"] = atr(df, 14)
        cache["atr_ma"] = cache["atr"].rolling(50, min_periods=14).mean()

    atr_v = cache["atr"].iloc[i]
    atr_m = cache["atr_ma"].iloc[i]

    if np.isnan(atr_v) or np.isnan(atr_m) or atr_m == 0:
        return 0.0

    # ATR 扩张 → 突破信号放大；收缩 → 信号减弱
    expansion = (atr_v - atr_m) / atr_m
    return np.clip(expansion, -1, 1)


# ====== 主决策函数 ======

def compute_signals(
    df: pd.DataFrame,
    params: DecisionParams,
    regime: pd.Series,
) -> pd.Series:
    """
    加权决策函数：计算每根K线的交易信号。

    Returns:
        pd.Series: 1=做多, -1=做空, 0=不操作
    """
    params.normalize_weights()
    n = len(df)
    signals = pd.Series(0, index=df.index)
    cache = {}

    for i in range(max(params.slow_ma_period, 50), n):
        trend = _trend_signal(df, params, i, cache)
        momentum = _momentum_signal(df, params, i, cache)
        revert = _mean_revert_signal(df, params, i, cache)
        volume = _volume_signal(df, params, i, cache)
        volatility = _volatility_signal(df, params, i, cache)

        composite = (
            params.trend_weight * trend +
            params.momentum_weight * momentum +
            params.mean_revert_weight * revert +
            params.volume_weight * volume +
            params.volatility_weight * volatility
        )

        # 方向偏好过滤
        if params.long_bias > 0.7 and composite < 0:
            composite *= (1 - params.long_bias) * 2
        elif params.long_bias < 0.3 and composite > 0:
            composite *= params.long_bias * 2

        # Regime 调制：只削弱逆势信号，不削弱顺势
        if regime is not None and params.regime_sensitivity > 0:
            r = regime.iloc[i] if i < len(regime) else "SIDEWAYS"
            s = params.regime_sensitivity
            if r == "BULL":
                if composite < 0:
                    composite *= (1 - s)       # 牛市做空信号削弱
                else:
                    composite *= (1 + s * 0.3) # 牛市做多信号小幅增强
            elif r == "BEAR":
                if composite > 0:
                    composite *= (1 - s)       # 熊市做多信号削弱
                else:
                    composite *= (1 + s * 0.3) # 熊市做空信号小幅增强

        if composite > params.entry_threshold:
            signals.iloc[i] = 1
        elif composite < -params.entry_threshold:
            signals.iloc[i] = -1

    return signals
