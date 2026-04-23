from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Optional

import pandas as pd

from core.decision import DecisionParams


REGIME_BULL = "BULL"
REGIME_BEAR = "BEAR"
REGIME_SIDEWAYS = "SIDEWAYS"
DEFAULT_MAINTENANCE_RATE = 0.004


@dataclass
class IncrementalBar:
    timestamp: pd.Timestamp
    open: float
    high: float
    low: float
    close: float
    volume: float


@dataclass
class IncrementalOpenPosition:
    direction: int
    entry_price: float
    mark_price: float
    leverage: float
    margin: float
    notional: float
    liquidation_price: float
    unrealized_pnl: float


@dataclass
class _SimPosition:
    direction: int
    entry_price: float
    leverage: float
    margin: float
    sl_price: float
    tp_price: float
    entry_time: int
    trailing_high: float = 0.0
    trailing_low: float = 0.0
    trailing_init: bool = False


def _clamp(value: float, min_v: float, max_v: float) -> float:
    if value < min_v:
        return min_v
    if value > max_v:
        return max_v
    return value


def _max_float(a: float, b: float) -> float:
    return a if a > b else b


def _min_float(a: float, b: float) -> float:
    return a if a < b else b


def _max_int(a: int, b: int) -> int:
    return a if a > b else b


def _ewm_update(curr: float, prev: float, span: int) -> float:
    if span <= 0:
        return math.nan
    alpha = 2.0 / (float(span) + 1.0)
    if math.isnan(prev):
        return curr
    return curr * alpha + prev * (1 - alpha)


def _ewm_update_nan(curr: float, prev: float, span: int) -> float:
    if math.isnan(curr):
        return prev
    return _ewm_update(curr, prev, span)


def _rsi_from_averages(avg_gain: float, avg_loss: float) -> float:
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))


def _position_qty(pos: _SimPosition) -> float:
    if pos.entry_price <= 0 or pos.leverage <= 0 or pos.margin <= 0:
        return 0.0
    return pos.margin * pos.leverage / pos.entry_price


def _position_unrealized_pnl(pos: _SimPosition, mark: float) -> float:
    qty = _position_qty(pos)
    if pos.direction > 0:
        return (mark - pos.entry_price) * qty
    return (pos.entry_price - mark) * qty


def _used_margin(positions: list[_SimPosition]) -> float:
    total = 0.0
    for pos in positions:
        if pos.margin > 0:
            total += pos.margin
    return total


def _account_equity(wallet_balance: float, positions: list[_SimPosition], mark: float) -> float:
    equity = wallet_balance
    for pos in positions:
        equity += _position_unrealized_pnl(pos, mark)
    return equity


def _free_margin(wallet_balance: float, positions: list[_SimPosition], mark: float) -> float:
    return _account_equity(wallet_balance, positions, mark) - _used_margin(positions)


def _unrealized_pnl_pct(direction: int, entry: float, current: float, leverage: float) -> float:
    if entry == 0:
        return 0.0
    if direction > 0:
        return (current - entry) / entry * leverage
    return (entry - current) / entry * leverage


def _book_liquidation_threshold(
    positions: list[_SimPosition],
    wallet_balance: float,
    maintenance_rate: float,
) -> tuple[float, str, bool]:
    signed_qty = 0.0
    abs_qty = 0.0
    signed_entry_notional = 0.0
    for pos in positions:
        qty = _position_qty(pos)
        if qty <= 0:
            continue
        sign = 1.0 if pos.direction > 0 else -1.0
        signed_qty += sign * qty
        abs_qty += qty
        signed_entry_notional += sign * pos.entry_price * qty
    slope = signed_qty - maintenance_rate * abs_qty
    if abs_qty <= 0 or abs(slope) <= 1e-12:
        return 0.0, "", False
    price = (signed_entry_notional - wallet_balance) / slope
    if math.isnan(price) or math.isinf(price) or price <= 0:
        return 0.0, "", False
    return price, ("down" if slope > 0 else "up"), True


class RollingStats:
    def __init__(self, size: int) -> None:
        self.size = size
        self.values: list[float] = []
        self.sum = 0.0
        self.sum_sq = 0.0

    def push(self, value: float) -> None:
        if self.size <= 0:
            return
        if len(self.values) == self.size:
            old = self.values.pop(0)
            self.sum -= old
            self.sum_sq -= old * old
        self.values.append(value)
        self.sum += value
        self.sum_sq += value * value

    def __len__(self) -> int:
        return len(self.values)

    def mean(self) -> float:
        if not self.values:
            return math.nan
        return self.sum / len(self.values)

    def sample_std(self) -> float:
        n = len(self.values)
        if n <= 1:
            return math.nan
        mean = self.mean()
        variance = (self.sum_sq - float(n) * mean * mean) / float(n - 1)
        if variance < 0 and variance > -1e-12:
            variance = 0.0
        if variance < 0:
            return math.nan
        return math.sqrt(variance)


class RollingHistory:
    def __init__(self, size: int) -> None:
        self.size = size
        self.values: list[float] = []

    def push(self, value: float) -> None:
        if self.size <= 0:
            return
        if len(self.values) == self.size:
            self.values.pop(0)
        self.values.append(value)

    def __len__(self) -> int:
        return len(self.values)

    def oldest(self) -> float:
        if not self.values:
            return math.nan
        return self.values[0]


class IncrementalIndicatorState:
    def __init__(
        self,
        fast_ma: int,
        slow_ma: int,
        rsi_period: int,
        macd_fast: int,
        macd_slow: int,
        macd_signal: int,
        bb_period: int,
        super_period: int,
        super_mult: float,
    ) -> None:
        self.fast_ma_period = fast_ma
        self.slow_ma_period = slow_ma
        self.rsi_period = rsi_period
        self.macd_fast = macd_fast
        self.macd_slow = macd_slow
        self.macd_signal = macd_signal
        self.bb_period = bb_period
        self.super_period = super_period
        self.super_mult = super_mult

        self.count = 0
        self.prev_close = 0.0
        self.prev_high = 0.0
        self.prev_low = 0.0
        self.have_prev = False

        self.ema_fast = math.nan
        self.ema_slow = math.nan
        self.ema50 = math.nan

        self.atr14 = math.nan
        self.plus_dm_ema14 = math.nan
        self.minus_dm_ema14 = math.nan
        self.plus_di = math.nan
        self.minus_di = math.nan
        self.adx = math.nan

        self.super_atr = math.nan
        self.super_prev_upper = math.nan
        self.super_prev_lower = math.nan
        self.super_line = math.nan
        self.super_dir = 1

        self.rsi_avg_gain = 0.0
        self.rsi_avg_loss = 0.0
        self.rsi_seed_gain = 0.0
        self.rsi_seed_loss = 0.0
        self.rsi_seed_count = 0
        self.rsi = math.nan

        self.macd_fast_ema = math.nan
        self.macd_slow_ema = math.nan
        self.macd_signal_ema = math.nan
        self.macd_hist = math.nan

        self.bb_closes = RollingStats(bb_period)
        self.bb_mid = math.nan
        self.bb_upper = math.nan
        self.bb_lower = math.nan

        self.obv = 0.0
        self.obv_ema = math.nan

        self.volumes = RollingStats(20)
        self.vol_ma = math.nan

        self.atr_values = RollingStats(50)
        self.atr_ma = math.nan

    def push(self, bar: IncrementalBar) -> None:
        tr = bar.high - bar.low
        plus_dm = 0.0
        minus_dm = 0.0
        price_up = False
        if self.have_prev:
            tr = _max_float(bar.high - bar.low, _max_float(abs(bar.high - self.prev_close), abs(bar.low - self.prev_close)))
            up_move = bar.high - self.prev_high
            down_move = self.prev_low - bar.low
            plus_dm = up_move
            minus_dm = down_move
            if not (plus_dm > minus_dm and plus_dm > 0):
                plus_dm = 0.0
            if not (minus_dm > plus_dm and minus_dm > 0):
                minus_dm = 0.0
            price_up = bar.close > self.prev_close

        self.ema_fast = _ewm_update(bar.close, self.ema_fast, self.fast_ma_period)
        self.ema_slow = _ewm_update(bar.close, self.ema_slow, self.slow_ma_period)
        self.ema50 = _ewm_update(bar.close, self.ema50, 50)
        self.atr14 = _ewm_update(tr, self.atr14, 14)
        self.plus_dm_ema14 = _ewm_update(plus_dm, self.plus_dm_ema14, 14)
        self.minus_dm_ema14 = _ewm_update(minus_dm, self.minus_dm_ema14, 14)
        if not math.isnan(self.atr14) and self.atr14 != 0:
            self.plus_di = 100.0 * (self.plus_dm_ema14 / self.atr14)
            self.minus_di = 100.0 * (self.minus_dm_ema14 / self.atr14)
            den = self.plus_di + self.minus_di
            dx = math.nan
            if den != 0 and not math.isnan(den):
                dx = 100.0 * abs(self.plus_di - self.minus_di) / den
            self.adx = _ewm_update_nan(dx, self.adx, 14)

        self.super_atr = _ewm_update(tr, self.super_atr, self.super_period)
        hl2 = (bar.high + bar.low) / 2.0
        prev_upper = self.super_prev_upper
        prev_lower = self.super_prev_lower
        prev_line = self.super_line
        prev_dir = self.super_dir
        upper = hl2 + self.super_mult * self.super_atr
        lower = hl2 - self.super_mult * self.super_atr
        if math.isnan(self.super_atr):
            upper = math.nan
            lower = math.nan

        if not self.have_prev:
            self.super_dir = 1
            self.super_line = math.nan
        else:
            delta = bar.close - self.prev_close
            gain = delta if delta > 0 else 0.0
            loss = -delta if delta < 0 else 0.0
            if self.rsi_seed_count < self.rsi_period:
                self.rsi_seed_gain += gain
                self.rsi_seed_loss += loss
                self.rsi_seed_count += 1
                if self.rsi_seed_count == self.rsi_period:
                    self.rsi_avg_gain = self.rsi_seed_gain / float(self.rsi_period)
                    self.rsi_avg_loss = self.rsi_seed_loss / float(self.rsi_period)
                    self.rsi = _rsi_from_averages(self.rsi_avg_gain, self.rsi_avg_loss)
            else:
                self.rsi_avg_gain = (self.rsi_avg_gain * float(self.rsi_period - 1) + gain) / float(self.rsi_period)
                self.rsi_avg_loss = (self.rsi_avg_loss * float(self.rsi_period - 1) + loss) / float(self.rsi_period)
                self.rsi = _rsi_from_averages(self.rsi_avg_gain, self.rsi_avg_loss)

            if not math.isnan(prev_upper) and not math.isnan(prev_lower):
                if bar.close > prev_upper:
                    self.super_dir = 1
                elif bar.close < prev_lower:
                    self.super_dir = -1
                else:
                    self.super_dir = prev_dir
                if self.super_dir == 1:
                    if prev_dir == 1 and not math.isnan(prev_line):
                        self.super_line = _max_float(lower, prev_line)
                    else:
                        self.super_line = lower
                else:
                    if prev_dir == -1 and not math.isnan(prev_line):
                        self.super_line = _min_float(upper, prev_line)
                    else:
                        self.super_line = upper
            else:
                self.super_dir = prev_dir

        self.macd_fast_ema = _ewm_update(bar.close, self.macd_fast_ema, self.macd_fast)
        self.macd_slow_ema = _ewm_update(bar.close, self.macd_slow_ema, self.macd_slow)
        macd_line = self.macd_fast_ema - self.macd_slow_ema
        self.macd_signal_ema = _ewm_update(macd_line, self.macd_signal_ema, self.macd_signal)
        self.macd_hist = macd_line - self.macd_signal_ema

        self.bb_closes.push(bar.close)
        self.bb_mid = math.nan
        self.bb_upper = math.nan
        self.bb_lower = math.nan
        if len(self.bb_closes) >= self.bb_period:
            self.bb_mid = self.bb_closes.mean()
            std = self.bb_closes.sample_std()
            if not math.isnan(std):
                self.bb_upper = self.bb_mid + 2 * std
                self.bb_lower = self.bb_mid - 2 * std

        if not self.have_prev:
            self.obv = 0.0
        elif price_up:
            self.obv += bar.volume
        elif bar.close < self.prev_close:
            self.obv -= bar.volume
        self.obv_ema = _ewm_update(self.obv, self.obv_ema, 20)

        self.volumes.push(bar.volume)
        self.vol_ma = math.nan
        if len(self.volumes) >= 20:
            self.vol_ma = self.volumes.mean()

        self.atr_values.push(self.atr14)
        self.atr_ma = math.nan
        if len(self.atr_values) >= 50:
            self.atr_ma = self.atr_values.mean()

        self.super_prev_upper = upper
        self.super_prev_lower = lower
        self.prev_close = bar.close
        self.prev_high = bar.high
        self.prev_low = bar.low
        self.have_prev = True
        self.count += 1


class IncrementalRegimeState:
    def __init__(self, window: int, min_duration: int) -> None:
        self.window = window
        self.min_duration = min_duration
        self.ema50_history = RollingHistory(window + 1)
        self.close_history = RollingHistory(window + 1)
        self.current = REGIME_SIDEWAYS
        self.pending = ""
        self.pending_count = 0
        self.index = 0

    def push(self, close_px: float, ema50: float, adx: float) -> str:
        self.ema50_history.push(ema50)
        self.close_history.push(close_px)
        raw = REGIME_SIDEWAYS
        if self.index >= self.window and len(self.ema50_history) > self.window and len(self.close_history) > self.window:
            prev_ema = self.ema50_history.oldest()
            prev_close = self.close_history.oldest()
            slope = 0.0
            if prev_ema != 0 and not math.isnan(prev_ema) and not math.isnan(ema50):
                slope = (ema50 - prev_ema) / prev_ema
            ret = 0.0
            if prev_close != 0:
                ret = (close_px - prev_close) / prev_close
            if not math.isnan(adx) and adx > 25:
                if slope > 0.01:
                    raw = REGIME_BULL
                elif slope < -0.01:
                    raw = REGIME_BEAR
            elif abs(ret) > 0.05:
                raw = REGIME_BULL if ret > 0 else REGIME_BEAR

        if self.index == 0 or self.min_duration <= 1:
            self.current = raw
            self.index += 1
            return self.current
        if raw == self.current:
            self.pending = ""
            self.pending_count = 0
            self.index += 1
            return self.current
        if raw == self.pending:
            self.pending_count += 1
            if self.pending_count >= self.min_duration:
                self.current = self.pending
                self.pending = ""
                self.pending_count = 0
            self.index += 1
            return self.current
        self.pending = raw
        self.pending_count = 1
        self.index += 1
        return self.current


class IncrementalStrategyState:
    def __init__(self, initial_capital: float, maintenance_rate: float) -> None:
        self.initial_capital = initial_capital
        self.maintenance_rate = maintenance_rate
        self.wallet_balance = initial_capital
        self.total_deposited = initial_capital
        self.blowups = 0
        self.positions: list[_SimPosition] = []
        self.roll_count = 0
        self.prev_regime = REGIME_SIDEWAYS

    def advance(self, idx: int, bar: IncrementalBar, signal: int, regime: str, params: DecisionParams, atr: float) -> None:
        if idx == 0:
            self.prev_regime = regime
            return

        price = bar.close
        high = bar.high
        low = bar.low
        if math.isnan(atr) or atr <= 0:
            atr = price * 0.02

        if self.wallet_balance < self.initial_capital * 0.01 and not self.positions:
            self.blowups += 1
            self.wallet_balance = self.initial_capital
            self.total_deposited += self.initial_capital
            self.roll_count = 0

        liq_price, liq_dir, ok = _book_liquidation_threshold(self.positions, self.wallet_balance, self.maintenance_rate)
        if ok:
            triggered = (liq_dir == "down" and low <= liq_price) or (liq_dir == "up" and high >= liq_price)
            if triggered:
                total_pnl = 0.0
                for pos in self.positions:
                    pnl_pct = _unrealized_pnl_pct(pos.direction, pos.entry_price, liq_price, pos.leverage)
                    total_pnl += pos.margin * pnl_pct
                self.wallet_balance += total_pnl
                if self.wallet_balance < 0:
                    self.wallet_balance = 0.0
                self.positions = []

        closed_idx: set[int] = set()
        non_liq_closed = 0
        for pos_idx, pos in enumerate(self.positions):
            exit_price = 0.0
            exit_reason = ""
            if pos.direction == 1 and low <= pos.sl_price:
                exit_price = pos.sl_price
                exit_reason = "stop_loss"
            if pos.direction == -1 and high >= pos.sl_price:
                exit_price = pos.sl_price
                exit_reason = "stop_loss"

            if exit_price == 0.0:
                if pos.direction == 1 and high >= pos.tp_price:
                    exit_price = pos.tp_price
                    exit_reason = "take_profit"
                if pos.direction == -1 and low <= pos.tp_price:
                    exit_price = pos.tp_price
                    exit_reason = "take_profit"

            if exit_price == 0.0 and params.trailing_enabled:
                trail_dist = params.trailing_distance_atr * atr
                if pos.direction == 1:
                    if (not pos.trailing_init) or high > pos.trailing_high:
                        pos.trailing_high = high
                        pos.trailing_init = True
                    if pos.trailing_high > pos.entry_price * (1 + params.trailing_activation_pct):
                        trail_sl = pos.trailing_high - trail_dist
                        if low <= trail_sl:
                            exit_price = trail_sl
                            exit_reason = "trailing_stop"
                else:
                    if (not pos.trailing_init) or low < pos.trailing_low:
                        pos.trailing_low = low
                        pos.trailing_init = True
                    if pos.trailing_low < pos.entry_price * (1 - params.trailing_activation_pct):
                        trail_sl = pos.trailing_low + trail_dist
                        if high >= trail_sl:
                            exit_price = trail_sl
                            exit_reason = "trailing_stop"

            if exit_price == 0.0 and params.exit_on_regime_change and regime != self.prev_regime:
                exit_price = price
                exit_reason = "regime_change"

            if exit_price == 0.0 and signal != 0 and signal != pos.direction:
                exit_price = price
                exit_reason = "signal_reverse"

            if exit_price != 0.0:
                pnl_pct = _unrealized_pnl_pct(pos.direction, pos.entry_price, exit_price, pos.leverage)
                self.wallet_balance += pos.margin * pnl_pct
                if self.wallet_balance < 0:
                    self.wallet_balance = 0.0
                closed_idx.add(pos_idx)
                if exit_reason != "liquidation":
                    non_liq_closed += 1

        if closed_idx:
            self.positions = [pos for pos_idx, pos in enumerate(self.positions) if pos_idx not in closed_idx]
            if non_liq_closed > 0:
                self.roll_count = max(0, self.roll_count - non_liq_closed)

        if params.rolling_enabled and self.positions and self.roll_count < params.rolling_max_times:
            available_free_margin = _free_margin(self.wallet_balance, self.positions, price)
            additions: list[_SimPosition] = []
            for pos in self.positions:
                unrealized = _unrealized_pnl_pct(pos.direction, pos.entry_price, price, pos.leverage)
                if unrealized < params.rolling_trigger_pct:
                    continue
                float_profit = pos.margin * unrealized
                new_margin = float_profit * params.rolling_reinvest_pct
                if new_margin <= 0 or available_free_margin < new_margin:
                    continue
                leverage = _min_float(params.base_leverage, params.max_leverage)
                sl_dist = params.sl_atr_mult * atr
                tp_dist = sl_dist * params.tp_rr_ratio
                sl_price = price - sl_dist
                tp_price = price + tp_dist
                if pos.direction < 0:
                    sl_price = price + sl_dist
                    tp_price = price - tp_dist
                additions.append(
                    _SimPosition(
                        direction=pos.direction,
                        entry_price=price,
                        leverage=leverage,
                        margin=new_margin,
                        sl_price=sl_price,
                        tp_price=tp_price,
                        entry_time=idx,
                    )
                )
                available_free_margin -= new_margin
                self.roll_count += 1
                if params.rolling_move_stop:
                    pos.sl_price = pos.entry_price
            self.positions.extend(additions)

        if not self.positions:
            available_free_margin = _free_margin(self.wallet_balance, self.positions, price)
            if signal != 0 and available_free_margin > 0:
                leverage = _min_float(params.base_leverage, params.max_leverage)
                margin = _min_float(available_free_margin * params.risk_per_trade, available_free_margin * params.max_position_pct)
                margin = _min_float(margin, available_free_margin)
                sl_dist = params.sl_atr_mult * atr
                tp_dist = sl_dist * params.tp_rr_ratio
                sl_price = price - sl_dist
                tp_price = price + tp_dist
                if signal < 0:
                    sl_price = price + sl_dist
                    tp_price = price - tp_dist
                self.positions.append(
                    _SimPosition(
                        direction=signal,
                        entry_price=price,
                        leverage=leverage,
                        margin=margin,
                        sl_price=sl_price,
                        tp_price=tp_price,
                        entry_time=idx,
                    )
                )
                self.roll_count = 0

        self.prev_regime = regime

    def open_positions(self, last_bar: IncrementalBar) -> list[IncrementalOpenPosition]:
        if not self.positions:
            return []
        book_liq_price, _, has_liq = _book_liquidation_threshold(self.positions, self.wallet_balance, DEFAULT_MAINTENANCE_RATE)
        out: list[IncrementalOpenPosition] = []
        for pos in self.positions:
            pnl_pct = _unrealized_pnl_pct(pos.direction, pos.entry_price, last_bar.close, pos.leverage)
            liq_price = book_liq_price if has_liq else 0.0
            out.append(
                IncrementalOpenPosition(
                    direction=pos.direction,
                    entry_price=pos.entry_price,
                    mark_price=last_bar.close,
                    leverage=pos.leverage,
                    margin=pos.margin,
                    notional=pos.margin * pos.leverage,
                    liquidation_price=liq_price,
                    unrealized_pnl=pos.margin * pnl_pct,
                )
            )
        return out


class RealtimeIncrementalEvaluator:
    def __init__(
        self,
        params: DecisionParams,
        *,
        initial_capital: float,
        maintenance_rate: float = DEFAULT_MAINTENANCE_RATE,
        regime_window: int = 48,
        regime_min_duration: int = 192,
    ) -> None:
        self.params = DecisionParams.from_dict(params.to_dict())
        self.params.normalize_weights()
        self.initial_capital = initial_capital if initial_capital > 0 else 10000.0
        self.maintenance_rate = maintenance_rate if maintenance_rate > 0 else DEFAULT_MAINTENANCE_RATE
        self.index = 0
        self.last_bar: Optional[IncrementalBar] = None
        self.indicators = IncrementalIndicatorState(
            self.params.fast_ma_period,
            self.params.slow_ma_period,
            self.params.rsi_period,
            self.params.macd_fast,
            self.params.macd_slow,
            self.params.macd_signal,
            self.params.bb_period,
            _max_int(self.params.fast_ma_period, 7),
            self.params.supertrend_mult,
        )
        self.regime = IncrementalRegimeState(regime_window, regime_min_duration)
        self.strategy = IncrementalStrategyState(self.initial_capital, self.maintenance_rate)
        self.last_signal = 0
        self.last_composite = 0.0
        self.last_regime = REGIME_SIDEWAYS

    def seed_dataframe(self, df: pd.DataFrame) -> None:
        for row in df.itertuples(index=False):
            self.step(
                IncrementalBar(
                    timestamp=pd.Timestamp(row.timestamp).tz_convert("UTC"),
                    open=float(row.open),
                    high=float(row.high),
                    low=float(row.low),
                    close=float(row.close),
                    volume=float(row.volume),
                )
            )

    def advance_dataframe(self, df: pd.DataFrame) -> None:
        if df.empty:
            return
        self.seed_dataframe(df)

    def step(self, bar: IncrementalBar) -> None:
        if self.last_bar is not None and bar.timestamp <= self.last_bar.timestamp:
            return
        prev_close = math.nan if self.last_bar is None else self.last_bar.close
        self.indicators.push(bar)
        regime = self.regime.push(bar.close, self.indicators.ema50, self.indicators.adx)
        signal, composite = self._compute_signal(bar.close, prev_close, regime)
        self.strategy.advance(self.index, bar, signal, regime, self.params, self.indicators.atr14)
        self.last_bar = bar
        self.last_signal = signal
        self.last_composite = composite
        self.last_regime = regime
        self.index += 1

    def last_bar_time(self) -> Optional[pd.Timestamp]:
        return None if self.last_bar is None else self.last_bar.timestamp

    def open_positions(self) -> list[IncrementalOpenPosition]:
        if self.last_bar is None:
            return []
        return self.strategy.open_positions(self.last_bar)

    def _compute_signal(self, close: float, prev_close: float, regime: str) -> tuple[int, float]:
        start = _max_int(self.params.slow_ma_period, 50)
        if self.index < start:
            return 0, 0.0

        trend_sig = _compute_incremental_trend_signal(self.indicators)
        momentum_sig = _compute_incremental_momentum_signal(self.indicators, close)
        mean_revert_sig = _compute_incremental_mean_revert_signal(self.indicators, close)
        volume_sig = _compute_incremental_volume_signal(self.index, self.indicators, close, prev_close)
        volatility_sig = _compute_incremental_volatility_signal(self.indicators)

        composite = (
            self.params.trend_weight * trend_sig
            + self.params.momentum_weight * momentum_sig
            + self.params.mean_revert_weight * mean_revert_sig
            + self.params.volume_weight * volume_sig
            + self.params.volatility_weight * volatility_sig
        )

        if self.params.long_bias > 0.7 and composite < 0:
            composite *= (1.0 - self.params.long_bias) * 2.0
        if self.params.long_bias < 0.3 and composite > 0:
            composite *= self.params.long_bias * 2.0

        sensitivity = self.params.regime_sensitivity
        if regime == REGIME_BULL:
            if composite < 0:
                composite *= (1.0 - sensitivity)
            else:
                composite *= (1.0 + sensitivity * 0.3)
        elif regime == REGIME_BEAR:
            if composite > 0:
                composite *= (1.0 - sensitivity)
            else:
                composite *= (1.0 + sensitivity * 0.3)

        if composite > self.params.entry_threshold:
            return 1, composite
        if composite < -self.params.entry_threshold:
            return -1, composite
        return 0, composite


def _compute_incremental_trend_signal(state: IncrementalIndicatorState) -> float:
    ema_sig = 0.0
    if not math.isnan(state.ema_slow) and state.ema_slow != 0:
        ema_diff = (state.ema_fast - state.ema_slow) / state.ema_slow
        ema_sig = _clamp(ema_diff * 80.0, -1.0, 1.0)
    st_sig = 1.0 if state.super_dir == 1 else -1.0
    di_sig = 0.0
    if not math.isnan(state.plus_di) and not math.isnan(state.minus_di):
        di_sig = _clamp((state.plus_di - state.minus_di) / 30.0, -1.0, 1.0)
    confidence = 0.5
    if not math.isnan(state.adx):
        confidence = 0.5 + 0.5 * min(state.adx / 30.0, 1.0)
    raw = ema_sig * 0.40 + st_sig * 0.30 + di_sig * 0.30
    return _clamp(raw * confidence, -1.0, 1.0)


def _compute_incremental_momentum_signal(state: IncrementalIndicatorState, close: float) -> float:
    rsi_sig = 0.0
    if not math.isnan(state.rsi):
        rsi_sig = _clamp(((state.rsi - 50.0) / 50.0) * 1.5, -1.0, 1.0)
    macd_sig = 0.0
    if close > 0 and not math.isnan(state.macd_hist):
        macd_sig = _clamp(state.macd_hist / (close * 0.002), -1.0, 1.0)
    return _clamp(rsi_sig * 0.5 + macd_sig * 0.5, -1.0, 1.0)


def _compute_incremental_mean_revert_signal(state: IncrementalIndicatorState, close: float) -> float:
    if math.isnan(state.bb_upper) or math.isnan(state.bb_lower) or math.isnan(state.bb_mid) or state.bb_upper == state.bb_lower:
        return 0.0
    position = (close - state.bb_mid) / ((state.bb_upper - state.bb_lower) / 2.0)
    return _clamp(-position * 0.8, -1.0, 1.0)


def _compute_incremental_volume_signal(index: int, state: IncrementalIndicatorState, close: float, prev_close: float) -> float:
    if math.isnan(state.obv_ema) or state.obv_ema == 0 or math.isnan(state.vol_ma) or state.vol_ma == 0:
        return 0.0
    obv_sig = _clamp((state.obv - state.obv_ema) / abs(state.obv_ema) * 10.0, -1.0, 1.0)
    vol_ratio = state.volumes.values[-1] / state.vol_ma
    vol_boost = _clamp((vol_ratio - 1.0) * 0.5, 0.0, 0.5)
    price_dir = -1.0
    if index > 0 and not math.isnan(prev_close) and close > prev_close:
        price_dir = 1.0
    return _clamp(obv_sig * 0.7 + price_dir * vol_boost * 0.3, -1.0, 1.0)


def _compute_incremental_volatility_signal(state: IncrementalIndicatorState) -> float:
    if math.isnan(state.atr14) or math.isnan(state.atr_ma) or state.atr_ma == 0:
        return 0.0
    expansion = (state.atr14 - state.atr_ma) / state.atr_ma
    return _clamp(expansion, -1.0, 1.0)
