"""Python mirror of internal/ambush/backtest.Run (Go).

Same state machine, same sort order, same applyTradeCosts arithmetic.
Diverging outputs from Go are cross-language drift, surfaced by the
verify fingerprint at upload time.

Key design choices to match Go exactly:
  - Sort events by (trigger_ts, base) — same as Go sort.SliceStable.
  - Active-position gate: events that arrive while active_until is held
    record single_position_lock skips (live-executor parity).
  - Capital checks before sizing: insufficient_capital then notional_zero.
  - Wallet-driven sizing: margin_usd = wallet × position_pct;
    notional_usd = margin_usd × leverage; pnl_usd = margin_usd × pnl_pct.
  - EquityCurve: seed at firstEventTS with initial_capital; one point per
    closed trade.
  - walletSharpe uses POPULATION variance (sumSq/n - mean^2). Do NOT use
    sample variance — matches Go metrics.go::walletSharpe comment.
  - walletMaxDrawdownPct returns NEGATIVE percent (legacy sign convention
    matching Go walletMaxDrawdownPct).

The close cascade and cost model are reused from backtest.py
(simulate_position / _apply_trade_costs) which already mirrors Go byte-
for-byte. The only new logic here is the outer chained event-walking loop.
"""

from __future__ import annotations

import dataclasses
import datetime
import decimal
import math
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

# ---------------------------------------------------------------------------
# Re-export data helpers from the existing backtest module so callers only
# need to import chained_harness.
# ---------------------------------------------------------------------------
import sys
_scripts_dir = Path(__file__).resolve().parent.parent
if str(_scripts_dir) not in sys.path:
    sys.path.insert(0, str(_scripts_dir))

from ambush.backtest import (  # noqa: E402
    load_event_dataset,
    load_klines_for_base,
    klines_after_trigger,
    klines_window_ending_at,
    decide_from_window,
    simulate_position,
    _apply_trade_costs,
    _sl_atr_mult_or_default,
)
from ambush.decision import decide, DecisionInput  # noqa: E402


# ---------------------------------------------------------------------------
# Notional floor matching Go's notionalFloorUSDC = 10.
# ---------------------------------------------------------------------------
NOTIONAL_FLOOR_USDC = Decimal("10")

# barsPerHour: 15m bars. Same as Go const.
BARS_PER_HOUR = 4


# ---------------------------------------------------------------------------
# Data classes (mirroring Go types in metrics.go).
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class EquityPoint:
    ts: datetime.datetime
    wallet: Decimal


@dataclasses.dataclass
class Trade:
    base: str
    symbol: str
    side: str                         # "long" | "short" | "skip"
    reason: str
    exit_reason: str = ""
    entry_time: Optional[datetime.datetime] = None
    exit_time: Optional[datetime.datetime] = None
    entry: float = 0.0
    exit_price: float = 0.0
    raw_pnl_pct: float = 0.0
    gross_pnl_pct: float = 0.0
    pnl_pct: float = 0.0
    depth_cost_pct: float = 0.0
    trading_fee_pct: float = 0.0
    funding_fee_pct: float = 0.0
    entry_fee_pct: float = 0.0
    exit_fee_pct: float = 0.0
    depth_filled_ratio: float = 1.0
    position_pct: float = 0.0
    leverage: int = 0
    bars_held: int = 0


@dataclasses.dataclass
class Result:
    initial_capital: Decimal
    final_wallet: Decimal
    equity_curve: List[EquityPoint]
    trades: List[Trade]
    events_total: int
    events_traded: int
    events_skipped: int
    skips_by_reason: Dict[str, int]
    total_return_pct: float
    sharpe: float
    max_drawdown_pct: float
    win_rate: float
    win_count: int = 0
    total_trades: int = 0
    avg_pnl_pct: float = 0.0
    gross_return_pct: float = 0.0
    depth_cost_pct: float = 0.0
    trading_fee_pct: float = 0.0
    funding_fee_pct: float = 0.0
    avg_gross_pnl_pct: float = 0.0
    compounded_return_pct: float = 0.0


# ---------------------------------------------------------------------------
# AmbushBotParams — minimal parameter container that maps the JSON shape
# produced by propose.py.  from_dict() is the canonical constructor.
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class SideParams:
    leverage: int
    position_pct: Decimal
    stop_loss_pct: Decimal
    trailing_pct: Decimal
    max_hold_hours: int
    cooldown_bars: int
    sl_atr_mult: Decimal
    # long only
    momentum_bars: int = 0
    # short only
    entry_delay_bars: int = 0


@dataclasses.dataclass
class RhythmParams:
    max_trades_per_event: int = 1
    same_coin_dedup_days: int = 7


@dataclasses.dataclass
class AmbushBotParams:
    direction: str
    long: SideParams
    short: SideParams
    rhythm: RhythmParams

    @classmethod
    def from_dict(cls, d: dict) -> "AmbushBotParams":
        """Parse the JSON shape emitted by propose.py."""
        def _d(v) -> Decimal:
            return Decimal(str(v)) if v is not None else Decimal("0")

        lp = d.get("long_params", {})
        sp = d.get("short_params", {})
        rp = d.get("rhythm", {})
        return cls(
            direction=d.get("direction", "balanced"),
            long=SideParams(
                leverage=int(lp.get("leverage", 2)),
                position_pct=_d(lp.get("position_pct", "0.20")),
                stop_loss_pct=_d(lp.get("stop_loss_pct", "0.08")),
                trailing_pct=_d(lp.get("trailing_pct", "0.30")),
                max_hold_hours=int(lp.get("max_hold_hours", 6)),
                cooldown_bars=int(lp.get("cooldown_bars", 1)),
                sl_atr_mult=_d(lp.get("sl_atr_mult", "2.0")),
                momentum_bars=int(lp.get("momentum_bars", 0) or 0),
            ),
            short=SideParams(
                leverage=int(sp.get("leverage", 2)),
                position_pct=_d(sp.get("position_pct", "0.20")),
                stop_loss_pct=_d(sp.get("stop_loss_pct", "0.40")),
                trailing_pct=_d(sp.get("trailing_pct", "0.25")),
                max_hold_hours=int(sp.get("max_hold_hours", 6)),
                cooldown_bars=int(sp.get("cooldown_bars", 1)),
                sl_atr_mult=_d(sp.get("sl_atr_mult", "2.5")),
                entry_delay_bars=int(sp.get("entry_delay_bars", 0) or 0),
            ),
            rhythm=RhythmParams(
                max_trades_per_event=int(rp.get("max_trades_per_event", 1)),
                same_coin_dedup_days=int(rp.get("same_coin_dedup_days", 7)),
            ),
        )

    def to_dict(self) -> dict:
        """Inverse of from_dict — produces the propose.py JSON wire shape."""
        return {
            "strategy_type": "ambush",
            "direction": self.direction,
            "trigger": {
                "surge_15m_threshold": 0.10,
                "oi_mc_threshold": 0.0,
                "z_score_threshold": 0.0,
            },
            "long_params": {
                "leverage": self.long.leverage,
                "position_pct": str(self.long.position_pct),
                "stop_loss_pct": str(self.long.stop_loss_pct),
                "trailing_pct": str(self.long.trailing_pct),
                "max_hold_hours": self.long.max_hold_hours,
                "cooldown_bars": self.long.cooldown_bars,
                "sl_atr_mult": str(self.long.sl_atr_mult),
                "momentum_bars": self.long.momentum_bars,
            },
            "short_params": {
                "leverage": self.short.leverage,
                "position_pct": str(self.short.position_pct),
                "stop_loss_pct": str(self.short.stop_loss_pct),
                "trailing_pct": str(self.short.trailing_pct),
                "max_hold_hours": self.short.max_hold_hours,
                "cooldown_bars": self.short.cooldown_bars,
                "sl_atr_mult": str(self.short.sl_atr_mult),
                "entry_delay_bars": self.short.entry_delay_bars,
            },
            "rhythm": {
                "max_trades_per_event": self.rhythm.max_trades_per_event,
                "same_coin_dedup_days": self.rhythm.same_coin_dedup_days,
            },
        }


# ---------------------------------------------------------------------------
# Dataset — loaded from the skill's data_cache/ambush layout.
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class EventRow:
    """One row from the merged events+features dataset."""
    symbol: str
    base: str
    trigger_ts: datetime.datetime  # UTC naive (matches Go dataset.go)
    trigger_date: str
    price_at_trigger: float
    surge_15m: float
    rsi_14: float
    chg_before_24h: float
    oi_mc: float


@dataclasses.dataclass
class Dataset:
    events: List[EventRow]
    klines_cache: Dict[str, Any]  # base -> pd.DataFrame


def load_dataset(data_dir) -> Dataset:
    """Load events.csv + features.csv + klines/ from data_dir.

    This is the Python analogue of Go's LoadDataset(dir string).
    """
    data_dir = Path(data_dir)
    df = load_event_dataset(data_dir)
    events: List[EventRow] = []
    klines_cache: Dict[str, Any] = {}
    for _, row in df.iterrows():
        ts_raw = row["trigger_ts"]
        if isinstance(ts_raw, pd.Timestamp):
            ts = ts_raw
        else:
            ts = pd.Timestamp(ts_raw, tz="UTC")
        if ts.tzinfo is not None:
            ts = ts.tz_convert(None)
        trigger_ts = ts.to_pydatetime()
        events.append(EventRow(
            symbol=str(row["symbol"]),
            base=str(row["base"]),
            trigger_ts=trigger_ts,
            trigger_date=str(row["trigger_date"]),
            price_at_trigger=float(row["price_at_trigger"]),
            surge_15m=float(row["surge_15m"]),
            rsi_14=float(row["rsi_14"]) if pd.notna(row.get("rsi_14")) else 50.0,
            chg_before_24h=float(row["change_before_24h_pct"]) if pd.notna(row.get("change_before_24h_pct")) else 0.0,
            oi_mc=float(row["oi_mc"]) if pd.notna(row.get("oi_mc")) else 0.0,
        ))
        base = str(row["base"])
        if base not in klines_cache:
            kl = load_klines_for_base(data_dir, base)
            klines_cache[base] = kl  # None if missing
    return Dataset(events=events, klines_cache=klines_cache)


# ---------------------------------------------------------------------------
# ChainState — mirror of Go chainState struct + methods.
# ---------------------------------------------------------------------------

class _ChainState:
    """Mirror of Go internal/ambush/backtest/harness.go::chainState."""

    def __init__(self, initial_capital: Decimal, event_capacity: int):
        self.initial_capital: Decimal = Decimal(initial_capital)
        self.wallet: Decimal = Decimal(initial_capital)
        self.active_until: Optional[datetime.datetime] = None
        self.active_symbol: str = ""
        self.first_event_ts: Optional[datetime.datetime] = None

        self.equity_curve: List[EquityPoint] = []
        self.trades: List[Trade] = []
        self.skips_by_reason: Dict[str, int] = {}
        self.events_total: int = 0
        self.events_traded: int = 0
        self.events_skipped: int = 0

    def _note_first_event(self, ts: datetime.datetime) -> None:
        if self.first_event_ts is None:
            self.first_event_ts = ts

    def record_skip(self, ev: EventRow, reason: str, locked_symbol: str = "") -> None:
        """Mirror Go chainState.recordSkip."""
        self._note_first_event(ev.trigger_ts)
        self.events_total += 1
        self.events_skipped += 1
        self.skips_by_reason[reason] = self.skips_by_reason.get(reason, 0) + 1
        sym = locked_symbol if locked_symbol else ev.symbol
        self.trades.append(Trade(
            base=ev.base,
            symbol=sym,
            side="skip",
            reason=reason,
            entry_time=ev.trigger_ts,
            exit_time=ev.trigger_ts,
        ))

    def close_trade(
        self,
        ev: EventRow,
        dec_side: str,
        dec_reason: str,
        entry_ts: datetime.datetime,
        exit_ts: datetime.datetime,
        cost: dict,
        pnl_usd: Decimal,
        exit_reason: str,
        position_pct_val: float,
        leverage_val: int,
        bars_held: int,
    ) -> None:
        """Mirror Go chainState.closeTrade."""
        self._note_first_event(ev.trigger_ts)
        self.events_total += 1
        self.events_traded += 1
        self.wallet = self.wallet + pnl_usd
        self.active_until = exit_ts
        self.active_symbol = ev.symbol

        self.trades.append(Trade(
            base=ev.base,
            symbol=ev.symbol,
            side=dec_side,
            reason=dec_reason,
            exit_reason=exit_reason,
            entry_time=entry_ts,
            exit_time=exit_ts,
            entry=cost.get("entry_price", 0.0),
            exit_price=cost.get("exit_price", 0.0),
            raw_pnl_pct=cost.get("raw_pnl_pct", 0.0),
            gross_pnl_pct=cost.get("gross_pnl_pct", 0.0),
            pnl_pct=cost.get("pnl_pct", 0.0),
            depth_cost_pct=cost.get("depth_cost_pct", 0.0),
            trading_fee_pct=cost.get("trading_fee_pct", 0.0),
            funding_fee_pct=cost.get("funding_fee_pct", 0.0),
            entry_fee_pct=cost.get("entry_fee_pct", 0.0),
            exit_fee_pct=cost.get("exit_fee_pct", 0.0),
            depth_filled_ratio=cost.get("depth_filled_ratio", 1.0),
            position_pct=position_pct_val,
            leverage=leverage_val,
            bars_held=bars_held,
        ))
        self.equity_curve.append(EquityPoint(ts=exit_ts, wallet=self.wallet))

    def finalize(self) -> Result:
        """Mirror Go chainState.finalize."""
        # Prepend seed point at firstEventTS so chart starts flat.
        # Go: seed uses firstEventTS (trigger of first observed event),
        # NOT the first close TS — avoids two points at same ts.
        if self.first_event_ts is not None:
            seed = EquityPoint(ts=self.first_event_ts, wallet=self.initial_capital)
            self.equity_curve = [seed] + self.equity_curve

        if self.equity_curve:
            final_wallet = self.equity_curve[-1].wallet
        else:
            final_wallet = self.initial_capital

        return _aggregate(self, final_wallet)


# ---------------------------------------------------------------------------
# Metrics helpers — mirror of Go metrics.go.
# ---------------------------------------------------------------------------

def _wallet_max_drawdown_pct(curve: List[EquityPoint]) -> float:
    """Peak-to-trough drawdown. Returns NEGATIVE percent (e.g. -12.5 for
    12.5% drawdown), matching Go walletMaxDrawdownPct sign convention."""
    if len(curve) < 2:
        return 0.0
    peak = float(curve[0].wallet)
    if peak <= 0:
        return 0.0
    max_dd = 0.0
    for p in curve:
        v = float(p.wallet)
        if v > peak:
            peak = v
            continue
        if peak <= 0:
            continue
        dd = (v - peak) / peak * 100
        if dd < max_dd:
            max_dd = dd
    return max_dd


def _wallet_sharpe(curve: List[EquityPoint]) -> float:
    """mean(per-step return) / std(per-step return). Uses POPULATION variance
    (sumSq/n - mean^2) — matches Go walletSharpe. Returns 0 for short curves."""
    if len(curve) < 3:
        return 0.0
    rets: List[float] = []
    for i in range(1, len(curve)):
        prev = float(curve[i - 1].wallet)
        cur = float(curve[i].wallet)
        if prev <= 0:
            continue
        rets.append((cur - prev) / prev)
    if len(rets) < 2:
        return 0.0
    n = len(rets)
    mean = sum(rets) / n
    var = sum(r * r for r in rets) / n - mean * mean
    if var <= 0:
        return 0.0
    return mean / math.sqrt(var)


def _aggregate(state: _ChainState, final_wallet: Decimal) -> Result:
    """Mirror of Go aggregate(). Assembles Result from chainState."""
    total_return = 0.0
    if state.initial_capital > 0:
        total_return = float(
            (final_wallet - state.initial_capital) / state.initial_capital * 100
        )

    win_count = 0
    total_trades = state.events_traded
    gross_pct_sum = 0.0
    depth_pct_sum = 0.0
    trading_fee_sum = 0.0
    funding_fee_sum = 0.0
    net_pct_sum = 0.0

    for t in state.trades:
        if t.side == "skip":
            continue
        if t.pnl_pct > 0:
            win_count += 1
        gross_pct_sum += t.gross_pnl_pct
        depth_pct_sum += t.depth_cost_pct
        trading_fee_sum += t.trading_fee_pct
        funding_fee_sum += t.funding_fee_pct
        net_pct_sum += t.pnl_pct

    win_rate = (win_count / total_trades) if total_trades > 0 else 0.0
    avg_pnl = (net_pct_sum / total_trades) if total_trades > 0 else 0.0
    avg_gross = (gross_pct_sum / total_trades) if total_trades > 0 else 0.0

    return Result(
        initial_capital=state.initial_capital,
        final_wallet=final_wallet,
        equity_curve=state.equity_curve,
        trades=state.trades,
        events_total=state.events_total,
        events_traded=state.events_traded,
        events_skipped=state.events_skipped,
        skips_by_reason=dict(state.skips_by_reason),
        total_return_pct=total_return,
        compounded_return_pct=total_return,
        sharpe=_wallet_sharpe(state.equity_curve),
        max_drawdown_pct=_wallet_max_drawdown_pct(state.equity_curve),
        win_rate=win_rate,
        win_count=win_count,
        total_trades=total_trades,
        avg_pnl_pct=avg_pnl,
        gross_return_pct=gross_pct_sum,
        depth_cost_pct=depth_pct_sum,
        trading_fee_pct=trading_fee_sum,
        funding_fee_pct=funding_fee_sum,
        avg_gross_pnl_pct=avg_gross,
    )


# ---------------------------------------------------------------------------
# ChainedHarness — the main public class.
# ---------------------------------------------------------------------------

class ChainedHarness:
    """Mirror of Go internal/ambush/backtest.Run.

    Usage:
        from core.data_cache_archive import resolve_data_root
        params = AmbushBotParams.from_dict(raw_params)
        dataset = load_dataset(resolve_data_root() / "ambush")
        result = ChainedHarness(params).run(dataset, Decimal("10000"))
    """

    def __init__(self, params: AmbushBotParams):
        self.p = params

    def run(self, dataset: Dataset, initial_capital) -> Result:
        """Run the chained harness over the dataset.

        Mirrors Go Run(p, ds, initialCapital) exactly:
          1. Sort events by (trigger_ts, base) — deterministic.
          2. For each event:
             a. Active-position gate → single_position_lock skip.
             b. Entry decision (decide) → skip if SideSkip.
             c. Capital checks → insufficient_capital / notional_zero skip.
             d. simulate_position (close cascade + costs).
             e. Convert margin-relative pnl_pct → USD; update wallet.
          3. finalize() → seed equity curve, aggregate metrics.
        """
        initial_capital = Decimal(str(initial_capital))
        max_hold_bars = (
            max(self.p.long.max_hold_hours, self.p.short.max_hold_hours) * BARS_PER_HOUR
        )

        # Deterministic sort: (trigger_ts, base). Matches Go sort.SliceStable.
        events = sorted(dataset.events, key=lambda e: (e.trigger_ts, e.base))

        state = _ChainState(initial_capital, len(events))

        for ev in events:
            # --- Active-position gate ---
            if state.active_until is not None and ev.trigger_ts <= state.active_until:
                state.record_skip(ev, "single_position_lock", state.active_symbol)
                continue
            # Clear expired lock.
            if state.active_until is not None:
                state.active_until = None
                state.active_symbol = ""

            # --- Entry decision ---
            dec = decide(DecisionInput(
                surge_15m=ev.surge_15m,
                rsi_14=ev.rsi_14 if ev.rsi_14 else 50.0,
                chg_before_24h=ev.chg_before_24h,
                direction=self.p.direction,
            ))
            if dec.side == "skip":
                state.record_skip(ev, dec.reason)
                continue

            # --- Capital checks (before sizing, matching Go order) ---
            if state.wallet <= Decimal("0"):
                state.record_skip(ev, "insufficient_capital")
                continue

            position_pct_val = self._position_pct(dec.side)
            leverage_val = self._leverage(dec.side)
            margin_usd = state.wallet * Decimal(str(position_pct_val))
            notional_usd = margin_usd * Decimal(leverage_val)
            if notional_usd < NOTIONAL_FLOOR_USDC:
                state.record_skip(ev, "notional_zero")
                continue

            # --- Fetch klines ---
            klines_df = dataset.klines_cache.get(ev.base)
            if klines_df is None or klines_df.empty:
                state.record_skip(ev, "missing_klines")
                continue

            delay_bars = self._entry_delay_bars(dec.side)
            trigger_ts_pd = pd.Timestamp(ev.trigger_ts)

            bars_after = klines_after_trigger(
                klines_df, trigger_ts_pd, max_hold_bars + delay_bars + 1
            )
            if not bars_after:
                state.record_skip(ev, "missing_klines")
                continue
            if len(bars_after) <= delay_bars:
                state.record_skip(ev, "insufficient_klines_after_delay")
                continue

            # Apply entry delay (momentum_bars / entry_delay_bars).
            delayed_bars = bars_after[delay_bars:]
            if delay_bars > 0:
                raw_entry = float(delayed_bars[0]["open"])
                entry_ts = _pd_ts_to_dt(delayed_bars[0]["ts"])
                # BT-1: mirror live deferred reconfirm (event_handler._run_deferred_open
                # / Go decideFromBars). Re-decide at the wake bar over the 96-bar
                # window ending there (pre-trigger history included), and expire the
                # deferred open on a side flip or too-few-bars — live expired_recompute.
                fresh_side = decide_from_window(
                    klines_window_ending_at(klines_df, pd.Timestamp(delayed_bars[0]["ts"]), 96),
                    self.p.direction,
                )
                if fresh_side is None or fresh_side != dec.side:
                    state.record_skip(ev, "expired_recompute")
                    continue
            else:
                raw_entry = ev.price_at_trigger
                entry_ts = ev.trigger_ts

            # --- Simulate position (close cascade + costs) ---
            side_params = self.p.short if dec.side == "short" else self.p.long
            outcome = simulate_position(
                klines_after=delayed_bars,
                raw_entry=raw_entry,
                entry_time=pd.Timestamp(entry_ts),
                side=dec.side,
                leverage=leverage_val,
                position_pct=position_pct_val,
                trailing_pct=float(side_params.trailing_pct),
                max_hold_hours=side_params.max_hold_hours,
                max_hold_bars=max_hold_bars,
                sl_atr_mult=_sl_atr_mult_or_default(float(side_params.sl_atr_mult), dec.side),
                direction=self.p.direction,
                klines_full=klines_df,
            )

            # Re-run _apply_trade_costs to get the cost dict (outcome is a
            # TradeOutcome dataclass). We need the dict form for close_trade.
            # simulate_position already computed the exit; extract exit_time.
            if outcome.bars_held > 0 and outcome.bars_held <= len(delayed_bars):
                exit_bar_idx = outcome.bars_held - 1
                exit_ts = _pd_ts_to_dt(delayed_bars[exit_bar_idx]["ts"])
                raw_exit = float(delayed_bars[exit_bar_idx]["close"])
                bars_used = delayed_bars[:outcome.bars_held]
            else:
                exit_ts = _pd_ts_to_dt(delayed_bars[-1]["ts"])
                raw_exit = float(delayed_bars[-1]["close"])
                bars_used = delayed_bars

            cost = _apply_trade_costs(
                side=dec.side,
                raw_entry=raw_entry,
                raw_exit=raw_exit,
                entry_time=pd.Timestamp(entry_ts),
                exit_time=pd.Timestamp(exit_ts),
                position_pct=position_pct_val,
                leverage=leverage_val,
                bars=bars_used,
            )

            # Convert margin-relative pnl_pct to USD using chain wallet margin.
            # pnl_pct is leveraged-margin-relative; margin_usd = wallet × position_pct.
            pnl_usd = margin_usd * Decimal(str(cost["pnl_pct"]))

            state.close_trade(
                ev=ev,
                dec_side=dec.side,
                dec_reason=dec.reason,
                entry_ts=entry_ts,
                exit_ts=exit_ts,
                cost=cost,
                pnl_usd=pnl_usd,
                exit_reason=outcome.exit_reason,
                position_pct_val=position_pct_val,
                leverage_val=leverage_val,
                bars_held=outcome.bars_held,
            )

        return state.finalize()

    def _entry_delay_bars(self, side: str) -> int:
        """Mirror Go entryDelayBars: long → momentum_bars, short → entry_delay_bars."""
        if side == "short":
            return int(self.p.short.entry_delay_bars)
        return int(self.p.long.momentum_bars)

    def _position_pct(self, side: str) -> float:
        if side == "long":
            return float(self.p.long.position_pct)
        return float(self.p.short.position_pct)

    def _leverage(self, side: str) -> int:
        if side == "long":
            return int(self.p.long.leverage)
        return int(self.p.short.leverage)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _pd_ts_to_dt(ts) -> datetime.datetime:
    """Convert pandas Timestamp / datetime to a plain UTC-naive datetime."""
    if isinstance(ts, pd.Timestamp):
        if ts.tzinfo is not None:
            ts = ts.tz_convert(None)
        return ts.to_pydatetime()
    if isinstance(ts, datetime.datetime):
        if ts.tzinfo is not None:
            import pytz  # best-effort; fall back to replace
            try:
                return ts.astimezone(pytz.utc).replace(tzinfo=None)
            except Exception:
                return ts.replace(tzinfo=None)
        return ts
    return datetime.datetime.fromisoformat(str(ts))
