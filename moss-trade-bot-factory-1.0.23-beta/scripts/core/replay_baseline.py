from __future__ import annotations

from datetime import timedelta
from decimal import Decimal

import pandas as pd


def _D(v) -> Decimal:
    if isinstance(v, Decimal):
        return v
    if isinstance(v, float):
        return Decimal(str(v))
    return Decimal(v)


REPLAY_BASELINE_PROFILE = "replay_baseline_v1"
FIXED_REPLAY_TAKER_FEE_RATE = 0.00045
FIXED_REPLAY_FUNDING_RATE = 0.0000125
# Go realtime-replay currently wires TradeService without a metadata provider,
# so replay execution effectively skips lot-size normalization (LotSize=0).
FIXED_REPLAY_LOT_SIZE = 0.0
FIXED_REPLAY_DEPTH_SNAPSHOT_CAPTURED_AT = "2026-04-10T10:34:16.333Z"
FIXED_REPLAY_DEPTH_SNAPSHOT_MID_PRICE = 71703.5

FIXED_REPLAY_BID_DEPTH_LEVELS = [
    (71703.0, 7.44435),
    (71702.0, 2.01422),
    (71701.0, 1.46441),
    (71700.0, 3.49097),
    (71699.0, 1.88264),
    (71698.0, 5.50923),
    (71697.0, 5.23172),
    (71696.0, 5.61227),
    (71695.0, 7.00781),
    (71694.0, 8.68475),
    (71693.0, 0.68836),
    (71692.0, 17.38444),
    (71691.0, 5.51787),
    (71690.0, 12.88651),
    (71689.0, 1.12108),
    (71688.0, 5.94543),
    (71687.0, 9.78345),
    (71686.0, 4.93759),
    (71685.0, 6.32010),
    (71684.0, 20.63088),
]

FIXED_REPLAY_ASK_DEPTH_LEVELS = [
    (71704.0, 0.19192),
    (71705.0, 0.00081),
    (71707.0, 0.01620),
    (71708.0, 0.10037),
    (71709.0, 0.21643),
    (71710.0, 1.04583),
    (71711.0, 0.21333),
    (71712.0, 1.04571),
    (71713.0, 2.14167),
    (71714.0, 0.23920),
    (71715.0, 3.24684),
    (71716.0, 1.49144),
    (71717.0, 0.17586),
    (71718.0, 0.29103),
    (71719.0, 2.62892),
    (71720.0, 2.71263),
    (71721.0, 0.38773),
    (71722.0, 4.38876),
    (71723.0, 5.53460),
    (71724.0, 2.64018),
]


def replay_baseline_assumptions_text() -> str:
    return (
        "执行假设: taker 固定 4.5bps；funding 固定 +0.00125% 并按整点结算；"
        "成交使用固定 Hyperliquid BTC 20 档深度模板，围绕 mark 缩放并按多档顺序撮合。"
    )


def build_fixed_replay_depth_book(mark_price: float) -> dict[str, list[tuple[float, float]]]:
    if mark_price <= 0:
        return {"bids": [], "asks": []}
    # Use Decimal for price scaling to match Go's shopspring/decimal precision
    d_mark = _D(mark_price)
    d_mid = _D(FIXED_REPLAY_DEPTH_SNAPSHOT_MID_PRICE)
    return {
        "bids": [(float(d_mark * _D(price) / d_mid), qty) for price, qty in FIXED_REPLAY_BID_DEPTH_LEVELS],
        "asks": [(float(d_mark * _D(price) / d_mid), qty) for price, qty in FIXED_REPLAY_ASK_DEPTH_LEVELS],
    }


def floor_qty_to_lot_size(qty: float, lot_size: float = FIXED_REPLAY_LOT_SIZE) -> float:
    qty = abs(float(qty))
    lot_size = float(lot_size)
    if qty <= 0 or lot_size <= 0:
        return max(0.0, qty)
    steps = int(qty / lot_size)
    return steps * lot_size


def simulate_replay_baseline_fills_per_level(direction: int, requested_qty: float, mark_price: float) -> list[tuple[float, float]]:
    """Return per-level fills as [(fill_price, fill_qty), ...].

    Matches Go executeAcrossBookLevels which creates one SourceFill per depth level.
    """
    requested_qty = floor_qty_to_lot_size(requested_qty)
    if requested_qty <= 0 or mark_price <= 0:
        return []
    book = build_fixed_replay_depth_book(mark_price)
    levels = book["asks"] if direction > 0 else book["bids"]
    remaining = requested_qty
    fills: list[tuple[float, float]] = []
    for price, qty in levels:
        if remaining <= 0:
            break
        take_qty = floor_qty_to_lot_size(min(remaining, qty))
        if take_qty <= 0:
            continue
        fills.append((price, take_qty))
        remaining -= take_qty
    return fills


def simulate_replay_baseline_fill(direction: int, requested_qty: float, mark_price: float) -> tuple[float, float, float]:
    """Simulate multi-level depth book fill. Uses Decimal internally for price precision."""
    requested_qty = floor_qty_to_lot_size(requested_qty)
    if requested_qty <= 0 or mark_price <= 0:
        return 0.0, 0.0, 0.0

    book = build_fixed_replay_depth_book(mark_price)
    levels = book["asks"] if direction > 0 else book["bids"]
    d_remaining = _D(requested_qty)
    d_fill_notional = Decimal(0)
    d_filled_qty = Decimal(0)

    for price, qty in levels:
        if d_remaining <= 0:
            break
        d_price = _D(price)
        d_qty = _D(qty)
        take_qty = min(d_remaining, d_qty)
        if take_qty <= 0:
            continue
        d_fill_notional += take_qty * d_price
        d_filled_qty += take_qty
        d_remaining -= take_qty

    if d_filled_qty <= 0:
        return 0.0, 0.0, 0.0
    avg_price = d_fill_notional / d_filled_qty
    return float(avg_price), float(d_filled_qty), float(d_fill_notional)


def synthesize_replay_minute_candles(df: pd.DataFrame) -> pd.DataFrame:
    if "timestamp" not in df.columns or len(df) == 0:
        return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])

    ordered = df.copy().sort_values("timestamp").reset_index(drop=True)
    ordered["timestamp"] = pd.to_datetime(ordered["timestamp"], utc=True)
    if len(ordered) == 1:
        return ordered[["timestamp", "open", "high", "low", "close", "volume"]].copy()

    deltas = ordered["timestamp"].diff().dropna()
    step = deltas.iloc[0] if len(deltas) > 0 else pd.Timedelta(minutes=1)
    step_minutes = max(1, int(round(step.total_seconds() / 60.0)))
    if step_minutes <= 1:
        return ordered[["timestamp", "open", "high", "low", "close", "volume"]].copy()

    rows: list[dict] = []
    for row in ordered.itertuples(index=False):
        ts = pd.Timestamp(row.timestamp).tz_convert("UTC")
        close_px = float(row.close)
        volume = float(getattr(row, "volume", 0.0) or 0.0) / step_minutes
        for minute in range(step_minutes):
            rows.append(
                {
                    "timestamp": ts + pd.Timedelta(minutes=minute),
                    "open": close_px,
                    "high": close_px,
                    "low": close_px,
                    "close": close_px,
                    "volume": volume,
                }
            )
    out = pd.DataFrame(rows)
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True)
    return out


def build_fixed_replay_funding_events(df: pd.DataFrame) -> list[dict]:
    if "timestamp" not in df.columns or len(df) == 0:
        return []

    timestamps = pd.to_datetime(df["timestamp"], utc=True)
    start_ts = timestamps.iloc[0]
    end_ts = timestamps.iloc[-1]
    next_settlement = start_ts.floor("h") + timedelta(hours=1)
    last_settlement = end_ts.floor("h")
    events: list[dict] = []

    while next_settlement <= last_settlement:
        idx = timestamps.searchsorted(next_settlement)
        if idx >= len(df):
            idx = len(df) - 1
        oracle_price = float(df["close"].iloc[idx])
        events.append(
            {
                "timestamp": next_settlement,
                "funding_rate": FIXED_REPLAY_FUNDING_RATE,
                "oracle_price": oracle_price,
            }
        )
        next_settlement += timedelta(hours=1)

    return events
