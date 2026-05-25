#!/usr/bin/env python3
"""Generate data fingerprint from a logical data_cache path or local CSV.

Usage:
    python fetch_data.py --data data_cache/hyperliquid_BTCUSDC_15m_2025-07-01_304d.csv --symbol BTC/USDC --timeframe 15m
    python fetch_data.py --data data_cache/hyperliquid_BTCUSDC_15m_2025-07-01_304d.csv --symbol BTC/USDC --timeframe 15m --start 2025-08-01 --end 2026-01-01
"""

import argparse
import json
import hashlib
import sys
import os

from core.data_cache_archive import ensure_data_file


def _normalize_symbol_pair(raw: str) -> str:
    value = str(raw or "").strip().upper().replace("-", "/")
    if not value:
        return ""
    if "/" in value:
        base, quote = value.split("/", 1)
        base = base.strip()
        quote = quote.strip()
        if not base or not quote:
            return ""
        return f"{base}/{quote}"

    # Compact symbol formats, e.g. BTCUSDC / BTCUSDT
    for quote in ("USDC", "USDT", "BUSD"):
        if value.endswith(quote) and len(value) > len(quote):
            return f"{value[:-len(quote)]}/{quote}"
    return value


def _compact_symbol(pair: str) -> str:
    """Convert slash format to compact format: 'ETH/USDC' -> 'ETHUSDC'."""
    return pair.replace("/", "").replace("-", "").replace(":", "").upper()


def _detect_exchange(csv_path: str) -> str:
    """Replay always runs against the Hyperliquid perpetual universe."""
    del csv_path
    return "hyperliquid"


def _parse_utc(raw: str):
    if not raw:
        return None
    import pandas as pd
    return pd.to_datetime(raw, utc=True)


def _date_tag(ts) -> str:
    if ts is None:
        return ""
    return ts.strftime("%Y-%m-%dT%H-%M-%SZ")


def _range_cache_path(csv_path: str, symbol: str, timeframe: str, start_ts, end_ts) -> str:
    del csv_path
    base_dir = os.path.join(os.environ.get("TMPDIR", "/tmp"), "moss-trade-bot-ranges")
    os.makedirs(base_dir, exist_ok=True)
    safe_symbol = _compact_symbol(symbol)
    tag = f"{_date_tag(start_ts)}_{_date_tag(end_ts)}"
    return os.path.join(base_dir, f"hyperliquid_{safe_symbol}_{timeframe}_{tag}.csv")


def _filter_df_range(df, start_ts, end_ts):
    if start_ts is None and end_ts is None:
        return df
    import pandas as pd
    out = df.copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True)
    if start_ts is not None:
        out = out[out["timestamp"] >= start_ts]
    if end_ts is not None:
        out = out[out["timestamp"] < end_ts]
    out = out.reset_index(drop=True)
    if out.empty:
        raise ValueError("No candles left after applying --start/--end range")
    return out


def fingerprint_from_df(df, csv_path: str, symbol: str, timeframe: str, range_mode: str = "default") -> dict:
    checksum_raw = ",".join(f"{v:.2f}" for v in df["close"])
    checksum = "sha256:" + hashlib.sha256(checksum_raw.encode()).hexdigest()
    return {
        "symbol": _compact_symbol(symbol),
        "timeframe": timeframe,
        "exchange": _detect_exchange(csv_path),
        "range_mode": range_mode,
        "start": str(df["timestamp"].iloc[0]),
        "end": str(df["timestamp"].iloc[-1]),
        "bars": len(df),
        # Round to 8 decimals to keep relative error well under server's
        # 0.5% hard-fail tolerance for low-price coins (DOGE/TRX/XRP-class).
        # 2-decimal rounding silently breaks any coin priced under ~$10.
        "first_close": round(float(df["close"].iloc[0]), 8),
        "last_close": round(float(df["close"].iloc[-1]), 8),
        "checksum": checksum,
        "csv_path": os.path.abspath(csv_path),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Generate data fingerprint from a logical data_cache path or local CSV"
    )
    parser.add_argument(
        "--data", required=True,
        help="Path to OHLCV CSV file, or a logical data_cache CSV path from the release asset.",
    )
    parser.add_argument(
        "--symbol", default="BTC/USDC",
        help="Trading pair (default: BTC/USDC)",
    )
    parser.add_argument("--timeframe", default="15m")
    parser.add_argument("--start", default=None, help="Custom inclusive start date/datetime.")
    parser.add_argument("--end", default=None, help="Custom exclusive end date/datetime.")
    parser.add_argument("--output", default=None, help="Output clipped CSV path.")
    parser.add_argument("--range-mode", choices=["default", "custom"], default=None)
    args = parser.parse_args()

    symbol = _normalize_symbol_pair(args.symbol)
    if not symbol:
        print(f"Error: invalid symbol: {args.symbol}", file=sys.stderr)
        sys.exit(1)

    try:
        data_path = ensure_data_file(args.data)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    import pandas as pd
    df = pd.read_csv(data_path, parse_dates=["timestamp"])
    for col in ["timestamp", "open", "high", "low", "close", "volume"]:
        if col not in df.columns:
            raise ValueError(f"CSV must have column '{col}'")
    start_ts = _parse_utc(args.start)
    end_ts = _parse_utc(args.end)
    df = _filter_df_range(df, start_ts, end_ts)
    range_mode = args.range_mode or ("custom" if args.start or args.end else "default")
    csv_path = data_path
    if args.output or args.start or args.end:
        csv_path = args.output or _range_cache_path(
            data_path,
            symbol,
            args.timeframe,
            start_ts or df["timestamp"].iloc[0],
            end_ts or df["timestamp"].iloc[-1],
        )
        df.to_csv(csv_path, index=False)
    fingerprint = fingerprint_from_df(df, csv_path, symbol, args.timeframe, range_mode)
    print(json.dumps(fingerprint, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
