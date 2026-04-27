#!/usr/bin/env python3
"""Generate data fingerprint from local CSV.

Usage:
    python fetch_data.py --data /path/to/ohlcv.csv --symbol BTC/USDC --timeframe 15m
"""

import argparse
import json
import hashlib
import sys
import os


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

    # Compact symbol formats, e.g. BTCUSDT
    for quote in ("USDT", "USDC", "BUSD"):
        if value.endswith(quote) and len(value) > len(quote):
            return f"{value[:-len(quote)]}/{quote}"
    return value


def _compact_symbol(pair: str) -> str:
    """Convert slash format to compact format: 'ETH/USDT' -> 'ETHUSDT'."""
    return pair.replace("/", "").replace("-", "").replace(":", "").upper()


def _detect_exchange(csv_path: str) -> str:
    """Detect exchange from CSV filename."""
    basename = os.path.basename(csv_path).lower()
    if "hyperliquid" in basename or "hyper" in basename:
        return "hyperliquid"
    return "hyperliquid"


def fingerprint_from_df(df, csv_path: str, symbol: str, timeframe: str) -> dict:
    checksum_raw = ",".join(f"{v:.2f}" for v in df["close"])
    checksum = "sha256:" + hashlib.sha256(checksum_raw.encode()).hexdigest()
    return {
        "symbol": _compact_symbol(symbol),
        "timeframe": timeframe,
        "exchange": _detect_exchange(csv_path),
        "start": str(df["timestamp"].iloc[0]),
        "end": str(df["timestamp"].iloc[-1]),
        "bars": len(df),
        "first_close": round(float(df["close"].iloc[0]), 2),
        "last_close": round(float(df["close"].iloc[-1]), 2),
        "checksum": checksum,
        "csv_path": os.path.abspath(csv_path),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Generate data fingerprint from local CSV"
    )
    parser.add_argument(
        "--data", required=True,
        help="Path to OHLCV CSV file.",
    )
    parser.add_argument(
        "--symbol", default="BTC/USDT",
        help="Trading pair (default: BTC/USDT)",
    )
    parser.add_argument("--timeframe", default="15m")
    args = parser.parse_args()

    symbol = _normalize_symbol_pair(args.symbol)
    if not symbol:
        print(f"Error: invalid symbol: {args.symbol}", file=sys.stderr)
        sys.exit(1)

    if not os.path.isfile(args.data):
        print(f"Error: CSV file not found: {args.data}", file=sys.stderr)
        sys.exit(1)

    import pandas as pd
    df = pd.read_csv(args.data, parse_dates=["timestamp"])
    for col in ["timestamp", "open", "high", "low", "close", "volume"]:
        if col not in df.columns:
            raise ValueError(f"CSV must have column '{col}'")
    fingerprint = fingerprint_from_df(df, args.data, symbol, args.timeframe)
    print(json.dumps(fingerprint, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
