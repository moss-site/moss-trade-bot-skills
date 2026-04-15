#!/usr/bin/env python3
"""Download Binance UM futures OHLCV or generate fingerprint from existing CSV.

提供下载脚本：用户或 Agent 可运行本脚本从 Binance 拉取期货数据，无需手动去网页下载。
数据严格限制：仅 Binance USDT 本位期货 (binanceusdm)。

Usage:
    # 下载模式：从 Binance 拉取 2025-10-06~2026-03-03 数据（固定区间，与平台验证一致）
    python fetch_data.py --symbol BTC/USDT --timeframe 15m

    # 指纹模式：用户已有 CSV 时，仅生成指纹
    python fetch_data.py --data /path/to/ohlcv.csv --symbol BTC/USDT --timeframe 15m
"""

import argparse
import json
import hashlib
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.fetcher import fetch_ohlcv, get_ohlcv_cache_path, EXCHANGE_ID


def fingerprint_from_df(df, csv_path: str, symbol: str, timeframe: str) -> dict:
    checksum_raw = ",".join(f"{v:.2f}" for v in df["close"])
    checksum = hashlib.sha256(checksum_raw.encode()).hexdigest()
    return {
        "symbol": symbol,
        "timeframe": timeframe,
        "exchange": EXCHANGE_ID,
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
        description="Download Binance UM OHLCV or generate fingerprint from CSV"
    )
    parser.add_argument(
        "--data",
        help="Path to existing CSV. If omitted, fetch from Binance.",
    )
    parser.add_argument("--symbol", default="BTC/USDT")
    parser.add_argument("--timeframe", default="15m")
    parser.add_argument("--days", type=int, default=148, help="Days to fetch (default 148, with --since yields ~2025-10-06 to 2026-03-03)")
    parser.add_argument("--since", default="2025-10-06", help="Start date YYYY-MM-DD (default 2025-10-06, fixed range for platform verification)")
    parser.add_argument("--output", default=None, help="Output CSV path (download mode only)")
    args = parser.parse_args()

    if args.data:
        # 指纹模式：从用户已有 CSV 生成指纹
        if not os.path.isfile(args.data):
            print(f"Error: CSV file not found: {args.data}", file=sys.stderr)
            sys.exit(1)
        import pandas as pd
        df = pd.read_csv(args.data, parse_dates=["timestamp"])
        for col in ["timestamp", "open", "high", "low", "close", "volume"]:
            if col not in df.columns:
                raise ValueError(f"CSV must have column '{col}'")
        fingerprint = fingerprint_from_df(df, args.data, args.symbol, args.timeframe)
    else:
        # 下载模式：从 Binance 拉取
        df = fetch_ohlcv(
            symbol=args.symbol,
            timeframe=args.timeframe,
            days=args.days,
            since_date=args.since,
            use_cache=True,
        )
        if args.output:
            csv_path = args.output
            df.to_csv(csv_path, index=False)
        else:
            csv_path = get_ohlcv_cache_path(
                args.symbol, args.timeframe, args.days, args.since
            )
            df.to_csv(csv_path, index=False)
        fingerprint = fingerprint_from_df(df, csv_path, args.symbol, args.timeframe)

    print(json.dumps(fingerprint, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
