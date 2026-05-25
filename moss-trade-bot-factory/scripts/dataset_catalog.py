#!/usr/bin/env python3
"""Discover bundled fixed OHLCV datasets and their actual coverage."""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import re
import sys
from datetime import datetime, timezone

from core.data_cache_archive import ensure_data_cache


SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATA_DIR = os.path.join(SCRIPTS_DIR, "data_cache")
CSV_RE = re.compile(r"^hyperliquid_([A-Z0-9]+?)(USDC|USDT|BUSD)_([0-9a-z]+)_(.+)\.csv$", re.IGNORECASE)


def _base_asset(raw: str) -> str:
    value = (raw or "").strip().upper()
    value = value.replace("/", "").replace("-", "").replace("_", "")
    if ":" in value:
        value = value.split(":", 1)[0]
    for quote in ("USDC", "USDT", "BUSD"):
        if value.endswith(quote) and len(value) > len(quote):
            return value[: -len(quote)]
    return value


def _parse_ts(raw: str) -> datetime:
    value = (raw or "").strip()
    if value.endswith("Z"):
        value = value[:-1] + "+00:00"
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
        try:
            return datetime.strptime(value.split("+", 1)[0], fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            pass
    try:
        ts = datetime.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"unsupported timestamp: {raw}") from exc
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    return ts.astimezone(timezone.utc)


def _summary(path: str) -> dict:
    name = os.path.basename(path)
    match = CSV_RE.match(name)
    if not match:
        return {}
    base, quote, timeframe, dataset_version = match.groups()
    first = None
    last = None
    rows = 0
    with open(path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or "timestamp" not in reader.fieldnames:
            raise ValueError(f"{path}: missing timestamp header")
        for row in reader:
            rows += 1
            ts = row.get("timestamp", "")
            if first is None:
                first = ts
            last = ts
    if first is None or last is None:
        raise ValueError(f"{path}: empty csv")
    return {
        "base": base.upper(),
        "quote": quote.upper(),
        "symbol": f"{base.upper()}/{quote.upper()}",
        "compact": f"{base.upper()}{quote.upper()}",
        "timeframe": timeframe.lower(),
        "dataset_version": dataset_version,
        "csv_path": os.path.abspath(path),
        "start": first,
        "end": last,
        "bars": rows,
        "filename": name,
        "_start_ts": _parse_ts(first).timestamp(),
        "_mtime": os.path.getmtime(path),
    }


def discover(data_dir: str, timeframe: str) -> list[dict]:
    paths = sorted(glob.glob(os.path.join(data_dir, "hyperliquid_*_*.csv")))
    out = []
    for path in paths:
        item = _summary(path)
        if item and item["timeframe"] == timeframe.lower():
            out.append(item)
    return out


def choose(items: list[dict], symbol: str) -> dict | None:
    base = _base_asset(symbol)
    candidates = [item for item in items if item["base"] == base]
    if not candidates:
        return None
    candidates.sort(key=lambda item: (item["bars"], item["_mtime"], item["_start_ts"]), reverse=True)
    return candidates[0]


def _public(item: dict) -> dict:
    return {k: v for k, v in item.items() if not k.startswith("_")}


def main() -> int:
    parser = argparse.ArgumentParser(description="Discover bundled fixed dataset coverage")
    parser.add_argument("--symbol", default="BTC/USDC", help="Symbol to resolve, e.g. BTC/USDC")
    parser.add_argument("--timeframe", default="15m")
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    parser.add_argument("--list", action="store_true", help="List every discovered dataset")
    args = parser.parse_args()

    data_dir = args.data_dir
    if os.path.abspath(data_dir) == os.path.abspath(DEFAULT_DATA_DIR):
        data_dir = str(ensure_data_cache())

    items = discover(data_dir, args.timeframe)
    if args.list:
        payload = {"datasets": [_public(item) for item in sorted(items, key=lambda x: x["compact"])]}
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0

    selected = choose(items, args.symbol)
    payload = {
        "found": selected is not None,
        "requested_symbol": args.symbol,
        "timeframe": args.timeframe,
        "available_symbols": sorted({item["symbol"] for item in items}),
    }
    if selected is not None:
        payload.update(_public(selected))
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if selected is not None else 2


if __name__ == "__main__":
    sys.exit(main())
