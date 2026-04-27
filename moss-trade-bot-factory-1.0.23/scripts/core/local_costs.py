from __future__ import annotations

import json
import os
import time
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd


LOCAL_DEFAULT_TAKER_FEE_RATE = 0.00045
HYPERLIQUID_INFO_DEFAULT_BASE_URL = "https://api.hyperliquid.xyz"
HYPERLIQUID_INFO_PATH = "/info"
HYPERLIQUID_DEFAULT_TIMEOUT_SEC = 10.0

_SHARE_ROOT = Path(__file__).resolve().parents[2]
_DATA_DIR = _SHARE_ROOT / "data_cache"


def normalize_coin(symbol: str) -> str:
    value = (symbol or "").strip().upper().replace("/", "").replace("-", "")
    if value.endswith("USDC"):
        return value[:-4]
    if value.endswith("USDC"):
        return value[:-4]
    return value


def local_taker_fee_rate() -> float:
    raw = os.environ.get("LOCAL_TAKER_FEE_RATE", "").strip()
    if raw:
        return float(raw)
    return LOCAL_DEFAULT_TAKER_FEE_RATE


def fetch_hyperliquid_funding_history(
    symbol: str,
    start_time: datetime,
    end_time: datetime,
    *,
    use_cache: bool = True,
) -> list[dict]:
    coin = normalize_coin(symbol)
    if not coin:
        return []

    _DATA_DIR.mkdir(parents=True, exist_ok=True)
    cache_key = f"{coin}_{start_time.strftime('%Y%m%dT%H%M')}_{end_time.strftime('%Y%m%dT%H%M')}"
    cache_file = _DATA_DIR / f"hyperliquid_funding_{cache_key}.json"
    if use_cache and cache_file.exists() and time.time() - cache_file.stat().st_mtime < 6 * 3600:
        return _parse_funding_payload(json.loads(cache_file.read_text(encoding="utf-8")))

    body = json.dumps(
        {
            "type": "fundingHistory",
            "coin": coin,
            "startTime": int(start_time.timestamp() * 1000),
            "endTime": int(end_time.timestamp() * 1000),
        }
    ).encode("utf-8")
    base_url = os.environ.get("HYPERLIQUID_INFO_BASE_URL", HYPERLIQUID_INFO_DEFAULT_BASE_URL).strip() or HYPERLIQUID_INFO_DEFAULT_BASE_URL
    url = f"{base_url.rstrip('/')}{HYPERLIQUID_INFO_PATH}"
    req = urllib.request.Request(
        url,
        data=body,
        headers={
            "Content-Type": "application/json",
            "User-Agent": "moss-agent-trade-local-review/1.0",
        },
        method="POST",
    )
    timeout = float(os.environ.get("HYPERLIQUID_INFO_TIMEOUT_SEC", str(HYPERLIQUID_DEFAULT_TIMEOUT_SEC)))
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        payload = json.loads(resp.read().decode("utf-8"))

    if use_cache:
        cache_file.write_text(json.dumps(payload), encoding="utf-8")
    return _parse_funding_payload(payload)


def average_funding_rate(funding_events: list[dict]) -> Optional[float]:
    if not funding_events:
        return None
    return sum(event["funding_rate"] for event in funding_events) / len(funding_events)


def filter_funding_events(funding_events: list[dict], start_time: pd.Timestamp, end_time: pd.Timestamp) -> list[dict]:
    if not funding_events:
        return []
    return [event for event in funding_events if start_time < event["timestamp"] <= end_time]


def _parse_funding_payload(payload: list[dict]) -> list[dict]:
    events = []
    for entry in payload or []:
        ts_ms = entry.get("time")
        rate = entry.get("fundingRate")
        if ts_ms is None or rate is None:
            continue
        events.append(
            {
                "timestamp": pd.to_datetime(int(ts_ms), unit="ms", utc=True),
                "funding_rate": float(rate),
            }
        )
    events.sort(key=lambda item: item["timestamp"])
    return events
