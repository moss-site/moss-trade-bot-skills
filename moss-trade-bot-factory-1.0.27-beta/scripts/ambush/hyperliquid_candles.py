"""HTTP wrapper around Hyperliquid public `info` candleSnapshot.

Returns last `limit` 15m K-lines for an HL coin (base symbol, e.g. "SAGA"
not "SAGAUSDT" — HL info API uses bare base names). No HMAC; pure read.

Normalizes HL's wire shape:
  {"t": int_ms, "T": int_ms, "s": str, "i": str,
   "o": str, "c": str, "h": str, "l": str, "v": str, "n": int}
into the bar dict the rest of ambush uses:
  {"ts": datetime UTC, "open": float, "close": float,
   "high": float, "low": float, "volume": float}

Failures are surfaced via HyperliquidCandleError (NOT raised through to the
caller's main flow — callers must catch and fall back to the snapshot path).
"""
from __future__ import annotations

import json
import urllib.error
import urllib.request
from datetime import datetime, timedelta, timezone
from typing import Any


_HL_INFO_URL = "https://api.hyperliquid.xyz/info"
_DEFAULT_INTERVAL = "15m"
_DEFAULT_TIMEOUT_SECONDS = 10.0


class HyperliquidCandleError(RuntimeError):
    """Raised on any HTTP / payload error from the HL info endpoint."""


def fetch(symbol: str, interval: str = _DEFAULT_INTERVAL, limit: int = 96,
          timeout: float = _DEFAULT_TIMEOUT_SECONDS) -> list[dict]:
    """Fetch the last `limit` 15m candles for HL base `symbol`.

    Caller must catch HyperliquidCandleError and fall back to snapshot
    behaviour — this function never returns a partial result on failure.

    Programming errors (bad `interval` or non-positive `limit`) raise
    `ValueError` — these indicate caller bugs and are not catchable through
    the normal HL-failure fallback path. Only `HyperliquidCandleError`
    should be caught for runtime degradation.
    """
    if interval != _DEFAULT_INTERVAL:
        # Other intervals are valid HL inputs but ambush only uses 15m.
        # Reject up-front to make the call site contract explicit.
        raise ValueError(f"only interval=15m supported, got {interval!r}")
    if limit <= 0:
        raise ValueError(f"limit must be positive, got {limit}")

    # 15m bar = 900s; ask for a window slightly larger than `limit` × 900
    # so the latest in-flight bar doesn't truncate the result.
    end_time = datetime.now(timezone.utc)
    start_time = end_time - timedelta(minutes=15 * (limit + 2))
    body = json.dumps({
        "type": "candleSnapshot",
        "req": {
            "coin": symbol,
            "interval": interval,
            "startTime": int(start_time.timestamp() * 1000),
            "endTime": int(end_time.timestamp() * 1000),
        },
    }).encode()
    req = urllib.request.Request(
        _HL_INFO_URL,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with _urlopen(req, timeout=timeout) as resp:
            raw = resp.read()
    except urllib.error.HTTPError as e:
        raise HyperliquidCandleError(
            f"HL candleSnapshot HTTP {e.code} for {symbol}: {e.reason}"
        ) from e
    except (urllib.error.URLError, TimeoutError) as e:
        raise HyperliquidCandleError(
            f"HL candleSnapshot network error for {symbol}: {e}"
        ) from e

    try:
        entries: list[dict[str, Any]] = json.loads(raw)
    except json.JSONDecodeError as e:
        raise HyperliquidCandleError(
            f"HL candleSnapshot malformed JSON for {symbol}: {e}"
        ) from e
    if not isinstance(entries, list):
        raise HyperliquidCandleError(
            f"HL candleSnapshot returned non-list for {symbol}: {type(entries)}"
        )

    bars = [_normalize(e, symbol) for e in entries]
    # HL returns oldest-first; preserve, but also defend-by-sort.
    bars.sort(key=lambda b: b["ts"])
    # Caller asked for `limit`; if HL returned more, trim to the most-recent.
    if len(bars) > limit:
        bars = bars[-limit:]
    return bars


def _normalize(entry: dict[str, Any], symbol: str) -> dict[str, Any]:
    try:
        ts_ms = int(entry["t"])
        return {
            "ts": datetime.fromtimestamp(ts_ms / 1000.0, tz=timezone.utc),
            "open": float(entry["o"]),
            "close": float(entry["c"]),
            "high": float(entry["h"]),
            "low": float(entry["l"]),
            "volume": float(entry.get("v", 0.0)),
        }
    except (KeyError, TypeError, ValueError) as e:
        raise HyperliquidCandleError(
            f"HL candleSnapshot bad entry for {symbol}: {entry!r} ({e})"
        ) from e


# Indirection so tests can monkey-patch urlopen without touching urllib globally.
def _urlopen(req, timeout: float):  # pragma: no cover — covered via patch
    return urllib.request.urlopen(req, timeout=timeout)
