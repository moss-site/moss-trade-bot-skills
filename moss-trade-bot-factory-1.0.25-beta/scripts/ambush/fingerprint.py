"""Python mirror of internal/ambush/backtest/fingerprint.go.

Both files MUST produce identical hex SHA-256 for identical inputs;
divergence is a cross-language bug surfaced by the server's
fingerprint_mismatch check on /backtest/verify-job.

Key encoding invariants (must stay in sync with Go):
  - json.dumps with sort_keys=True and no whitespace (separators=(",",":"))
    mirrors Go's json.Marshal on map[string]any which sorts map keys.
  - Decimal values use StringFixed(N) — fixed-point, ROUND_HALF_UP.
  - initial_capital: StringFixed(2); all params decimals: StringFixed(8).
  - direction is lowercased + stripped (matching Go strings.ToLower/TrimSpace).
  - DatasetSHA256: events.csv → features.csv → klines sorted by filename;
    for each klines file: filename bytes then file content bytes.
"""

from __future__ import annotations

import hashlib
import json
import os
from decimal import Decimal, ROUND_HALF_UP


CURRENT_HARNESS_VERSION = "v1"


def canonical(
    strategy_type: str,
    harness_version: str,
    dataset_sha256: str,
    initial_capital,
    params,
) -> str:
    """Compute SHA-256(canonical_json(payload)).

    payload mirrors Go's Canonical map exactly:
      {dataset_sha256, harness_version, initial_capital, params, strategy_type}
    Keys are sorted by json.dumps(sort_keys=True) which replicates Go
    map[string]any marshal order.

    Args:
        initial_capital: Decimal or any value castable to Decimal.
        params: object with .direction, .long, .short, .rhythm attributes
                (AmbushBotParams) — or any duck-typed equivalent.
    """
    payload = {
        "dataset_sha256":  dataset_sha256,
        "harness_version": harness_version,
        "initial_capital": _fixed(initial_capital, 2),
        "params":          _canonical_params(params),
        "strategy_type":   strategy_type,
    }
    # sort_keys=True matches Go's map[string]any json.Marshal key ordering.
    # separators with no spaces matches Go's compact output.
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def _canonical_params(p) -> dict:
    """Convert params to a canonical map, matching Go canonicalParams().

    Uses fixed-point strings for all Decimal fields (8 dp for params,
    2 dp for initial_capital — done in canonical() not here).
    Key insertion order doesn't matter since the outer json.dumps sorts keys,
    but we keep the same keys as Go for clarity.
    """
    return {
        "direction": str(getattr(p, "direction", "")).strip().lower(),
        "long": {
            "cooldown_bars":  int(getattr(p.long, "cooldown_bars", 0)),
            "leverage":       int(getattr(p.long, "leverage", 0)),
            "max_hold_hours": int(getattr(p.long, "max_hold_hours", 0)),
            "momentum_bars":  int(getattr(p.long, "momentum_bars", 0)),
            "position_pct":   _fixed(p.long.position_pct, 8),
            "sl_atr_mult":    _fixed(p.long.sl_atr_mult, 8),
            "stop_loss_pct":  _fixed(p.long.stop_loss_pct, 8),
            "trailing_pct":   _fixed(p.long.trailing_pct, 8),
        },
        "rhythm": {
            "max_trades_per_event": int(getattr(p.rhythm, "max_trades_per_event", 1)),
            "same_coin_dedup_days": int(getattr(p.rhythm, "same_coin_dedup_days", 7)),
        },
        "short": {
            "cooldown_bars":    int(getattr(p.short, "cooldown_bars", 0)),
            "entry_delay_bars": int(getattr(p.short, "entry_delay_bars", 0)),
            "leverage":         int(getattr(p.short, "leverage", 0)),
            "max_hold_hours":   int(getattr(p.short, "max_hold_hours", 0)),
            "position_pct":     _fixed(p.short.position_pct, 8),
            "sl_atr_mult":      _fixed(p.short.sl_atr_mult, 8),
            "stop_loss_pct":    _fixed(p.short.stop_loss_pct, 8),
            "trailing_pct":     _fixed(p.short.trailing_pct, 8),
        },
    }


def _fixed(d, places: int) -> str:
    """Mirror of Go decimal.Decimal.StringFixed(places).

    Converts d to a fixed-point string with exactly `places` decimal digits,
    rounding HALF_UP (banker's rounding is not used — matches shopspring).
    """
    if not isinstance(d, Decimal):
        d = Decimal(str(d))
    q = Decimal(1).scaleb(-places)  # e.g. 1e-8 for places=8
    return str(d.quantize(q, rounding=ROUND_HALF_UP))


def dataset_sha256(dir_path) -> str:
    """Hash the on-disk dataset bundle.

    Mirrors Go DatasetSHA256(dir string):
      1. Hash events.csv content.
      2. Hash features.csv content.
      3. For each klines/*.csv in sorted filename order:
         hash filename bytes then file content bytes.

    Both sides (Go + Python) must agree on the SHA before computing the
    fingerprint — the server checks this against its bundled dataset.
    """
    h = hashlib.sha256()
    dir_path = str(dir_path)
    for name in ("events.csv", "features.csv"):
        path = os.path.join(dir_path, name)
        with open(path, "rb") as f:
            h.update(f.read())
    klines_dir = os.path.join(dir_path, "klines")
    filenames = sorted(fn for fn in os.listdir(klines_dir) if fn.endswith(".csv"))
    for fn in filenames:
        # Hash filename then content — reordering changes the digest (Go parity).
        h.update(fn.encode("utf-8"))
        with open(os.path.join(klines_dir, fn), "rb") as f:
            h.update(f.read())
    return h.hexdigest()
