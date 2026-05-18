#!/usr/bin/env python3
"""
Ambush E2E test injector.

Pushes a synthetic ambush_event into a *running* moss server via the
`_debug_publish` endpoint, so a live_runner WS subscriber receives it
exactly like a real Coinglass/Hyperliquid-detected event.  Built for ad-hoc
QA of the full lifecycle:

    skill (this script)  →  server (_debug_publish)  →  WS publish
                                                      ↘ live_runner.event_handler
                                                         → decide / open / defer
                                                         → place order
                                                         → close_monitor

The 5 rule-coverage shapes (matching scripts/ambush/decision.py) are
preset and pickable by name.  No need to hand-craft (surge_15m, rsi_14,
chg_before_24h) tuples each time.

Usage
-----
    # Pick the rule by name (recommended)
    python3 e2e_inject.py --creds /tmp/ambush_creds.json --rule long_momentum_init --symbol SAGA

    # Override individual fields (e.g. force a higher trigger_price)
    python3 e2e_inject.py --creds ... --rule short_spike_extreme \\
        --trigger-price 0.0500 --oi-mc 0.60

    # Direct numeric mode for boundary testing
    python3 e2e_inject.py --creds ... --symbol DYM \\
        --surge 0.22 --rsi 50 --chg 25 --oi-mc 0.45

Why this script exists
----------------------
Issue 5 from the 2026-05-18 E2E report: SHORT path's `entry_delay_bars=16`
(4h) makes the production-default SHORT signal untestable in interactive
QA.  Use --rule long_momentum_init / long_momentum_extend (momentum_bars=0
by default) for fast happy-path verification; use --rule short_spike_extreme
/ short_compound_overstretch / short_extreme_pullback for SHORT defer-path
audit (verifies the deferred-open thread spawns + logs without waiting 4
hours for the actual order).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PARENT = SCRIPT_DIR.parent
if str(PARENT) not in sys.path:
    sys.path.insert(0, str(PARENT))

from trading_client import TradingClient  # noqa: E402


# Each preset is the minimal (surge, rsi, chg) tuple that lands the named
# rule per scripts/ambush/decision.py BalancedDecideV0.
RULE_PRESETS: dict[str, dict[str, str]] = {
    "short_compound_overstretch": {
        # rule: surge > 0.20 AND chg_before_24h > 20
        "surge_15m": "0.22",
        "rsi_14": "50",
        "change_before_24h_pct": "25",
    },
    "short_extreme_pullback": {
        # rule: chg_before_24h > 100 AND rsi > 70
        "surge_15m": "0.05",
        "rsi_14": "72",
        "change_before_24h_pct": "120",
    },
    "short_spike_extreme": {
        # rule: surge > 0.25 (single-K spike, no other gate)
        "surge_15m": "0.30",
        "rsi_14": "50",
        "change_before_24h_pct": "5",
    },
    "long_momentum_init": {
        # rule: 0.10 < surge < 0.15 AND rsi < 60 AND chg < 10
        "surge_15m": "0.12",
        "rsi_14": "55",
        "change_before_24h_pct": "8",
    },
    "long_momentum_extend": {
        # rule: 30 <= chg <= 80 AND rsi > 75
        "surge_15m": "0.05",
        "rsi_14": "78",
        "change_before_24h_pct": "50",
    },
    "skip_no_match": {
        # All thresholds below firing — useful for runner-path probing.
        "surge_15m": "0.05",
        "rsi_14": "50",
        "change_before_24h_pct": "5",
    },
}


def _load_creds(path: str) -> dict:
    p = Path(path).expanduser()
    if not p.is_file():
        raise SystemExit(f"--creds {path} not found")
    return json.loads(p.read_text())


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--creds", required=True,
                   help="agent_creds JSON path (HMAC creds for the target bot)")
    p.add_argument("--symbol", default="SAGA",
                   help="HL spot symbol (e.g. SAGA / DYM / TST). Default: SAGA")
    p.add_argument("--binance-symbol", default="",
                   help="Binance perp symbol; defaults to <symbol>USDT")
    p.add_argument("--trigger-price", default="0.0208",
                   help="Trigger price (decimal string). Default: 0.0208 (SAGA-ish)")
    p.add_argument("--rule", choices=sorted(RULE_PRESETS.keys()),
                   help="Pick a rule preset (overrides --surge/--rsi/--chg unless those are also given)")
    p.add_argument("--surge", dest="surge_15m", default="",
                   help="surge_15m as decimal (0.10 = 10%%). Overrides rule preset.")
    p.add_argument("--rsi", dest="rsi_14", default="",
                   help="rsi_14 (0-100). Overrides rule preset.")
    p.add_argument("--chg", dest="change_before_24h_pct", default="",
                   help="change_before_24h_pct (e.g. 25 for +25%%). Overrides rule preset.")
    p.add_argument("--oi-mc", default="0.45",
                   help="oi_mc (decimal, e.g. 0.45). Must clear server trigger threshold.")
    p.add_argument("--z-score", default="3.0",
                   help="z_score (server tracks but doesn't gate at skill layer)")
    p.add_argument("--dry-run", action="store_true",
                   help="Print the planned POST body and exit; don't actually publish")
    args = p.parse_args()

    creds = _load_creds(args.creds)

    fields = {"surge_15m": "", "rsi_14": "", "change_before_24h_pct": ""}
    if args.rule:
        fields.update(RULE_PRESETS[args.rule])
    # CLI flags override rule preset
    if args.surge_15m:
        fields["surge_15m"] = args.surge_15m
    if args.rsi_14:
        fields["rsi_14"] = args.rsi_14
    if args.change_before_24h_pct:
        fields["change_before_24h_pct"] = args.change_before_24h_pct
    if not all(fields.values()):
        raise SystemExit(
            "must pass --rule (e.g. --rule long_momentum_init) OR all three of "
            "--surge / --rsi / --chg"
        )

    body = {
        "hl_symbol": args.symbol,
        "binance_symbol": args.binance_symbol or f"{args.symbol}USDT",
        "trigger_price": args.trigger_price,
        "oi_mc": args.oi_mc,
        "z_score": args.z_score,
        **fields,
    }

    if args.dry_run:
        print("[ambush-e2e-inject] DRY RUN — would POST:")
        print(json.dumps(body, indent=2, ensure_ascii=False))
        return 0

    base_url = creds.get("base_url", "")
    bot_id = creds.get("bot_id", "")
    if not (base_url and bot_id):
        raise SystemExit("creds missing base_url or bot_id")
    tc = TradingClient(
        api_key=creds["api_key"], api_secret=creds["api_secret"],
        base_url=base_url, bot_id=bot_id,
    )
    path = f"/agent/realtime/bots/{bot_id}/ambush-events/_debug_publish"
    print(f"[ambush-e2e-inject] POST {base_url}{path}")
    print(f"[ambush-e2e-inject] rule={args.rule or '<custom>'} body={json.dumps(body, ensure_ascii=False)}")
    resp = tc._request("POST", path, body=body)
    print(f"[ambush-e2e-inject] response: {json.dumps(resp, ensure_ascii=False)}")
    if isinstance(resp, dict) and resp.get("debug_published"):
        print(f"[ambush-e2e-inject] OK — event_id={resp.get('event_id')} subscribers={resp.get('subscribers')}")
        print("[ambush-e2e-inject] tail your live_runner log to see the decision + order:")
        print("  tail -f /tmp/ambush_runner.log  # on the host running live_runner")
        return 0
    print("[ambush-e2e-inject] server did not confirm debug_published — check AMBUSH_DEBUG_PUBLISH env on server", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
