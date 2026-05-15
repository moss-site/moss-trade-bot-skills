#!/usr/bin/env python3
"""
Ambush mode live runner. Connects to the moss server's
`/ambush-events/{bootstrap,ws,...}` endpoints and converts each
broadcast event into a long / short / skip decision + order.

Usage:
    python -m ambush.live_runner --creds ~/.moss-trade-bot/agent_creds.json

    # or with explicit overrides:
    python -m ambush.live_runner --creds creds.json \
        --params-file ambush_params.json \
        --db /tmp/ambush_state.db \
        --platform-url https://moss-dev.moss.site

The runner spins up two concurrent consumers — a WS streamer plus a
REST poller — that share the same SQLite cursor + dedup tables.
Either channel alone is sufficient; both running is just belt-and-
suspenders for production. Layered the same way as the
hyperliquid-copy-trade follow_service.

Required fields in creds JSON:
  api_key, api_secret, bot_id, base_url   (base_url overridable via --platform-url)

Optional fields:
  ambush_params (dict) — bot's direction + leverage + position_pct;
                          falls back to params-file / defaults.

The runner is stateless across restarts — all per-event state lives
in the SQLite file at `--db`. Killing + restarting picks up where it
left off via the cursor on first bootstrap.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import signal
import sys
from pathlib import Path

# Allow running from anywhere — scripts/ root on sys.path.
_SCRIPTS_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _SCRIPTS_ROOT not in sys.path:
    sys.path.insert(0, _SCRIPTS_ROOT)

from ambush import ambush_client
from ambush import ambush_poller
from ambush import ambush_ws
from ambush import event_handler
from ambush import live_database as db
from trading_client import TradingClient


logger = logging.getLogger("ambush.runner")


DEFAULT_AMBUSH_PARAMS = {
    "direction": "balanced",
    # leverage capped at 3 for all ambush coins — Hyperliquid enforces this
    # for the low-cap perp universe (127 / 230 coins as of 2026-05). Server
    # `ValidateAmbushBotParams` rejects > 3; downstream `validateSourceLeverage`
    # rejects again at order placement. "Aggressive" tuning happens via
    # position_pct, not leverage.
    "long_params": {
        "leverage": 3,
        "position_pct": "0.20",
    },
    "short_params": {
        "leverage": 3,
        "position_pct": "0.30",
    },
    "rhythm": {
        "max_trades_per_event": 1,
        "same_coin_dedup_days": 7,
    },
}


def _load_json(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def _resolve_ambush_params(creds: dict, params_file: str | None) -> dict:
    """Priority: explicit params file > creds.ambush_params > defaults."""
    if params_file:
        return _load_json(params_file)
    if creds.get("ambush_params"):
        return creds["ambush_params"]
    return DEFAULT_AMBUSH_PARAMS


async def _main_async(args: argparse.Namespace) -> int:
    creds = _load_json(args.creds)
    ambush_params = _resolve_ambush_params(creds, args.params_file)
    direction = str(ambush_params.get("direction") or "balanced").lower()

    base_url = args.platform_url or creds.get("base_url", "")
    if not base_url:
        logger.error("base_url missing; pass --platform-url or set creds.base_url")
        return 2
    bot_id = creds.get("bot_id", "")
    if not bot_id:
        logger.error("bot_id missing in creds; create realtime ambush bot first")
        return 2

    # SQLite state.
    db_path = args.db or os.path.expanduser("~/.moss-trade-bot/ambush_state.db")
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    db.init_db(db_path)
    logger.info("ambush runner: db=%s cursor=%d direction=%s",
                db_path, db.get_last_event_id_seen(db_path), direction)

    # Single TradingClient handles both account/order RPC and serves as
    # the HMAC signer baseline for AmbushClient's REST + WS endpoints.
    # symbol is mutated per-event inside event_handler.process_event.
    tc = TradingClient(
        api_key=creds["api_key"],
        api_secret=creds["api_secret"],
        base_url=base_url,
        bot_id=bot_id,
        symbol="BTCUSDC",  # placeholder; per-event override happens later
    )
    ac = ambush_client.AmbushClient(tc, bot_id=bot_id)
    handler = event_handler.EventHandler(tc, ambush_params)

    # Lifecycle.
    stop_event = asyncio.Event()

    def _sig_handler(signum, _frame):
        logger.info("ambush runner: signal %s received, draining...", signum)
        stop_event.set()

    loop = asyncio.get_event_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, lambda s=sig: _sig_handler(s, None))
        except NotImplementedError:
            # Windows / non-asyncio-supported envs — fall back to default.
            signal.signal(sig, _sig_handler)

    ws_task = asyncio.create_task(
        ambush_ws.run_ws(ac, db_path, direction, handler, stop_event),
        name="ambush-ws",
    )
    poll_task = asyncio.create_task(
        ambush_poller.run_poller(
            ac, db_path, direction, handler, stop_event,
            interval_sec=args.poll_interval,
        ),
        name="ambush-poll",
    )

    logger.info("ambush runner: started ws+poller, awaiting events")
    try:
        await asyncio.gather(ws_task, poll_task, return_exceptions=True)
    finally:
        logger.info("ambush runner: stopped")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Ambush mode live runner")
    parser.add_argument("--creds", required=True, help="Agent credentials JSON")
    parser.add_argument(
        "--params-file", default=None,
        help="Optional ambush params JSON (overrides creds.ambush_params)",
    )
    parser.add_argument(
        "--db", default="",
        help="SQLite state file (default ~/.moss-trade-bot/ambush_state.db)",
    )
    parser.add_argument(
        "--platform-url", default="",
        help="Moss server origin (https://...). Overrides creds.base_url.",
    )
    parser.add_argument(
        "--poll-interval", type=float, default=30.0,
        help="REST poller tick in seconds (default 30)",
    )
    parser.add_argument(
        "--log-level", default="INFO",
        help="Python logging level (DEBUG/INFO/WARNING/ERROR)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    try:
        return asyncio.run(_main_async(args))
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    sys.exit(main())
