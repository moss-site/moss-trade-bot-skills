#!/usr/bin/env python3
"""
Demo trade loop for an ambush bot — generates one trade every N minutes so
QA / colleague follow-trading services have a steady, predictable stream of
fills to test against.

What each tick does:

  1. If the bot is currently holding a non-flat position, close it (via the
     bot's HMAC-authenticated reduce-only close endpoint). Single-position-
     lock would otherwise reject the new open from #3.
  2. Pick the next symbol from a rotating watchlist (so the follower sees
     varied symbols, not just SAGA over and over).
  3. POST a synthetic `ambush.detected` event to the server-side debug
     publish endpoint (`/ambush-events/_debug_publish`). The skill's
     live-runner consumes the event over WS and places a real order via
     the standard /orders endpoint. From the follower-service POV the
     trade is indistinguishable from a real ambush event firing.
  4. Sleep until the next tick.

Symbol + side rotation:
  We alternate side (SHORT / LONG) AND symbol, both round-robin, so over an
  hour the follower observes mixed direction + diversified instruments.
  The synthesized event payload is tuned to trip the corresponding rule in
  `decision.balanced_decide_v0`:
    SHORT: surge_15m=0.30, rsi=70  → rule_short_spike_extreme
    LONG : surge_15m=0.12, rsi=55, change_before_24h_pct=5
                                   → rule_long_momentum_init

Usage:
    # foreground (logs to stdout)
    python -m ambush.demo_loop --creds ~/.moss-trade-bot/agent_creds.json
                                --interval 300

    # background on bastion (already what we do in this session):
    setsid nohup .venv/bin/python -m ambush.demo_loop \
        --creds /tmp/ambush_creds.json --interval 300 \
        > /tmp/ambush_demo_loop.log 2>&1 < /dev/null &
    disown

NOT production code — this only exists to give downstream follower-service
QA a reproducible source of trades. There's NO real ambush detection
happening; the events are synthetic by design (that's the whole point of
the _debug_publish endpoint).
"""

from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import logging
import os
import secrets
import signal
import sys
import time
import urllib.error
import urllib.request
from decimal import Decimal, InvalidOperation


logger = logging.getLogger("ambush.demo_loop")


# Watchlist symbols we cycle through. Order matters — we rotate by index.
# These must be HL-listed ambush coins that the server's watchlist seeds at
# boot. Confirmed via `ambush_watchlist` table on 2026-05-15:
# SAGA, DYM, TST, MAVIA.
DEFAULT_SYMBOLS = ["SAGA", "DYM", "TST", "MAVIA"]


# Synthetic payloads tuned to trip the skill's `balanced_decide_v0` rules
# in decision.py. Keep these in sync if the rules change.
PAYLOAD_SHORT = {
    "surge_15m": "0.30",            # > 0.25 → short_spike_extreme
    "z_score": "5.0",
    "rsi_14": "70",
    "change_before_24h_pct": "5.0",
}
PAYLOAD_LONG = {
    "surge_15m": "0.12",            # 0.10 < x < 0.15 AND rsi<60 AND chg<10
    "z_score": "3.0",               #   → long_momentum_init
    "rsi_14": "55",
    "change_before_24h_pct": "5.0",
}


class HMACClient:
    """Minimal HMAC-signed REST client for the bot's `/api/v2/moss/agent`
    surface. Reuses the same signing rules as `trading_client.TradingClient`
    but stays self-contained so this script has zero deps on the skill
    package — easier to run as a standalone cron."""

    def __init__(self, creds: dict):
        self.api_key = creds["api_key"]
        self.api_secret = creds["api_secret"]
        self.base_url = creds["base_url"].rstrip("/")
        self.bot_id = creds["bot_id"]

    def _sign(self, method: str, path: str, query: str, body: str) -> tuple[str, str, str]:
        ts = str(int(time.time()))
        nonce = secrets.token_hex(8)
        sign_str = f"{method}\n{path}\n{query}\n{body}\n{ts}\n{nonce}"
        sig = hmac.new(
            self.api_secret.encode(), sign_str.encode(), hashlib.sha256,
        ).hexdigest()
        return ts, nonce, sig

    def request(self, method: str, path: str, *, query: str = "", body: dict | None = None) -> dict:
        """Sign + send. Returns the parsed JSON response (or the server's
        error envelope on HTTP error — never raises on protocol-level
        failures, mirroring trading_client.TradingClient._request)."""
        raw_body = json.dumps(body, separators=(",", ":")) if body is not None else ""
        ts, nonce, sig = self._sign(method, path, query, raw_body)
        url = self.base_url + path + (f"?{query}" if query else "")
        req = urllib.request.Request(
            url,
            data=raw_body.encode() if raw_body else None,
            headers={
                "Content-Type": "application/json",
                "X-API-KEY":   self.api_key,
                "X-TS":        ts,
                "X-NONCE":     nonce,
                "X-SIGNATURE": sig,
            },
            method=method,
        )
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                raw = resp.read()
                return json.loads(raw) if raw else {"status": "ok"}
        except urllib.error.HTTPError as e:
            error_body = e.read().decode()
            try:
                return json.loads(error_body)
            except json.JSONDecodeError:
                return {"code": "HTTP_ERROR", "message": f"{e.code}: {error_body}"}


def get_positions(client: HMACClient) -> list[dict]:
    """List non-flat positions for the bot."""
    resp = client.request("GET", f"/api/v2/moss/agent/realtime/bots/{client.bot_id}/positions")
    if not isinstance(resp, dict) or "items" not in resp:
        # API response shape is {page, page_size, total, items}; fall through
        # to empty on protocol mismatch (treated as flat — safe; we'll just
        # try the open and let server-side lock catch any conflict).
        return []
    items = resp.get("items") or []
    out = []
    for p in items:
        try:
            qty = Decimal(str(p.get("net_qty") or "0"))
        except (InvalidOperation, ValueError):
            qty = Decimal("0")
        if qty != 0:
            out.append(p)
    return out


def close_position_for(client: HMACClient, symbol: str) -> dict:
    """Reduce-only close for one symbol. Same endpoint TradingClient.close
    uses; we hit it directly to keep this script dependency-free."""
    body = {
        "reasoning":    "ambush demo loop: rotating to next symbol",
        "reasoning_en": "ambush demo loop: rotating to next symbol",
    }
    return client.request(
        "POST",
        f"/api/v2/moss/agent/realtime/bots/{client.bot_id}/positions/{symbol}/close",
        body=body,
    )


def publish_debug_event(client: HMACClient, hl_symbol: str, side: str) -> dict:
    """Inject one synthetic ambush.detected event. The server insertion
    UNIQUE constraint is on (hl_symbol, trigger_ts) — we set trigger_ts to
    now in nanosecond precision so back-to-back calls don't collide."""
    payload = PAYLOAD_SHORT if side == "short" else PAYLOAD_LONG
    # nanosecond trigger_ts is RFC3339 with fractional seconds — server
    # accepts RFC3339 with any precision.
    trigger_ts = time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime()) + f".{int(time.time()*1000)%1000:03d}Z"
    body = {
        "hl_symbol":             hl_symbol,
        "binance_symbol":        hl_symbol + "USDT",
        "trigger_ts":            trigger_ts,
        "trigger_price":         "0.02500",       # placeholder; skill uses live mark
        "oi_mc":                 "0.05",
        "surge_15m":             payload["surge_15m"],
        "z_score":               payload["z_score"],
        "rsi_14":                payload["rsi_14"],
        "change_before_24h_pct": payload["change_before_24h_pct"],
    }
    return client.request(
        "POST",
        f"/api/v2/moss/agent/realtime/bots/{client.bot_id}/ambush-events/_debug_publish",
        body=body,
    )


# ── main loop ─────────────────────────────────────────────────────────────


_stop = False


def _signal_handler(signum, _frame):
    global _stop
    logger.info("demo loop: signal %s received, exiting after current tick", signum)
    _stop = True


def run_one_tick(client: HMACClient, hl_symbol: str, side: str) -> None:
    """One end-to-end cycle: close-if-held → publish → log result."""
    # 1. Close any existing position (any symbol — single-position-lock).
    open_positions = get_positions(client)
    if open_positions:
        for pos in open_positions:
            sym = str(pos.get("symbol") or "")
            if not sym:
                continue
            logger.info("demo loop: closing existing position %s qty=%s",
                        sym, pos.get("net_qty"))
            result = close_position_for(client, sym)
            order = (result or {}).get("order") if isinstance(result, dict) else None
            if isinstance(order, dict) and order.get("order_id"):
                logger.info("demo loop: closed %s order_id=%s avg=%s realized_pnl=%s",
                            sym, order.get("order_id"),
                            order.get("avg_fill_price"),
                            (result or {}).get("realized_pnl"))
            else:
                # Non-fatal — server might have race-closed via stop_loss
                # / OI-revert exit; just log and continue.
                logger.warning("demo loop: close %s non-success: %s",
                               sym, result)
    else:
        logger.info("demo loop: bot is flat, no close needed")

    # 2. Inject the synthetic detected event for the next symbol/side.
    logger.info("demo loop: publishing synthetic event %s/%s", hl_symbol, side)
    pub_result = publish_debug_event(client, hl_symbol, side)
    if isinstance(pub_result, dict) and pub_result.get("debug_published"):
        logger.info("demo loop: event published id=%s subscribers=%s",
                    pub_result.get("event_id"),
                    pub_result.get("subscribers"))
    else:
        logger.warning("demo loop: publish non-success: %s", pub_result)


def main() -> int:
    parser = argparse.ArgumentParser(description="Ambush bot demo trade loop")
    parser.add_argument("--creds", required=True, help="Bot HMAC creds JSON")
    parser.add_argument(
        "--interval", type=int, default=300,
        help="Seconds between trade cycles (default 300 = 5 min)",
    )
    parser.add_argument(
        "--symbols", default=",".join(DEFAULT_SYMBOLS),
        help="Comma-separated HL symbols to rotate (default: SAGA,DYM,TST,MAVIA)",
    )
    parser.add_argument(
        "--side-pattern", default="short,long",
        help="Comma-separated side rotation (default: short,long alternating)",
    )
    parser.add_argument(
        "--max-cycles", type=int, default=0,
        help="Stop after N cycles (0 = forever, useful for testing)",
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

    with open(os.path.expanduser(args.creds)) as f:
        creds = json.load(f)
    client = HMACClient(creds)

    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    sides   = [s.strip().lower() for s in args.side_pattern.split(",") if s.strip()]
    if not symbols or not sides:
        logger.error("demo loop: --symbols and --side-pattern must each have at least one entry")
        return 2

    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, _signal_handler)

    logger.info("demo loop: started bot_id=%s interval=%ds symbols=%s sides=%s",
                client.bot_id, args.interval, symbols, sides)

    cycle = 0
    while not _stop:
        sym = symbols[cycle % len(symbols)]
        side = sides[cycle % len(sides)]
        try:
            run_one_tick(client, sym, side)
        except Exception as e:
            logger.exception("demo loop: tick raised %s; continuing", e)
        cycle += 1
        if args.max_cycles and cycle >= args.max_cycles:
            logger.info("demo loop: reached --max-cycles=%d, exiting", args.max_cycles)
            break
        if _stop:
            break
        # Cancellable sleep.
        end = time.time() + args.interval
        while not _stop and time.time() < end:
            time.sleep(min(1.0, end - time.time()))

    logger.info("demo loop: stopped after %d cycles", cycle)
    return 0


if __name__ == "__main__":
    sys.exit(main())
