#!/usr/bin/env python3
"""
Direct-API trade loop for the demo bot — bypasses the ambush skill and
hits the bot's /orders + /positions/:symbol/close endpoints directly. Use
this when you want a predictable trade cadence on a bot regardless of
the skill's own decision logic (e.g., for follower QA where the skill's
own close_monitor would otherwise close positions on tight trailing
before the colleague's follower service can replicate them).

Each tick:
  1. List positions; for any non-flat symbol, POST close (reduce-only
     market IOC). Skipped if the bot is already flat.
  2. Open a new position via POST /orders with the next (symbol, side)
     from the round-robin lists. Notional = wallet_balance × position_pct.
  3. Sleep until next tick. SIGINT/SIGTERM finish current tick + exit.

This script does NOT touch the skill. The skill keeps running and its
close_monitor / handler are bystanders — the only thing it will react
to is real ambush.detected events (none are injected here).

If the skill's close_monitor is configured with a tight trailing_pct,
it WILL close positions opened by this script almost immediately. To
have persistent positions you must either:
  - Stop the skill (kill its live_runner process), OR
  - Restart the skill with a relaxed --params-file (trailing_pct ≥ 0.10).

Usage:
    python -m ambush.demo_direct_loop --creds /tmp/ambush_creds.json \\
        --interval 300 --notional 50 --leverage 3 \\
        --symbols SAGA --side-pattern short,long
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


logger = logging.getLogger("ambush.demo_direct_loop")


DEFAULT_SYMBOLS = ["SAGA"]
DEFAULT_SIDES   = ["short", "long"]


class HMACClient:
    """Minimal HMAC client identical to demo_loop.HMACClient — stays
    self-contained so this script has zero deps on the skill package."""

    def __init__(self, creds: dict):
        self.api_key = creds["api_key"]
        self.api_secret = creds["api_secret"]
        self.base_url = creds["base_url"].rstrip("/")
        self.bot_id = creds["bot_id"]

    def _sign(self, method: str, path: str, query: str, body: str) -> tuple[str, str, str]:
        ts = str(int(time.time()))
        nonce = secrets.token_hex(8)
        sign_str = f"{method}\n{path}\n{query}\n{body}\n{ts}\n{nonce}"
        sig = hmac.new(self.api_secret.encode(), sign_str.encode(), hashlib.sha256).hexdigest()
        return ts, nonce, sig

    def request(self, method: str, path: str, *, query: str = "", body: dict | None = None) -> dict:
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
    resp = client.request("GET", f"/api/v2/moss/agent/realtime/bots/{client.bot_id}/positions")
    items = resp.get("items") or [] if isinstance(resp, dict) else []
    out = []
    for p in items:
        try:
            qty = Decimal(str(p.get("net_qty") or "0"))
        except (InvalidOperation, ValueError):
            qty = Decimal("0")
        if qty != 0:
            out.append(p)
    return out


def get_account(client: HMACClient) -> dict:
    resp = client.request("GET", f"/api/v2/moss/agent/realtime/bots/{client.bot_id}/account")
    return resp if isinstance(resp, dict) else {}


def close_position(client: HMACClient, symbol: str) -> dict:
    """POST /positions/:symbol/close — reduce-only market IOC."""
    body = {
        "reasoning":    "demo_direct_loop: rotating to next trade",
        "reasoning_en": "demo_direct_loop: rotating to next trade",
    }
    return client.request(
        "POST",
        f"/api/v2/moss/agent/realtime/bots/{client.bot_id}/positions/{symbol}/close",
        body=body,
    )


def open_position(
    client: HMACClient, symbol: str, side: str, notional: str, leverage: int,
) -> dict:
    """POST /orders — open via market IOC. side ∈ {long,short} → buy/sell."""
    market_side = "buy" if side == "long" else "sell"
    body = {
        "symbol":           symbol if symbol.endswith("USDC") else f"{symbol}USDC",
        "side":             market_side,
        "order_type":       "market",
        "time_in_force":    "ioc",
        "leverage":         int(leverage),
        "notional":         notional,
        "reduce_only":      False,
        "reasoning":        f"demo_direct_loop: open {side} {symbol}",
        "reasoning_en":     f"demo_direct_loop: open {side} {symbol}",
        "client_order_id":  f"demo-direct-{int(time.time())}",
    }
    return client.request(
        "POST",
        f"/api/v2/moss/agent/realtime/bots/{client.bot_id}/orders",
        body=body,
    )


# ── main loop ─────────────────────────────────────────────────────────────


_stop = False


def _signal_handler(signum, _frame):
    global _stop
    logger.info("demo direct loop: signal %s received, exiting after current tick", signum)
    _stop = True


def _format_order_result(result) -> str:
    """One-line audit summary for an open/close result envelope."""
    if not isinstance(result, dict):
        return f"unexpected={result!r}"
    code = result.get("code")
    if code:
        return f"FAIL code={code} msg={result.get('message')}"
    order = result.get("order") or {}
    return (
        f"OK order_id={order.get('order_id')} "
        f"status={order.get('status')} "
        f"filled_qty={order.get('filled_qty')} "
        f"avg={order.get('avg_fill_price')} "
        f"realized_pnl={result.get('realized_pnl')}"
    )


def run_one_tick(
    client: HMACClient, symbol: str, side: str, notional: str, leverage: int,
) -> None:
    # 1. Close any existing position.
    open_now = get_positions(client)
    if open_now:
        for pos in open_now:
            sym = str(pos.get("symbol") or "")
            if not sym:
                continue
            logger.info("demo direct loop: closing existing %s qty=%s", sym, pos.get("net_qty"))
            result = close_position(client, sym)
            logger.info("demo direct loop: close %s → %s", sym, _format_order_result(result))
    else:
        logger.info("demo direct loop: bot flat — no close needed")

    # 2. Open new position.
    target_symbol = symbol if symbol.endswith("USDC") else f"{symbol}USDC"
    logger.info("demo direct loop: opening %s %s notional=%s lev=%d",
                target_symbol, side, notional, leverage)
    result = open_position(client, symbol, side, notional, leverage)
    logger.info("demo direct loop: open %s/%s → %s",
                target_symbol, side, _format_order_result(result))


def main() -> int:
    parser = argparse.ArgumentParser(description="Direct-API demo trade loop (bypasses skill)")
    parser.add_argument("--creds", required=True, help="Bot HMAC creds JSON")
    parser.add_argument("--interval", type=int, default=300, help="Seconds between cycles (default 300)")
    parser.add_argument(
        "--symbols", default=",".join(DEFAULT_SYMBOLS),
        help="Comma-separated HL symbols (without USDC suffix). Default SAGA.",
    )
    parser.add_argument(
        "--side-pattern", default=",".join(DEFAULT_SIDES),
        help="Comma-separated rotation. Default 'short,long' alternating.",
    )
    parser.add_argument(
        "--notional", default="50",
        help="USDC notional per open (default 50). Server enforces minimum + lot size.",
    )
    parser.add_argument(
        "--leverage", type=int, default=3,
        help="Leverage per open (default 3, matches HL ambush low-cap cap).",
    )
    parser.add_argument("--max-cycles", type=int, default=0, help="Stop after N cycles (0=forever)")
    parser.add_argument("--log-level", default="INFO")
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
        logger.error("--symbols and --side-pattern must each have at least one entry")
        return 2

    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, _signal_handler)

    logger.info(
        "demo direct loop: started bot_id=%s interval=%ds symbols=%s sides=%s notional=%s lev=%d",
        client.bot_id, args.interval, symbols, sides, args.notional, args.leverage,
    )

    cycle = 0
    while not _stop:
        sym  = symbols[cycle % len(symbols)]
        side = sides[cycle % len(sides)]
        try:
            run_one_tick(client, sym, side, args.notional, args.leverage)
        except Exception as e:
            logger.exception("demo direct loop: tick raised %s — continuing", e)
        cycle += 1
        if args.max_cycles and cycle >= args.max_cycles:
            logger.info("demo direct loop: reached --max-cycles=%d, exiting", args.max_cycles)
            break
        if _stop:
            break
        end = time.time() + args.interval
        while not _stop and time.time() < end:
            time.sleep(min(1.0, end - time.time()))

    logger.info("demo direct loop: stopped after %d cycles", cycle)
    return 0


if __name__ == "__main__":
    sys.exit(main())
