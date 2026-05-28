#!/usr/bin/env python3
"""
Short-accumulator demo loop — keeps a single bot perpetually short on one
symbol, scaling the position so total notional stays inside a configurable
[low, high] band. Every tick adds something (per the "一直加仓" product
intent) — when the band ceiling is about to be breached, the add becomes
a trim instead, then a small re-add the next tick.

Each tick:
  1. List positions.
  2. If any position exists in the wrong direction (long) on the target
     symbol, full-close it first (resets to flat → next step opens fresh
     short).
  3. Compute current short notional = |net_qty| × mark_price. Compare to
     the band:

        notional > target_high   → POST reduce-only buy with notional
                                   (current - target_mid), then a small
                                   sell to satisfy "一直加仓".
        notional < target_low    → POST sell with notional
                                   (target_mid - current). Bootstraps
                                   from flat / sub-band into mid-band.
        target_low ≤ n ≤ high    → POST sell with notional add_increment
                                   (or as much of it as fits under high).

  4. Sleep until next tick. SIGINT/SIGTERM finish current tick + exit.

This bypasses the ambush skill entirely (same as demo_direct_loop.py).
Skill should be stopped to avoid its close_monitor reaping positions
that we want to keep accumulating.

Defaults are tuned for SAGA: ~$15k mid-band notional uses ~$5k margin
at 3× leverage, well inside a $10k test account.

Usage:
    setsid nohup .venv/bin/python -m ambush.demo_short_accumulator \\
        --creds /tmp/ambush_creds.json --interval 300 \\
        --symbol SAGA --target-low 10000 --target-high 20000 \\
        --add-increment 1000 --leverage 3 \\
        > /tmp/ambush_short_accumulator.log 2>&1 < /dev/null &
    disown
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


logger = logging.getLogger("ambush.demo_short_accumulator")


class HMACClient:
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


def close_position_full(client: HMACClient, symbol: str) -> dict:
    body = {
        "reasoning":    "short_accumulator: reset wrong-side position",
        "reasoning_en": "short_accumulator: reset wrong-side position",
    }
    return client.request(
        "POST",
        f"/api/v2/moss/agent/realtime/bots/{client.bot_id}/positions/{symbol}/close",
        body=body,
    )


def submit_order(
    client: HMACClient, symbol: str, side: str, notional: Decimal,
    leverage: int, reduce_only: bool, tag: str,
) -> dict:
    """side ∈ {buy,sell}; notional in USDC. Market IOC."""
    body = {
        "symbol":           symbol if symbol.endswith("USDC") else f"{symbol}USDC",
        "side":             side,
        "order_type":       "market",
        "time_in_force":    "ioc",
        "leverage":         int(leverage),
        "notional":         format(notional.quantize(Decimal("0.01")), "f"),
        "reduce_only":      reduce_only,
        "reasoning":        f"short_accumulator: {tag}",
        "reasoning_en":     f"short_accumulator: {tag}",
        "client_order_id":  f"short-acc-{int(time.time()*1000)}",
    }
    return client.request(
        "POST",
        f"/api/v2/moss/agent/realtime/bots/{client.bot_id}/orders",
        body=body,
    )


def _format_result(result) -> str:
    if not isinstance(result, dict):
        return f"unexpected={result!r}"
    code = result.get("code")
    if code:
        return f"FAIL code={code} msg={result.get('message')}"
    order = result.get("order") or {}
    return (
        f"OK order_id={order.get('order_id')} "
        f"status={order.get('status')} "
        f"qty={order.get('filled_qty')} "
        f"avg={order.get('avg_fill_price')}"
    )


def compute_current_short_notional(pos: dict) -> tuple[Decimal, Decimal, Decimal]:
    """Return (notional, qty_abs, mark_price). Negative net_qty = short.
    Returns (0,0,0) if pos is None or flat."""
    if not pos:
        return Decimal("0"), Decimal("0"), Decimal("0")
    try:
        qty = Decimal(str(pos.get("net_qty") or "0"))
        mark = Decimal(str(pos.get("mark_price") or "0"))
    except (InvalidOperation, ValueError):
        return Decimal("0"), Decimal("0"), Decimal("0")
    if mark <= 0:
        return Decimal("0"), abs(qty), Decimal("0")
    # Prefer the response's pre-computed notional when present (matches
    # server's view); fall back to |qty| × mark_price.
    server_notional_raw = pos.get("notional")
    if server_notional_raw is not None:
        try:
            return abs(Decimal(str(server_notional_raw))), abs(qty), mark
        except (InvalidOperation, ValueError):
            pass
    return abs(qty) * mark, abs(qty), mark


# ── tick logic ────────────────────────────────────────────────────────────


_stop = False


def _signal_handler(signum, _frame):
    global _stop
    logger.info("short accumulator: signal %s received, exiting after current tick", signum)
    _stop = True


def run_one_tick(
    client: HMACClient,
    symbol: str,
    leverage: int,
    target_low: Decimal,
    target_high: Decimal,
    target_mid: Decimal,
    add_increment: Decimal,
    min_action_notional: Decimal,
) -> None:
    target_symbol = symbol if symbol.endswith("USDC") else f"{symbol}USDC"

    # 1. Inspect current positions; find the one on our target symbol
    #    and close any unrelated symbols (shouldn't happen on a clean
    #    bot but be defensive about leftover positions from prior modes).
    positions = get_positions(client)
    target_pos = None
    for p in positions:
        sym = str(p.get("symbol") or "").upper()
        if sym == target_symbol:
            target_pos = p
        else:
            # Wrong symbol — flush it. Single_position_lock would already
            # reject a new open while this is held, so we must clean first.
            logger.warning("short accumulator: flushing stray position %s qty=%s",
                           sym, p.get("net_qty"))
            r = close_position_full(client, sym)
            logger.info("short accumulator: stray close %s → %s", sym, _format_result(r))

    # Re-fetch in case we just closed something — server position state
    # advances atomically post-fill, so the next reads see flat.
    if target_pos is None or (positions and any(
        str(p.get("symbol") or "").upper() != target_symbol for p in positions
    )):
        positions = get_positions(client)
        target_pos = next(
            (p for p in positions if str(p.get("symbol") or "").upper() == target_symbol),
            None,
        )

    # 2. If the target symbol position is currently LONG, close it fully
    #    so we can rebuild on the SHORT side. Then re-read.
    if target_pos is not None:
        try:
            qty = Decimal(str(target_pos.get("net_qty") or "0"))
        except (InvalidOperation, ValueError):
            qty = Decimal("0")
        if qty > 0:
            logger.info("short accumulator: target %s is LONG qty=%s — closing to flip",
                        target_symbol, qty)
            r = close_position_full(client, target_symbol)
            logger.info("short accumulator: long-close %s → %s",
                        target_symbol, _format_result(r))
            target_pos = None  # treat as flat for the next step

    # 3. Compute current short notional + decide action.
    notional, qty_abs, mark = compute_current_short_notional(target_pos)
    logger.info(
        "short accumulator: %s current short notional=%s qty=%s mark=%s (band=[%s, %s])",
        target_symbol, notional, qty_abs, mark, target_low, target_high,
    )

    if notional > target_high:
        # Over the band — trim back to mid. We use a reduce-only BUY
        # (closing the short partially); excess is what we want gone.
        excess = (notional - target_mid).quantize(Decimal("0.01"))
        if excess >= min_action_notional:
            logger.info("short accumulator: over band, trimming notional=%s back toward mid",
                        excess)
            r = submit_order(client, target_symbol, "buy", excess, leverage,
                             reduce_only=True, tag=f"trim_to_mid n={excess}")
            logger.info("short accumulator: trim → %s", _format_result(r))
        else:
            logger.info("short accumulator: over band but excess<%s — skipping",
                        min_action_notional)
        return

    if notional < target_low:
        # Under the band — push toward mid.
        delta = (target_mid - notional).quantize(Decimal("0.01"))
        logger.info("short accumulator: under band, adding notional=%s toward mid", delta)
        r = submit_order(client, target_symbol, "sell", delta, leverage,
                         reduce_only=False, tag=f"build_to_mid n={delta}")
        logger.info("short accumulator: build → %s", _format_result(r))
        return

    # In-band — add one increment (capped by remaining headroom).
    headroom = (target_high - notional).quantize(Decimal("0.01"))
    add = min(add_increment, headroom)
    if add < min_action_notional:
        logger.info(
            "short accumulator: in-band notional=%s, headroom=%s < min_action %s — skipping",
            notional, headroom, min_action_notional,
        )
        return
    logger.info("short accumulator: in-band, adding increment notional=%s (headroom=%s)",
                add, headroom)
    r = submit_order(client, target_symbol, "sell", add, leverage,
                     reduce_only=False, tag=f"add_increment n={add}")
    logger.info("short accumulator: add → %s", _format_result(r))


def main() -> int:
    parser = argparse.ArgumentParser(description="Short-only accumulator demo loop")
    parser.add_argument("--creds", required=True, help="Bot HMAC creds JSON")
    parser.add_argument("--interval", type=int, default=300, help="Seconds between cycles (default 300)")
    parser.add_argument("--symbol", default="SAGA", help="HL symbol (default SAGA)")
    parser.add_argument("--target-low", default="10000", help="Lower band of notional (default 10000 USDC)")
    parser.add_argument("--target-high", default="20000", help="Upper band of notional (default 20000 USDC)")
    parser.add_argument("--add-increment", default="1000", help="Per-cycle add when in-band (default 1000 USDC)")
    parser.add_argument("--leverage", type=int, default=3, help="Leverage (default 3)")
    parser.add_argument(
        "--min-action-notional", default="20",
        help="Minimum notional per order (server has its own floor; default 20 USDC)",
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

    target_low = Decimal(args.target_low)
    target_high = Decimal(args.target_high)
    if target_high <= target_low:
        logger.error("--target-high must be > --target-low")
        return 2
    target_mid = ((target_low + target_high) / 2).quantize(Decimal("0.01"))
    add_increment = Decimal(args.add_increment)
    min_action_notional = Decimal(args.min_action_notional)

    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, _signal_handler)

    logger.info(
        "short accumulator: started bot_id=%s symbol=%s interval=%ds "
        "band=[%s,%s] mid=%s add=%s lev=%d",
        client.bot_id, args.symbol, args.interval,
        target_low, target_high, target_mid, add_increment, args.leverage,
    )

    cycle = 0
    while not _stop:
        try:
            run_one_tick(
                client, args.symbol, args.leverage,
                target_low, target_high, target_mid,
                add_increment, min_action_notional,
            )
        except Exception as e:
            logger.exception("short accumulator: tick raised %s — continuing", e)
        cycle += 1
        if args.max_cycles and cycle >= args.max_cycles:
            logger.info("short accumulator: reached --max-cycles=%d, exiting", args.max_cycles)
            break
        if _stop:
            break
        end = time.time() + args.interval
        while not _stop and time.time() < end:
            time.sleep(min(1.0, end - time.time()))

    logger.info("short accumulator: stopped after %d cycles", cycle)
    return 0


if __name__ == "__main__":
    sys.exit(main())
