"""
Per-event decision + order dispatch for ambush mode.

Called by both `ambush_ws.py` (WS live + bootstrap replay) and
`ambush_poller.py` (REST fallback). All routing into the local SQLite
state goes through `live_database` so dedup + decision audit + cursor
advance happen exactly once per event_id regardless of which channel
delivered it first.

Decision logic lives in `decision.py` (1:1 port of Go
`BalancedDecideV0`). This module is the glue between:

  raw event envelope (from WS / REST)
    → dedup check (events.event_id + decisions.event_id)
    → upsert events row
    → decide() → side (long/short/skip) + reason
    → on long/short: client.open_long() / .open_short() with notional
                     derived from current wallet_balance × position_pct
    → record_decision(decision, reason, order_id?, order_status, error?)
    → advance last_event_id_seen cursor

Single-position lock is enforced server-side via the `active_symbol`
column on the bot row (see `MaybeUpdateAmbushActiveSymbol` /
`MaybeClearAmbushActiveSymbolOnFlat`). The skill optimistically sends
the order; the server returns 423 LOCKED if another symbol is held,
which we record as `order_status='rejected'` and move on.
"""

from __future__ import annotations

import json
import logging
import time
import traceback
from decimal import Decimal, InvalidOperation
from typing import Any

from ambush import decision as decision_mod
from ambush import live_database as db
from trading_client import TradingClient


logger = logging.getLogger("ambush.handler")


# Server returns these HTTP-translated error codes when the bot is
# already holding a position on a different symbol. We don't retry —
# the lock is by design (1 position at a time per ambush bot).
_LOCK_REJECT_CODES = {
    "single_position_lock",
    "position_lock",
    "LOCKED",
}


class EventHandler:
    """Holds the per-bot context needed to convert an event into an order.

    The handler is constructed once per live-runner process and reused
    across WS bootstrap / WS live / REST poller channels. It is NOT
    thread-safe — callers serialize via `live_database._lock` plus
    asyncio event loop ordering.
    """

    def __init__(
        self,
        client: TradingClient,
        ambush_params: dict,
        *,
        max_notional_floor: float = 10.0,
    ):
        self.client = client
        self.ambush_params = ambush_params or {}
        self.max_notional_floor = float(max_notional_floor)

        # Cache side-specific config; falls back to safe defaults if the
        # bot was created with a partial params blob (shouldn't happen
        # post Phase 2 server validation, but be defensive).
        self.long_cfg = self.ambush_params.get("long_params") or {}
        self.short_cfg = self.ambush_params.get("short_params") or {}

    # ── per-side leverage / position_pct lookup ─────────────────────────

    def _leverage_for(self, side: str) -> int:
        cfg = self.long_cfg if side == decision_mod.SIDE_LONG else self.short_cfg
        try:
            lev = int(cfg.get("leverage", 1))
        except (TypeError, ValueError):
            lev = 1
        return max(1, lev)

    def _position_pct_for(self, side: str) -> Decimal:
        cfg = self.long_cfg if side == decision_mod.SIDE_LONG else self.short_cfg
        try:
            return Decimal(str(cfg.get("position_pct", "0")))
        except (InvalidOperation, TypeError, ValueError):
            return Decimal("0")

    # ── notional computation ────────────────────────────────────────────

    def _compute_notional(self, side: str) -> Decimal:
        """notional = wallet_balance × position_pct.

        Leverage controls only required margin (margin = notional/lev),
        which the server applies. We send the unleveraged face value as
        `notional` and the side's `leverage` separately. Floors at
        `max_notional_floor` to avoid below-minimum orders on tiny
        accounts.
        """
        try:
            account = self.client.get_account()
        except Exception as e:
            logger.warning("ambush handler: get_account failed: %s", e)
            return Decimal("0")
        if isinstance(account, dict) and account.get("code"):
            logger.warning("ambush handler: account error: %s", account)
            return Decimal("0")

        # Try common fields used by the realtime account endpoint.
        wallet_raw = (
            account.get("wallet_balance")
            or account.get("account_value")
            or account.get("equity")
            or "0"
        )
        try:
            wallet = Decimal(str(wallet_raw))
        except (InvalidOperation, TypeError, ValueError):
            wallet = Decimal("0")
        if wallet <= 0:
            return Decimal("0")

        pct = self._position_pct_for(side)
        if pct <= 0:
            return Decimal("0")

        notional = wallet * pct
        floor = Decimal(str(self.max_notional_floor))
        if notional < floor:
            notional = floor
        # Round to 2 decimals for cleaner order body.
        return notional.quantize(Decimal("0.01"))


def _format_event_brief(event: dict) -> str:
    """Short human-readable identifier for logs."""
    return (
        f"event_id={event.get('event_id')} "
        f"sym={event.get('hl_symbol')} "
        f"trigger_ts={event.get('trigger_ts')}"
    )


def _build_reasoning(event: dict, decision: decision_mod.Decision) -> tuple[str, str]:
    """Audit text saved on the order. Bilingual since v2 wire format
    requires both `reasoning` and `reasoning_en`."""
    sym = event.get("hl_symbol", "?")
    surge = event.get("surge_15m", "?")
    rsi = event.get("rsi_14", "?")
    chg = event.get("change_before_24h_pct", "?")
    zh = (
        f"异动币策略对 {sym} 触发 {decision.side} 决策："
        f"surge_15m={surge}, rsi_14={rsi}, 24h前涨跌={chg}%。"
        f"匹配规则 {decision.reason}。"
    )
    en = (
        f"Ambush strategy triggered {decision.side} on {sym}: "
        f"surge_15m={surge}, rsi_14={rsi}, 24h_prior_change={chg}%. "
        f"Matched rule {decision.reason}."
    )
    return zh, en


def _classify_order_status(result: Any) -> tuple[str, str | None, str | None]:
    """Extract (status, order_id, error_msg) from a TradingClient order
    response. Server returns `order_id` on success; on lock-reject it
    returns a structured error envelope with `code` set.
    """
    if not isinstance(result, dict):
        return "failed", None, f"unexpected order response: {result!r}"

    order_id = result.get("order_id")
    if order_id:
        return "placed", str(order_id), None

    # error envelope shape: {"code": "...", "message": "..."}
    code = str(result.get("code") or "")
    msg = str(result.get("message") or "")
    if code in _LOCK_REJECT_CODES or "single_position_lock" in msg.lower():
        return "rejected", None, msg or code or "single_position_lock"
    return "failed", None, msg or code or "unknown_error"


def process_event(
    handler: EventHandler,
    db_path: str,
    direction: str,
    event: dict,
    source: str,
) -> None:
    """Process exactly one event envelope. Idempotent on event_id.

    `source` is one of: "ws_bootstrap" / "ws_live" / "poll" — used for
    log tagging only; dedup is purely by event_id.

    On any unexpected exception we log + record an error decision but
    do NOT re-raise; the WS / poller outer loops are responsible for
    their own retry semantics and we don't want one bad event to kill
    the consumer.
    """
    try:
        event_id_raw = event.get("event_id")
        if event_id_raw is None:
            logger.warning("ambush handler: event missing event_id, dropping: %s", event)
            return
        try:
            event_id = int(event_id_raw)
        except (TypeError, ValueError):
            logger.warning("ambush handler: event has non-int id, dropping: %r", event_id_raw)
            return

        # Cursor advance happens unconditionally below so we don't fetch
        # the same event forever — even on dedup / skip / failure.

        if db.is_event_processed(db_path, event_id):
            logger.debug("ambush handler: %s already processed (src=%s), skipping",
                         _format_event_brief(event), source)
            db.set_last_event_id_seen(db_path, event_id)
            return

        # Persist the event row (idempotent INSERT OR IGNORE).
        try:
            db.upsert_event(db_path, event)
        except Exception as e:
            logger.warning("ambush handler: upsert_event failed for %s: %s",
                           _format_event_brief(event), e)
            # Continue — we can still decide + try to order; dedup just
            # has weaker durability for this event.

        decision = decision_mod.decision_from_event(event, direction)
        logger.info(
            "ambush handler: %s src=%s → %s/%s",
            _format_event_brief(event), source, decision.side, decision.reason,
        )

        # SKIP: persist decision and move on.
        if decision.side == decision_mod.SIDE_SKIP:
            db.record_decision(db_path, event_id, decision.side, decision.reason)
            db.mark_event_synced(db_path, event_id)
            db.set_last_event_id_seen(db_path, event_id)
            return

        # LONG / SHORT: compute notional + leverage + place order.
        notional = handler._compute_notional(decision.side)
        leverage = handler._leverage_for(decision.side)
        if notional <= 0:
            logger.warning(
                "ambush handler: %s decided %s but notional=0 (account empty?), recording as rejected",
                _format_event_brief(event), decision.side,
            )
            db.record_decision(
                db_path, event_id, decision.side, decision.reason,
                order_status="rejected", error_msg="notional_zero",
            )
            db.set_last_event_id_seen(db_path, event_id)
            return

        # Symbol comes from the event envelope; TradingClient.symbol is
        # single-symbol by construction so we mutate it for this call.
        sym = str(event.get("hl_symbol") or "").strip().upper()
        if not sym:
            logger.warning("ambush handler: %s missing hl_symbol, rejecting",
                           _format_event_brief(event))
            db.record_decision(
                db_path, event_id, decision.side, decision.reason,
                order_status="rejected", error_msg="empty_symbol",
            )
            db.set_last_event_id_seen(db_path, event_id)
            return

        # The realtime POST /orders endpoint expects ${BASE}USDC suffix
        # for HL symbols (e.g. "PEPE" → "PEPEUSDC"). hl_symbol on the
        # event envelope is already uppercase base; build the perp.
        prev_symbol = handler.client.symbol
        handler.client.symbol = sym if sym.endswith("USDC") else f"{sym}USDC"

        client_order_id = f"ambush-{event_id}-{int(time.time())}"
        reasoning_zh, reasoning_en = _build_reasoning(event, decision)

        try:
            if decision.side == decision_mod.SIDE_LONG:
                result = handler.client.open_long(
                    f"{notional:.2f}", leverage, client_order_id,
                    reasoning_zh, reasoning_en,
                )
            else:
                result = handler.client.open_short(
                    f"{notional:.2f}", leverage, client_order_id,
                    reasoning_zh, reasoning_en,
                )
        except Exception as e:
            logger.exception("ambush handler: order submit raised for %s: %s",
                             _format_event_brief(event), e)
            db.record_decision(
                db_path, event_id, decision.side, decision.reason,
                order_status="failed", error_msg=str(e),
            )
            db.set_last_event_id_seen(db_path, event_id)
            handler.client.symbol = prev_symbol
            return
        finally:
            # restore single-symbol client invariant for any later code
            # that might still depend on it (positions list filter etc).
            handler.client.symbol = prev_symbol

        status, order_id, err = _classify_order_status(result)
        logger.info(
            "ambush handler: %s order %s status=%s order_id=%s err=%s",
            _format_event_brief(event), decision.side, status, order_id, err,
        )

        db.record_decision(
            db_path, event_id, decision.side, decision.reason,
            order_id=order_id, order_status=status, error_msg=err,
        )
        if status == "placed":
            db.mark_event_synced(db_path, event_id)
        db.set_last_event_id_seen(db_path, event_id)

    except Exception as e:
        # Last-resort catchall so the WS / poller loops don't die.
        logger.error(
            "ambush handler: unexpected error processing event=%r: %s\n%s",
            event, e, traceback.format_exc(),
        )
