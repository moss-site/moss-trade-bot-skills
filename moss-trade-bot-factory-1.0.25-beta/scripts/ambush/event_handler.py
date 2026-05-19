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

Single-position lock is enforced in two layers:

1. Server side, atomically: `enforceAmbushPositionLock` against
   `agent_trade_realtime_bot_configs.active_symbol` (set/cleared by
   `MaybeUpdateAmbushActiveSymbol` / `MaybeClearAmbushActiveSymbolOnFlat`
   in the same DB transaction as the fill). This is the source of truth
   and cannot race — a second-symbol order is rejected with
   `single_position_lock` even if it arrives mid-fill.

2. Skill side (this file), as a pre-check: before submitting the
   open order, list the bot's current positions and if any non-zero
   net_qty exists on a different base asset, record the decision as
   `single_position_lock` and skip. This avoids round-tripping a
   reject through the order endpoint when we already know the bot is
   locked. Stale reads (the position closes between the check and the
   order POST) still hit the server-side atomic lock, so a stale skip
   does not allow an invalid order through.
"""

from __future__ import annotations

import json
import logging
import threading
import time
import traceback
from datetime import datetime, timedelta, timezone
from decimal import Decimal, InvalidOperation
from typing import Any

from ambush import decision as decision_mod
from ambush import live_database as db
from trading_client import TradingClient


logger = logging.getLogger("ambush.handler")


# Detected (open) signals are only actionable for one 15m K-line cycle.
# If we receive one >15 minutes after its trigger_ts — typically because
# the skill was offline and the signal arrived via bootstrap/poller replay
# — the market conditions that justified the open (surge / RSI / OI Z-
# score) have almost certainly changed. A genuinely persistent signal
# would have re-fired in the next K-line; absence of a fresher event
# means the conditions reverted. We skip late detected signals rather
# than open on stale data.
#
# Note: exit_signal is NOT subject to this check — a late close is still
# useful (any close > no close), and exit_signal.reason is event-time
# state that doesn't decay (oi_revert/max_hold are still meaningful at
# emit-time even if delivery is delayed).
_DETECTED_SIGNAL_STALE_AFTER_SECONDS = 15 * 60  # 15 min = 1 K-line cycle
_DEFERRED_OPEN_MAX_LATE_SECONDS = 15 * 60


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

        # Cache side-specific + rhythm config; falls back to safe defaults
        # if the bot was created with a partial params blob (shouldn't
        # happen post Phase 2 server validation, but be defensive).
        self.long_cfg = self.ambush_params.get("long_params") or {}
        self.short_cfg = self.ambush_params.get("short_params") or {}
        self.rhythm_cfg = self.ambush_params.get("rhythm") or {}

    def _side_cfg(self, side: str) -> dict:
        """Per-direction params bag."""
        return self.long_cfg if side == decision_mod.SIDE_LONG else self.short_cfg

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


def _parse_trigger_ts(event: dict) -> datetime | None:
    """Parse ISO-8601/RFC3339 trigger_ts from the wire envelope. Returns
    None if missing or malformed. Server emits with `Z` suffix; we accept
    both `Z` and `+00:00` to be defensive against any future encoder
    change."""
    raw = event.get("trigger_ts")
    if not raw:
        return None
    return _parse_iso_utc(raw)


def _parse_iso_utc(raw: Any) -> datetime | None:
    if not raw:
        return None
    try:
        text = str(raw).strip()
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        dt = datetime.fromisoformat(text)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except (TypeError, ValueError):
        return None


def _event_age_seconds(event: dict) -> float | None:
    """Wall-clock seconds since the event's trigger_ts. Returns None when
    trigger_ts is missing/unparseable."""
    trigger_dt = _parse_trigger_ts(event)
    if trigger_dt is None:
        return None
    return (datetime.now(timezone.utc) - trigger_dt).total_seconds()


def _format_event_age(event: dict) -> str:
    """Human-friendly age for log lines: `123s` / `15m` / `2h3m`."""
    age = _event_age_seconds(event)
    if age is None:
        return "unknown"
    return _format_seconds(age)


def _format_timedelta(td) -> str:
    """Format a timedelta as a short human-friendly string."""
    return _format_seconds(td.total_seconds())


def _format_seconds(s: float) -> str:
    if s < 60:
        return f"{s:.0f}s"
    if s < 3600:
        return f"{s/60:.1f}m"
    if s < 86400:
        return f"{s/3600:.1f}h"
    return f"{s/86400:.1f}d"


def _is_detected_signal_stale(event: dict) -> bool:
    """True when the event is older than `_DETECTED_SIGNAL_STALE_AFTER_SECONDS`.
    Defensive: returns False when trigger_ts is unparseable (don't skip
    on data we can't classify — let the normal decider run)."""
    age = _event_age_seconds(event)
    if age is None:
        return False
    return age > _DETECTED_SIGNAL_STALE_AFTER_SECONDS


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


# Error codes/keywords that suggest the close attempt failed for a
# transient reason — the position is still open and a redelivery
# (WS reconnect bootstrap or REST poller) should retry. We deliberately
# DO NOT mark the exit signal as processed for these.
#
# Anything not on this list is treated as a terminal failure (validation
# reject, permission denied, etc.) and gets marked processed with
# action=failed so we don't spin forever on an unfixable error.
_TRANSIENT_CLOSE_ERROR_CODES = {
    "market_recovering",
    "MARKET_RECOVERING",
    "market_is_recovering",
    "HTTP_ERROR",            # urllib HTTP error wrapper (5xx + network)
    "internal_error",
    "INTERNAL_ERROR",
    "service_unavailable",
    "SERVICE_UNAVAILABLE",
    "deadline_exceeded",
    "DEADLINE_EXCEEDED",
}
_TRANSIENT_CLOSE_ERROR_KEYWORDS = (
    "market is recovering",
    "market recovering",
    "timeout",
    "timed out",
    "connection",
    "temporarily",
)


def _classify_close_result(result: Any) -> tuple[str, str | None, str | None, bool]:
    """Inspect a TradingClient.close_position result.

    Returns (action, order_id, error_msg, retriable).

    Result shapes (from trading_client._submit_market_order via
    _adapt_order_result):
      Success → dict with `order` sub-dict carrying `order_id`; plus
                `realized_pnl`, top-level `order_id`/`fill_price`/etc.
      Failure → dict {"code": "...", "message": "..."} pass-through
                from `_request`'s HTTP-error branch (no `order` key).
                TradingClient does NOT raise on HTTP error; the caller
                must classify.

    Retriable semantics:
      - Success → False (don't retry success)
      - Terminal failure (validation, unknown code) → False (mark processed)
      - Transient failure (recovering / 5xx / timeout) → True (do NOT
        mark processed; next redelivery retries)
    """
    if not isinstance(result, dict):
        # Truly unexpected — treat as terminal so we don't infinite-loop.
        return "failed", None, f"unexpected close response: {result!r}", False

    # Success path: real order dict with an order_id.
    order = result.get("order")
    if isinstance(order, dict) and order.get("order_id"):
        order_id = str(order.get("order_id"))
        status = str(order.get("status") or "").lower()
        # Non-fill terminal statuses (rejected) are NOT a successful close
        # — flag as terminal failure with the order id for audit.
        if status and status not in ("filled", "partially_filled", ""):
            return "failed", order_id, f"order status={status}", False
        return "closed", order_id, None, False

    # Error envelope.
    code = str(result.get("code") or "")
    msg = str(result.get("message") or "")
    lower_msg = msg.lower()
    is_transient = (
        code in _TRANSIENT_CLOSE_ERROR_CODES
        or any(kw in lower_msg for kw in _TRANSIENT_CLOSE_ERROR_KEYWORDS)
    )
    err_text = msg or code or "unknown_error"
    return "failed", None, err_text, is_transient


def process_exit_signal(
    handler: EventHandler,
    db_path: str,
    signal: dict,
    source: str,
) -> None:
    """Handle one platform-emitted exit signal envelope.

    Backend decides WHEN to suggest closing (OI revert / 60d max_hold);
    skill decides WHETHER to fully close based on the live position state.
    For MVP: if a non-flat position exists on the signal's hl_symbol, we
    flatten immediately via close_position. No partial-close logic yet —
    the architecture leaves room (skill could throttle, scale-out, etc.)
    but the simplest correct behavior is to honor the signal.

    NO staleness check (in contrast to process_event for detected
    signals). A late exit signal is still meaningful: the platform
    already decided the cluster should be closed (OI reverted or 60d
    deadline). Delivering that decision late doesn't invalidate it —
    closing a position now is still strictly better than holding it
    against a thesis that was determined to be over. The longer the
    signal is delayed, the MORE we want to act on it, not less.

    Idempotent on exit_signal_id: bootstrap replays after a disconnect
    re-deliver signals we already acted on; INSERT OR IGNORE in
    exit_signal_processed makes the second call a no-op + we early-return
    before re-issuing the close. Position-existence check is the
    secondary safety: if we already closed the position via the same
    signal, get_positions() returns flat and we record action=no_position.

    `source` ∈ {"ws_bootstrap", "ws_live", "poll"} for log tagging only.
    """
    try:
        sig_id_raw = signal.get("exit_signal_id")
        if sig_id_raw is None:
            logger.warning("ambush handler: exit signal missing exit_signal_id, dropping: %s", signal)
            return
        try:
            exit_signal_id = int(sig_id_raw)
        except (TypeError, ValueError):
            logger.warning("ambush handler: exit signal non-int id, dropping: %r", sig_id_raw)
            return

        opening_event_id = int(signal.get("opening_event_id") or 0)
        hl_symbol_raw = str(signal.get("hl_symbol") or "").strip().upper()
        reason = str(signal.get("reason") or "unknown")

        if not hl_symbol_raw:
            logger.warning(
                "ambush handler: exit signal id=%d missing hl_symbol, dropping",
                exit_signal_id,
            )
            db.set_last_exit_signal_id_seen(db_path, exit_signal_id)
            return

        if db.is_exit_signal_processed(db_path, exit_signal_id):
            logger.debug(
                "ambush handler: exit_signal_id=%d already processed (src=%s), skipping",
                exit_signal_id, source,
            )
            db.set_last_exit_signal_id_seen(db_path, exit_signal_id)
            return

        target_symbol = hl_symbol_raw if hl_symbol_raw.endswith("USDC") else f"{hl_symbol_raw}USDC"

        # Find the open position for this symbol (if any). The signal is
        # advisory — if the skill has already closed elsewhere (manual /
        # close_monitor / server stop_loss), we just record no_position.
        try:
            positions = handler.client.get_positions() or []
        except Exception as e:
            logger.warning(
                "ambush handler: exit_signal_id=%d get_positions failed: %s — will retry on next signal",
                exit_signal_id, e,
            )
            # Do NOT mark as processed; let a redelivery try again.
            return

        matching = None
        for pos in positions:
            sym = str(pos.get("symbol") or "").upper()
            if sym != target_symbol:
                continue
            try:
                qty = Decimal(str(pos.get("net_qty") or "0"))
            except (InvalidOperation, ValueError):
                qty = Decimal("0")
            if qty == 0:
                continue
            matching = pos
            break

        if matching is None:
            logger.info(
                "ambush handler: exit_signal_id=%d sym=%s reason=%s src=%s — no open position, recording no_position",
                exit_signal_id, target_symbol, reason, source,
            )
            db.record_exit_signal_processed(
                db_path, exit_signal_id, opening_event_id, target_symbol,
                reason, "no_position",
            )
            db.set_last_exit_signal_id_seen(db_path, exit_signal_id)
            return

        # Flatten. close_position uses reduce_only market IOC against
        # the bot's current net position size. TradingClient does NOT
        # raise on HTTP error — it returns {"code","message"} — so the
        # except branch only catches client-side / ValueError raises
        # (no open position, qty parse fail). Server errors are
        # classified below by _classify_close_result.
        prev_symbol = handler.client.symbol
        handler.client.symbol = target_symbol
        raised = None
        try:
            result = handler.client.close_position(
                reasoning=f"异动 平台退出信号 ({reason})",
                reasoning_en=f"ambush_exit_signal: {reason}",
            )
        except Exception as e:
            raised = e
            result = None
            logger.exception(
                "ambush handler: exit_signal_id=%d close raised: %s",
                exit_signal_id, e,
            )
        finally:
            handler.client.symbol = prev_symbol

        if raised is not None:
            # Client-side raise (e.g., "no open position found" from
            # _resolve_open_position). Same effect as no_position: the
            # bot is already flat for this symbol — record as terminal,
            # advance cursor.
            err_str = str(raised)
            is_no_position = (
                "no open position" in err_str.lower()
                or "no position" in err_str.lower()
            )
            action = "no_position" if is_no_position else "failed"
            db.record_exit_signal_processed(
                db_path, exit_signal_id, opening_event_id, target_symbol,
                reason, action, error_msg=err_str,
            )
            db.set_last_exit_signal_id_seen(db_path, exit_signal_id)
            return

        action, order_id_str, err_text, retriable = _classify_close_result(result)

        if action == "closed":
            order_dict = result.get("order") if isinstance(result, dict) else {}
            logger.info(
                "ambush handler: exit_signal_id=%d sym=%s reason=%s src=%s — "
                "closed order_id=%s filled_qty=%s avg=%s realized_pnl=%s",
                exit_signal_id, target_symbol, reason, source, order_id_str,
                (order_dict or {}).get("filled_qty"),
                (order_dict or {}).get("avg_fill_price"),
                (result or {}).get("realized_pnl"),
            )
            db.record_exit_signal_processed(
                db_path, exit_signal_id, opening_event_id, target_symbol,
                reason, "closed", order_id=order_id_str,
            )
            # Record close for downstream rhythm gates (cooldown_bars).
            # target_symbol is the USDC perp; strip to HL base to match
            # the dedup key used at open time.
            base = target_symbol[:-4] if target_symbol.upper().endswith("USDC") else target_symbol
            try:
                db.record_symbol_action(
                    db_path, base, "close",
                    event_id=opening_event_id or None,
                    order_id=order_id_str,
                )
            except Exception as e:
                logger.warning(
                    "ambush handler: failed to record close action for exit_signal_id=%d: %s",
                    exit_signal_id, e,
                )
            db.set_last_exit_signal_id_seen(db_path, exit_signal_id)
            return

        # action == "failed". Retriable failures (market_recovering /
        # HTTP 5xx / timeout) stay UNACKNOWLEDGED: don't insert into
        # exit_signal_processed and don't advance the cursor. Next WS
        # bootstrap or poller tick will redeliver the same signal and
        # we'll retry the close. Terminal failures get marked processed
        # with action=failed so we don't infinite-loop on something
        # that won't fix itself.
        if retriable:
            logger.warning(
                "ambush handler: exit_signal_id=%d sym=%s reason=%s — close "
                "TRANSIENT failure: %s — leaving unacked for redelivery",
                exit_signal_id, target_symbol, reason, err_text,
            )
            return

        logger.warning(
            "ambush handler: exit_signal_id=%d sym=%s reason=%s — close "
            "TERMINAL failure: %s — marking processed (no retry)",
            exit_signal_id, target_symbol, reason, err_text,
        )
        db.record_exit_signal_processed(
            db_path, exit_signal_id, opening_event_id, target_symbol,
            reason, "failed", order_id=order_id_str, error_msg=err_text,
        )
        db.set_last_exit_signal_id_seen(db_path, exit_signal_id)

    except Exception as e:
        logger.error(
            "ambush handler: unexpected error processing exit_signal=%r: %s\n%s",
            signal, e, traceback.format_exc(),
        )


def _run_deferred_open(
    handler: EventHandler,
    db_path: str,
    event_id: int,
    decision,
    target_symbol: str,
    notional: Decimal,
    leverage: int,
    client_order_id: str,
    reasoning_zh: str,
    reasoning_en: str,
    delay_seconds: float,
    sym_base: str,
) -> None:
    """Background thread body: sleep `delay_seconds`, then fire the open
    order and update the decision audit row. Errors are logged but never
    propagated (this is a daemon thread; raises would kill it silently).

    No re-evaluation of the decision after the delay — see process_event
    comments for rationale and v2 plan."""
    try:
        db.mark_deferred_open_running(db_path, event_id)
        time.sleep(delay_seconds)
        logger.info(
            "ambush handler: deferred-open firing event_id=%d sym=%s side=%s "
            "(after %ds wait)",
            event_id, target_symbol, decision.side, delay_seconds,
        )
        prev_symbol = handler.client.symbol
        handler.client.symbol = target_symbol
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
        finally:
            handler.client.symbol = prev_symbol
        status, order_id, err = _classify_order_status(result)
        logger.info(
            "ambush handler: deferred-open event_id=%d %s status=%s order_id=%s err=%s",
            event_id, decision.side, status, order_id, err,
        )
        db.update_decision_outcome(
            db_path, event_id,
            order_id=order_id, order_status=status, error_msg=err,
        )
        if status == "placed":
            try:
                db.record_symbol_action(
                    db_path, sym_base, "open",
                    event_id=event_id, order_id=order_id,
                )
            except Exception as e:
                logger.warning(
                    "ambush handler: deferred-open failed to record action for event_id=%d: %s",
                    event_id, e,
                )
            db.mark_deferred_open_done(db_path, event_id, status="completed")
        else:
            db.mark_deferred_open_done(db_path, event_id, status=status, error_msg=err)
    except Exception as e:
        logger.exception(
            "ambush handler: deferred-open thread crashed for event_id=%d: %s",
            event_id, e,
        )
        try:
            db.update_decision_outcome(
                db_path, event_id,
                order_status="deferred_failed", error_msg=str(e),
            )
            db.mark_deferred_open_done(
                db_path, event_id, status="failed", error_msg=str(e),
            )
        except Exception:
            pass


def resume_deferred_opens(handler: EventHandler, db_path: str) -> int:
    """Re-spawn deferred opens left by a previous live_runner process.

    Old implementation used only daemon threads; restart during the sleep
    window left decisions stuck at `order_status='deferred_open'` forever.
    The SQLite queue lets startup recreate the remaining wait. If a due time
    is already more than one K-line late, expire it instead of opening on a
    stale delayed signal.
    """
    resumed = 0
    for row in db.list_pending_deferred_opens(db_path):
        event_id = int(row["event_id"])
        event = row["event"]
        due_at = _parse_iso_utc(row.get("due_at"))
        if due_at is None:
            db.update_decision_outcome(
                db_path, event_id,
                order_status="deferred_failed", error_msg="invalid deferred due_at",
            )
            db.mark_deferred_open_done(
                db_path, event_id, status="failed", error_msg="invalid due_at",
            )
            continue

        now = datetime.now(timezone.utc)
        late_by = (now - due_at).total_seconds()
        if late_by > _DEFERRED_OPEN_MAX_LATE_SECONDS:
            msg = f"deferred open expired, late_by={_format_seconds(late_by)}"
            logger.warning("ambush handler: deferred-open event_id=%d expired: %s", event_id, msg)
            db.update_decision_outcome(
                db_path, event_id,
                order_status="deferred_expired", error_msg=msg,
            )
            db.mark_deferred_open_done(db_path, event_id, status="expired", error_msg=msg)
            continue

        decision = decision_mod.Decision(str(row["decision"]), str(row["reason"]))
        notional = handler._compute_notional(decision.side)
        leverage = handler._leverage_for(decision.side)
        if notional <= 0:
            msg = "notional_zero during deferred resume"
            db.update_decision_outcome(
                db_path, event_id,
                order_status="rejected", error_msg=msg,
            )
            db.mark_deferred_open_done(db_path, event_id, status="rejected", error_msg=msg)
            continue

        sym = str(event.get("hl_symbol") or "").strip().upper()
        if not sym:
            msg = "empty_symbol during deferred resume"
            db.update_decision_outcome(
                db_path, event_id,
                order_status="rejected", error_msg=msg,
            )
            db.mark_deferred_open_done(db_path, event_id, status="rejected", error_msg=msg)
            continue

        target_symbol = sym if sym.endswith("USDC") else f"{sym}USDC"
        client_order_id = f"ambush-{event_id}-deferred"
        reasoning_zh, reasoning_en = _build_reasoning(event, decision)
        delay_seconds = max(0.0, (due_at - now).total_seconds())
        threading.Thread(
            target=_run_deferred_open,
            args=(handler, db_path, event_id, decision, target_symbol,
                  notional, leverage, client_order_id,
                  reasoning_zh, reasoning_en, delay_seconds, sym),
            daemon=True,
            name=f"ambush-defer-resume-{event_id}",
        ).start()
        resumed += 1
    if resumed:
        logger.info("ambush handler: resumed %d deferred open(s)", resumed)
    return resumed


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

        # Staleness gate — skip late open signals. The 5-rule decider
        # operates on snapshot features (surge_15m / rsi_14 / chg_24h)
        # frozen at event trigger time; acting on them later would be
        # opening on conditions that may no longer hold. The next dual_gate
        # cycle would have re-fired if conditions persisted, so silence
        # means the signal reverted. Record skip + advance cursor to keep
        # the event from being re-attempted forever.
        if _is_detected_signal_stale(event):
            age_str = _format_event_age(event)
            logger.warning(
                "ambush handler: %s src=%s — STALE (age=%s > %ds), skipping (delayed bootstrap/poll delivery)",
                _format_event_brief(event), source, age_str,
                _DETECTED_SIGNAL_STALE_AFTER_SECONDS,
            )
            db.record_decision(
                db_path, event_id, decision_mod.SIDE_SKIP, "stale_signal",
                error_msg=f"signal age {age_str} exceeded stale threshold {_DETECTED_SIGNAL_STALE_AFTER_SECONDS}s",
            )
            db.mark_event_synced(db_path, event_id)
            db.set_last_event_id_seen(db_path, event_id)
            return

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

        # ── Rhythm gates (LONG/SHORT only) ──────────────────────────────
        # These three params used to be declared in ambush_params + sent
        # over the wire but had zero runtime effect (see 2026-05-17 design
        # review). They're now enforced here as pre-trade gates.
        sym_for_gates = str(event.get("hl_symbol") or "").strip().upper()
        if sym_for_gates:
            # 1. same_coin_dedup_days — block re-open on the same coin
            #    within N days of the last open. Uses the skill-local
            #    symbol_action_history table; per-runner state (multi-skill
            #    deployment would need server-side `ambush_last_trigger`).
            try:
                dedup_days = int(handler.rhythm_cfg.get("same_coin_dedup_days", 0) or 0)
            except (TypeError, ValueError):
                dedup_days = 0
            if dedup_days > 0:
                last_open = db.last_action_at(db_path, sym_for_gates, "open")
                if last_open is not None:
                    age = datetime.now(timezone.utc) - last_open
                    if age.total_seconds() < dedup_days * 86400:
                        logger.info(
                            "ambush handler: %s src=%s → SKIP same_coin_dedup_days "
                            "(last open %s ago, window=%dd)",
                            _format_event_brief(event), source,
                            _format_timedelta(age), dedup_days,
                        )
                        db.record_decision(
                            db_path, event_id, decision_mod.SIDE_SKIP, "same_coin_dedup",
                            error_msg=f"last open {_format_timedelta(age)} ago, dedup_days={dedup_days}",
                        )
                        db.mark_event_synced(db_path, event_id)
                        db.set_last_event_id_seen(db_path, event_id)
                        return

            # 2. cooldown_bars (per direction) — block re-open within
            #    N×15min of the last close on the same coin. Avoids
            #    immediately re-engaging after the previous trade exited.
            side_cfg = handler._side_cfg(decision.side)
            try:
                cooldown_bars = int(side_cfg.get("cooldown_bars", 0) or 0)
            except (TypeError, ValueError):
                cooldown_bars = 0
            if cooldown_bars > 0:
                last_close = db.last_action_at(db_path, sym_for_gates, "close")
                if last_close is not None:
                    age = datetime.now(timezone.utc) - last_close
                    cooldown_sec = cooldown_bars * 15 * 60
                    if age.total_seconds() < cooldown_sec:
                        logger.info(
                            "ambush handler: %s src=%s → SKIP cooldown_bars "
                            "(last close %s ago, cooldown=%d bars = %dmin)",
                            _format_event_brief(event), source,
                            _format_timedelta(age), cooldown_bars, cooldown_bars * 15,
                        )
                        db.record_decision(
                            db_path, event_id, decision_mod.SIDE_SKIP, "cooldown_active",
                            error_msg=f"last close {_format_timedelta(age)} ago, cooldown_bars={cooldown_bars}",
                        )
                        db.mark_event_synced(db_path, event_id)
                        db.set_last_event_id_seen(db_path, event_id)
                        return

            # 3. max_trades_per_event — number of opens already recorded
            #    against this event_id. v1 supports max=1 only; values >1
            #    are validated by the server but treated as 1 here (full
            #    multi-trade requires close→cooldown→reopen flow, deferred
            #    to v2 per the 2026-05-17 design spec "Open Issues").
            try:
                max_trades = int(handler.rhythm_cfg.get("max_trades_per_event", 1) or 1)
            except (TypeError, ValueError):
                max_trades = 1
            if max_trades > 1:
                logger.warning(
                    "ambush handler: %s rhythm.max_trades_per_event=%d but skill v1 "
                    "only supports 1 — treating as 1 (event_id dedup prevents re-open)",
                    _format_event_brief(event), max_trades,
                )
            already_opened_count = db.count_event_trades(db_path, event_id)
            if already_opened_count >= 1:
                logger.info(
                    "ambush handler: %s src=%s → SKIP max_trades_per_event "
                    "(event_id already has %d open recorded; v1 cap=1)",
                    _format_event_brief(event), source, already_opened_count,
                )
                db.record_decision(
                    db_path, event_id, decision_mod.SIDE_SKIP, "max_trades_reached",
                    error_msg=f"event_id {event_id} already has {already_opened_count} trade(s); cap=1",
                )
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
        target_symbol = sym if sym.endswith("USDC") else f"{sym}USDC"
        handler.client.symbol = target_symbol

        # Single-position lock pre-check. The server enforces this
        # atomically via active_symbol; we replicate the rule here so we
        # don't blast an order we know will be rejected. List positions
        # and skip if any non-dust position exists on a different base
        # asset. A stale read is harmless — the server still rejects.
        try:
            existing_positions = handler.client.get_positions() or []
        except Exception as e:
            # Don't block the trade on a flaky read. Server-side lock
            # still catches an invalid attempt.
            logger.warning(
                "ambush handler: pre-check get_positions failed for %s: %s",
                _format_event_brief(event), e,
            )
            existing_positions = []
        for pos in existing_positions:
            raw_qty = pos.get("net_qty") if "net_qty" in pos else pos.get("size", "0")
            try:
                qty = Decimal(str(raw_qty or "0"))
            except (InvalidOperation, ValueError):
                continue
            if qty == 0:
                continue
            held_sym = str(pos.get("symbol") or "").upper()
            if not held_sym or held_sym == target_symbol:
                continue
            logger.info(
                "ambush handler: %s decided %s/%s on %s but bot already holds %s qty=%s — skipping (single_position_lock pre-check)",
                _format_event_brief(event), decision.side, decision.reason,
                target_symbol, held_sym, qty,
            )
            db.record_decision(
                db_path, event_id, decision.side, "single_position_lock",
                order_status="rejected",
                error_msg=f"already holding {held_sym} qty={qty}, cannot open {target_symbol}",
            )
            db.mark_event_synced(db_path, event_id)
            db.set_last_event_id_seen(db_path, event_id)
            handler.client.symbol = prev_symbol
            return

        client_order_id = f"ambush-{event_id}-{int(time.time())}"
        reasoning_zh, reasoning_en = _build_reasoning(event, decision)

        # ── Entry-delay (LONG: momentum_bars, SHORT: entry_delay_bars) ──
        # Defer the open by N × 15min to either confirm momentum (long) or
        # avoid the spike top (short). MVP semantics: pure time delay, NO
        # re-evaluation of decision after delay. The original event's
        # snapshot features are baked into the decision; if the market
        # turned in the intervening N bars, that's an acceptable loss
        # vs the alternative of always opening on impulse. Full
        # momentum-confirmation (re-fetching K-lines + re-running rule)
        # is v2 (tracked in design doc Open Issues).
        side_cfg_for_delay = handler._side_cfg(decision.side)
        delay_bars = 0
        delay_param_name = ""
        if decision.side == decision_mod.SIDE_LONG:
            try:
                delay_bars = int(side_cfg_for_delay.get("momentum_bars", 0) or 0)
            except (TypeError, ValueError):
                delay_bars = 0
            delay_param_name = "momentum_bars"
        elif decision.side == decision_mod.SIDE_SHORT:
            try:
                delay_bars = int(side_cfg_for_delay.get("entry_delay_bars", 0) or 0)
            except (TypeError, ValueError):
                delay_bars = 0
            delay_param_name = "entry_delay_bars"
        delay_seconds = max(0, delay_bars) * 15 * 60

        if delay_seconds > 0:
            logger.info(
                "ambush handler: %s %s/%s DEFERRED by %ds (%s=%d bars) — "
                "order will fire in background after delay",
                _format_event_brief(event), decision.side, decision.reason,
                delay_seconds, delay_param_name, delay_bars,
            )
            due_at = datetime.now(timezone.utc) + timedelta(seconds=delay_seconds)
            # Queue first, then write the decision row that makes this event
            # processed. If the process dies in between, replay can recreate
            # the same queue row via ON CONFLICT instead of leaving a
            # deferred_open decision without a resumable due time.
            db.enqueue_deferred_open(
                db_path,
                event_id=event_id,
                due_at=due_at.isoformat(),
                delay_param_name=delay_param_name,
                delay_bars=delay_bars,
            )
            db.record_decision(
                db_path, event_id, decision.side, decision.reason,
                order_status="deferred_open",
                error_msg=f"deferred {delay_seconds}s ({delay_param_name}={delay_bars} bars)",
            )
            db.mark_event_synced(db_path, event_id)
            db.set_last_event_id_seen(db_path, event_id)
            handler.client.symbol = prev_symbol
            # Spawn daemon thread for the actual deferred open. Thread can
            # be long-lived: 2026-05-18 sweep recommends entry_delay_bars=16
            # (4h) as default; upper bound now 32 bars (8h). The SQLite
            # deferred_opens row above lets live_runner restart re-spawn
            # any still-deferred order.
            threading.Thread(
                target=_run_deferred_open,
                args=(handler, db_path, event_id, decision, target_symbol,
                      notional, leverage, client_order_id,
                      reasoning_zh, reasoning_en, delay_seconds,
                      sym_for_gates or sym),
                daemon=True,
                name=f"ambush-defer-{event_id}",
            ).start()
            return

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
            # Record successful open for downstream rhythm gates
            # (same_coin_dedup_days, cooldown_bars, max_trades_per_event).
            # Symbol stored is the HL base (without USDC suffix) to match
            # the dedup gate keying.
            try:
                db.record_symbol_action(
                    db_path, sym_for_gates or sym, "open",
                    event_id=event_id, order_id=order_id,
                )
            except Exception as e:
                logger.warning(
                    "ambush handler: failed to record open action for %s: %s",
                    _format_event_brief(event), e,
                )
        db.set_last_event_id_seen(db_path, event_id)

    except Exception as e:
        # Last-resort catchall so the WS / poller loops don't die.
        logger.error(
            "ambush handler: unexpected error processing event=%r: %s\n%s",
            event, e, traceback.format_exc(),
        )
