"""
Ambush close monitor — skill-side trailing / max_hold exit driver.

The server enforces the hard stop_loss floor via the
AmbushStopLossWatcher (cmd/server starts it; engine/AmbushStopLossWatcher
loop reads agent_trade_source_positions every 30s and calls
CloseRealtimePositionV2 when entry × (1 ± stop_loss_pct / leverage) is
breached). This module handles the other two exit rules, which depend
on per-position drift state the server doesn't track:

  • trailing_pct — close when mark retraces ≥ trailing_pct from the
    peak (long) / trough (short) since open.
  • max_hold_hours — close when wall-clock since opened_at exceeds the
    side-specific max_hold_hours.

Architecture:
  - 30s tick (cadence chosen to match server stop_loss; per-symbol
    we'd rather over-trigger via a tight trailing than under-trigger).
  - Each tick: list bot positions via TradingClient.get_positions().
    Position record carries mark_price + entry_price + symbol + net_qty.
  - For each non-flat position:
      * Look up local position_state row.
      * Bootstrap on first sight: row missing → write opened_at=now,
        peak=mark. Trailing ratchet starts fresh (worst case: skill
        restart loses historical peak; we never close prematurely).
      * Ratchet peak:  long → max(peak, mark);  short → min(peak, mark).
      * Compute drift:  long → (peak - mark) / peak;
                       short → (mark - peak) / peak.
      * If drift ≥ trailing_pct → close with reason "trailing".
      * Else if (now - opened_at) ≥ max_hold_hours → close with reason
        "max_hold".
  - For symbols no longer in the open positions list, drop the
    local row (cleaning up after a close that came from any channel:
    skill itself, server stop_loss watcher, or manual API).

Failures (network blip on get_positions, parse error, close error)
are logged + continue — never abort the loop. A single bad close
attempt does not block other positions.
"""

from __future__ import annotations

import asyncio
import logging
import time
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from typing import Optional

from ambush import hyperliquid_candles, indicators, live_database as db
from trading_client import TradingClient


logger = logging.getLogger("ambush.close_monitor")


# Default tick cadence. 15 min (one K-line cycle) per 2026-05-20 user
# direction. The server-side stop_loss watcher was removed the same
# day, so this tick is the ONLY thing that fires automatic closes on
# trailing/max_hold triggers — worst-case reaction time for an open
# position is one tick. Combined with 3x leverage and 168h SHORT
# max_hold, a 15min adverse move can be material; risk accepted per
# user profile (see docs/superpowers/audits/2026-05-18_ambush_implementation_audit.md
# §6 changelog entry for 2026-05-20).
DEFAULT_TICK_SECONDS = 15 * 60


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parse_decimal(raw, default: Decimal = Decimal("0")) -> Decimal:
    try:
        return Decimal(str(raw))
    except (InvalidOperation, ValueError, TypeError):
        return default


def _parse_iso(raw: str) -> Optional[datetime]:
    try:
        return datetime.fromisoformat(raw)
    except (ValueError, TypeError):
        return None


def _side_params(ambush_params: dict, side: str) -> dict:
    if side == "long":
        return ambush_params.get("long_params", {}) or {}
    return ambush_params.get("short_params", {}) or {}


def evaluate_exit(
    side: str,
    entry_price: Decimal,
    peak_price: Decimal,
    mark_price: Decimal,
    opened_at: datetime,
    now: datetime,
    trailing_pct: Decimal,
    max_hold_hours: Decimal,
) -> tuple[Optional[str], Decimal]:
    """Return (exit_reason, new_peak). exit_reason in {None, "trailing", "max_hold"}.

    new_peak is the ratcheted peak/trough — caller persists it whether
    or not an exit fires (so the ratchet advances steadily).
    """
    if side == "long":
        new_peak = peak_price if peak_price >= mark_price else mark_price
        if new_peak > 0 and trailing_pct > 0:
            drift = (new_peak - mark_price) / new_peak
            if drift >= trailing_pct:
                return "trailing", new_peak
    else:  # short
        # First-sight bootstrap may have set peak to mark; for a short
        # the "peak" is the lowest mark observed (= the most favorable).
        new_peak = peak_price if peak_price <= mark_price else mark_price
        if new_peak > 0 and trailing_pct > 0:
            drift = (mark_price - new_peak) / new_peak
            if drift >= trailing_pct:
                return "trailing", new_peak

    if max_hold_hours > 0:
        held_seconds = (now - opened_at).total_seconds()
        if held_seconds >= float(max_hold_hours) * 3600:
            return "max_hold", new_peak

    return None, new_peak


async def run_close_monitor(
    client: TradingClient,
    db_path: str,
    ambush_params: dict,
    stop_event: asyncio.Event,
    *,
    tick_seconds: int = DEFAULT_TICK_SECONDS,
    kline_driven: bool = False,
) -> None:
    """asyncio entry — runs until stop_event is set.

    Wrapped in a top-level try/except so any unexpected raise out of the
    inner await chain is surfaced loudly instead of being swallowed by
    asyncio.gather(..., return_exceptions=True) in the runner.

    When ``kline_driven=True``, each position is evaluated via the
    K-line cascade (_evaluate_position_kline_driven) which fetches live
    15m bars from Hyperliquid and applies priority-ordered exit logic
    (ATR stop_loss → trailing → max_hold → signal_reverse). Falls back
    to the legacy intratick path on fetch failure.
    """
    logger.info(
        "ambush close_monitor: started tick=%ds kline_driven=%s (skill owns trailing+max_hold; "
        "server owns stop_loss floor)",
        tick_seconds, kline_driven,
    )
    tick_count = 0
    heartbeat_every = max(1, int(60 / max(1, tick_seconds)))  # ~1/min
    try:
        while not stop_event.is_set():
            try:
                await asyncio.wait_for(stop_event.wait(), timeout=tick_seconds)
                break
            except asyncio.TimeoutError:
                pass
            try:
                await asyncio.get_event_loop().run_in_executor(
                    None, _run_tick, client, db_path, ambush_params, kline_driven
                )
            except Exception as e:
                logger.exception("ambush close_monitor: tick failed: %s", e)
            tick_count += 1
            if tick_count % heartbeat_every == 0:
                logger.info("ambush close_monitor: heartbeat ticks=%d", tick_count)
    except BaseException:
        logger.exception("ambush close_monitor: fatal — coroutine exiting")
        raise
    logger.info("ambush close_monitor: stopped")


def _run_tick(
    client: TradingClient,
    db_path: str,
    ambush_params: dict,
    kline_driven: bool = False,
) -> None:
    """One synchronous tick — get positions, evaluate, close on hit."""
    try:
        positions = client.get_positions() or []
    except Exception as e:
        logger.warning("ambush close_monitor: get_positions failed: %s", e)
        return

    # Build the set of symbols currently held with non-zero qty.
    open_symbols = set()
    for pos in positions:
        net_qty = _parse_decimal(pos.get("net_qty", "0"))
        if net_qty == 0:
            continue
        symbol = (pos.get("symbol") or "").upper()
        if not symbol:
            continue
        open_symbols.add(symbol)
        _evaluate_position(
            client, db_path, ambush_params, pos, symbol, net_qty,
            kline_driven=kline_driven,
        )

    # Garbage-collect local rows for symbols no longer open. Two close
    # channels reach here:
    #   1. close_monitor itself closed via trailing/max_hold above — the
    #      `record_symbol_action("close", ...)` call inside _close_now
    #      already recorded that event, so we'd double-record if we did it
    #      again here. Avoid double-counting by checking the last action.
    #   2. Externally closed: server stop_loss watcher, manual API close,
    #      ambush exit_signal handler, etc. In these cases no skill-side
    #      `close` row exists yet, and `same_coin_dedup_days` would stay
    #      armed forever (gate compares against last `open` ignoring whether
    #      a close happened). Backfill a synthetic close row so the dedup
    #      gate disarms naturally on the next event for this base.
    for row in db.list_position_states(db_path):
        symbol = row["symbol"]
        if symbol in open_symbols:
            continue
        logger.info(
            "ambush close_monitor: %s no longer open elsewhere — clearing local state",
            symbol,
        )
        db.delete_position_state(db_path, symbol)
        base = symbol[:-4] if symbol.upper().endswith("USDC") else symbol
        try:
            last_action_at = db.last_action_at(db_path, base, "close")
            opened_at_row = db.last_action_at(db_path, base, "open")
        except Exception as e:
            logger.warning("ambush close_monitor: last_action_at probe failed for %s: %s", base, e)
            last_action_at = None
            opened_at_row = None
        # Only backfill if there is a recent `open` AND no `close` after it
        # — otherwise we'd write a spurious close (e.g. on skill restart
        # where the bot is already flat and the open was days ago and
        # already paired).
        needs_backfill = (
            opened_at_row is not None
            and (last_action_at is None or last_action_at < opened_at_row)
        )
        if not needs_backfill:
            continue
        try:
            db.record_symbol_action(
                db_path, base, "close", order_id=None,
            )
            logger.info(
                "ambush close_monitor: backfilled close action for %s (external close detected; rhythm gates disarmed)",
                base,
            )
        except Exception as e:
            logger.warning(
                "ambush close_monitor: backfill close failed for %s: %s",
                base, e,
            )


def _evaluate_position(
    client: TradingClient,
    db_path: str,
    ambush_params: dict,
    pos: dict,
    symbol: str,
    net_qty: Decimal,
    *,
    kline_driven: bool = False,
) -> None:
    """Dispatch to K-line cascade or legacy intratick path."""
    if not kline_driven:
        _evaluate_position_legacy(client, db_path, ambush_params, pos, symbol, net_qty)
        return
    try:
        _evaluate_position_kline_driven(client, db_path, ambush_params, pos, symbol, net_qty)
    except hyperliquid_candles.HyperliquidCandleError as e:
        logger.warning(
            "close_monitor: K-line fetch failed for %s (%s); falling back to legacy intratick path",
            symbol, e,
        )
        _evaluate_position_legacy(client, db_path, ambush_params, pos, symbol, net_qty)


def _evaluate_position_legacy(
    client: TradingClient,
    db_path: str,
    ambush_params: dict,
    pos: dict,
    symbol: str,
    net_qty: Decimal,
) -> None:
    side = "long" if net_qty > 0 else "short"
    mark = _parse_decimal(pos.get("mark_price", "0"))
    entry = _parse_decimal(pos.get("entry_price", "0"))
    if mark <= 0 or entry <= 0:
        return  # Bad data, skip — bootstrapping with a zero would corrupt peak.

    state = db.get_position_state(db_path, symbol)
    if state is None:
        # Bootstrap: first sight of this position. peak starts at mark.
        db.upsert_position_state(
            db_path,
            symbol=symbol,
            side=side,
            entry_price=str(entry),
            opened_at=_now_iso(),
            peak_price=str(mark),
        )
        logger.info(
            "ambush close_monitor: bootstrap %s side=%s entry=%s peak=%s",
            symbol, side, entry, mark,
        )
        return  # Next tick will start ratcheting.

    peak = _parse_decimal(state["peak_price"], default=mark)
    opened_at = _parse_iso(state["opened_at"]) or datetime.now(timezone.utc)
    side_cfg = _side_params(ambush_params, side)
    trailing_pct = _parse_decimal(side_cfg.get("trailing_pct", "0"))
    max_hold_hours = _parse_decimal(side_cfg.get("max_hold_hours", "0"))

    now = datetime.now(timezone.utc)
    exit_reason, new_peak = evaluate_exit(
        side=side, entry_price=entry, peak_price=peak, mark_price=mark,
        opened_at=opened_at, now=now,
        trailing_pct=trailing_pct, max_hold_hours=max_hold_hours,
    )

    # Always persist the ratcheted peak — even if no exit fires this
    # tick, the next tick should compare against the updated peak.
    if new_peak != peak:
        db.update_position_peak(db_path, symbol, str(new_peak))

    if exit_reason is None:
        return

    logger.info(
        "ambush close_monitor: EXIT %s side=%s reason=%s entry=%s peak=%s mark=%s "
        "trailing_pct=%s max_hold_h=%s held_s=%.0f",
        symbol, side, exit_reason, entry, new_peak, mark, trailing_pct,
        max_hold_hours, (now - opened_at).total_seconds(),
    )

    # Mutate client.symbol for the close call (same pattern as event_handler).
    prev_symbol = client.symbol
    client.symbol = symbol
    try:
        # close_position issues a reduce_only market IOC against current
        # position size. We don't pass decision_mark because
        # _submit_market_order doesn't support it — the server falls
        # back to the cached fresh mark price for ambush coins, which
        # the realtime market loop keeps refreshed via Reconcile().
        reasoning_en = "ambush_close: {}".format(exit_reason)
        result = client.close_position(
            reasoning="异动 close ({})".format(exit_reason),
            reasoning_en=reasoning_en,
        )
        logger.info(
            "ambush close_monitor: closed %s reason=%s result=%s",
            symbol, exit_reason, _summarize_close_result(result),
        )
        # Drop local row immediately — server's position.updated event
        # would also drop it (via the next-tick GC), but doing it here
        # avoids re-evaluating the now-flat position before that.
        db.delete_position_state(db_path, symbol)
        # Record close for downstream rhythm gates (cooldown_bars).
        # Symbol-to-base mapping: position_state.symbol is the USDC perp
        # (e.g. "SAGAUSDC"); strip suffix to match event_handler's
        # gate key (HL base "SAGA").
        base = symbol[:-4] if symbol.upper().endswith("USDC") else symbol
        order_id_str = None
        if isinstance(result, dict):
            order = result.get("order") or {}
            if isinstance(order, dict) and order.get("order_id"):
                order_id_str = str(order["order_id"])
        try:
            db.record_symbol_action(
                db_path, base, "close", order_id=order_id_str,
            )
        except Exception as e:
            logger.warning(
                "ambush close_monitor: failed to record close action for %s: %s",
                symbol, e,
            )
    except Exception as e:
        logger.warning(
            "ambush close_monitor: close failed for %s reason=%s err=%s",
            symbol, exit_reason, e,
        )
    finally:
        client.symbol = prev_symbol


def _evaluate_position_kline_driven(
    client: TradingClient,
    db_path: str,
    ambush_params: dict,
    pos: dict,
    symbol: str,
    net_qty: Decimal,
) -> None:
    """K-line cascade: fetch 15m bars, apply priority-ordered exit logic.

    Priority 1 (T8): ATR stop_loss.
    Priority 2 (T10-next): max_hold — placeholder comment, added next task.
    Priority 3 (T9, this task): K-line-close trailing — ratchets peak/trough
        on each tick using latest K-line close; fires _close_now("trailing")
        when drift from peak exceeds trailing_pct.
    Priority 4 (future): signal_reverse.
    """
    side = "long" if net_qty > 0 else "short"
    entry = _parse_decimal(pos.get("entry_price", "0"))
    if entry <= 0:
        return  # bad data, skip

    # HL base symbol = strip USDC suffix
    base = symbol[:-4] if symbol.upper().endswith("USDC") else symbol
    bars = hyperliquid_candles.fetch(base, interval="15m", limit=96)
    if len(bars) < 15:
        logger.warning(
            "close_monitor: %s only %d bars from HL (need 15+); falling back to legacy",
            symbol, len(bars),
        )
        _evaluate_position_legacy(client, db_path, ambush_params, pos, symbol, net_qty)
        return

    last_close = Decimal(str(bars[-1]["close"]))
    atr = Decimal(str(indicators.compute_atr(bars, period=14)))
    side_cfg = ambush_params.get(f"{side}_params", {}) or {}
    sl_atr_mult = Decimal(str(side_cfg.get("sl_atr_mult", 2.0 if side == "long" else 2.5)))

    # Priority 1: ATR stop_loss
    if atr > 0:
        sl_dist = sl_atr_mult * atr / entry
        if side == "long":
            pnl_pct = (last_close - entry) / entry
        else:
            pnl_pct = (entry - last_close) / entry
        leverage = int(pos.get("leverage") or 1)
        if pnl_pct <= -sl_dist:
            _close_now(
                client, db_path, symbol, side, "atr_stop_loss",
                extras={
                    "atr": str(atr), "sl_dist_pct": str(sl_dist),
                    "last_close": str(last_close), "entry": str(entry),
                },
            )
            return

    # max_hold — added in T10 at priority 2 (between ATR and trailing).
    # K-line-close trailing (priority 3 after T10; currently runs at slot 2).
    state = db.get_position_state(db_path, symbol)
    trailing_pct = Decimal(str(side_cfg.get("trailing_pct", 0.25)))
    if state is None:
        # Bootstrap with current K-line close as initial peak.
        db.upsert_position_state(
            db_path, symbol=symbol, side=side,
            entry_price=str(entry),
            opened_at=datetime.now(timezone.utc).isoformat(),
            peak_price=str(last_close),
        )
        logger.info(
            "close_monitor: bootstrap %s side=%s entry=%s peak=%s (K-line close)",
            symbol, side, entry, last_close,
        )
        return  # need at least one tick of history to evaluate trailing
    prev_peak = _parse_decimal(state.get("peak_price", "0"))
    if side == "long":
        new_peak = max(prev_peak, last_close) if prev_peak > 0 else last_close
        drift = (new_peak - last_close) / new_peak if new_peak > 0 else Decimal("0")
    else:
        new_peak = min(prev_peak, last_close) if prev_peak > 0 else last_close
        drift = (last_close - new_peak) / new_peak if new_peak > 0 else Decimal("0")
    # Ratchet first (write the updated peak), then test.
    if new_peak != prev_peak:
        db.update_position_peak(db_path, symbol, str(new_peak))
    if drift >= trailing_pct:
        _close_now(
            client, db_path, symbol, side, "trailing",
            extras={"peak": str(new_peak), "last_close": str(last_close),
                    "drift_pct": str(drift), "trailing_pct": str(trailing_pct)},
        )
        return


def _close_now(
    client: TradingClient,
    db_path: str,
    symbol: str,
    side: str,
    reason: str,
    *,
    extras: dict | None = None,
) -> None:
    """Place a close order, delete position state, and record symbol action.

    Wraps client.close_position in try/except — never crashes the monitor.
    Uses the call-scoped ``symbol=`` override on close_position (no
    client.symbol mutation needed).
    """
    extras_str = (" " + " ".join(f"{k}={v}" for k, v in extras.items())) if extras else ""
    logger.info(
        "close_monitor: closing %s side=%s reason=%s%s", symbol, side, reason, extras_str
    )
    result = None
    try:
        result = client.close_position(
            position_side=side,
            symbol=symbol,
            reasoning="异动 close ({})".format(reason),
            reasoning_en="ambush_close: {}".format(reason),
        )
        logger.info(
            "ambush close_monitor: closed %s reason=%s result=%s",
            symbol, reason, _summarize_close_result(result),
        )
    except Exception as e:
        logger.warning(
            "close_monitor: close failed %s reason=%s err=%s", symbol, reason, e
        )
        return
    db.delete_position_state(db_path, symbol)
    base = symbol[:-4] if symbol.upper().endswith("USDC") else symbol
    order_id_str = None
    if isinstance(result, dict):
        order = result.get("order") or {}
        if isinstance(order, dict) and order.get("order_id"):
            order_id_str = str(order["order_id"])
    try:
        db.record_symbol_action(db_path, base, "close", order_id=order_id_str)
    except Exception as e:
        logger.warning(
            "close_monitor: failed to record symbol_action close for %s: %s", base, e
        )


def _summarize_close_result(result) -> str:
    """Best-effort one-liner from TradingClient.close_position's adapted result."""
    if not isinstance(result, dict):
        return repr(result)[:120]
    order = result.get("order") or {}
    return "filled_qty={} avg={} status={}".format(
        order.get("filled_qty"), order.get("avg_fill_price"), order.get("status"),
    )
