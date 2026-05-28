"""
REST poller fallback for ambush events.

Mirrors hyperliquid-copy-trade `follow_service/moss_poller.py` —
this is the safety net for when the WS consumer is down or lossy.
Both `ambush_ws.py` and `ambush_poller.py` route through the same
`event_handler.process_event(...)`, and dedup happens in SQLite at
the events table, so duplicate delivery is harmless.

Loop:
  every `interval_sec` seconds:
    1. read cursor (last_event_id_seen) from SQLite
    2. GET /ambush-events?since=<cursor>&limit=<batch_size>
    3. for each returned event: process_event(...) (advances cursor)
"""

from __future__ import annotations

import asyncio
import logging

from ambush import ambush_client
from ambush import event_handler
from ambush import live_database as db


logger = logging.getLogger("ambush.poller")


async def run_poller(
    client: ambush_client.AmbushClient,
    db_path: str,
    direction: str,
    handler: event_handler.EventHandler,
    stop_event: asyncio.Event,
    *,
    interval_sec: float = 30.0,
    batch_size: int = 100,
) -> None:
    """Main poller loop. Returns when stop_event is set.

    `interval_sec` is deliberately ~2× the typical WS event cadence —
    in practice WS catches everything and this loop just confirms the
    cursor matches server tip. If WS is down, server max event lag is
    `interval_sec` (default 30s) before the poller fills in.
    """
    loop = asyncio.get_event_loop()
    while not stop_event.is_set():
        try:
            since = db.get_last_event_id_seen(db_path)
            since_exit = db.get_last_exit_signal_id_seen(db_path)
            resp = await loop.run_in_executor(
                None,
                lambda: client.get_events_since(
                    since_event_id=since,
                    since_exit_signal_id=since_exit,
                    limit=batch_size,
                ),
            )
            events = resp.get("items") if isinstance(resp, dict) else []
            exit_signals = resp.get("exit_signal_items") if isinstance(resp, dict) else []
            if events:
                logger.info(
                    "ambush poller: fetched %d events since id=%d",
                    len(events), since,
                )
                for ev in events:
                    if stop_event.is_set():
                        break
                    await loop.run_in_executor(
                        None,
                        event_handler.process_event,
                        handler, db_path, ev, "poll",
                    )
            else:
                logger.debug("ambush poller: no new events since id=%d", since)
            if exit_signals:
                logger.info(
                    "ambush poller: fetched %d exit signals since id=%d",
                    len(exit_signals), since_exit,
                )
                for sig in exit_signals:
                    if stop_event.is_set():
                        break
                    await loop.run_in_executor(
                        None,
                        event_handler.process_exit_signal,
                        handler, db_path, sig, "poll",
                    )
        except asyncio.CancelledError:
            logger.info("ambush poller: cancelled")
            return
        except Exception as e:
            logger.warning("ambush poller: tick error: %s", e)

        # Cancellable sleep — wake immediately on stop.
        try:
            await asyncio.wait_for(stop_event.wait(), timeout=interval_sec)
        except asyncio.TimeoutError:
            pass
