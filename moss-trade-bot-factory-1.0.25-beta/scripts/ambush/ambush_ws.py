"""
WebSocket consumer for ambush events.

Mirrors hyperliquid-copy-trade `follow_service/moss_ws.py` reconnect pattern:

  loop:
    1. fetch bootstrap (since=cursor) → catch up missed events
    2. process bootstrap.recent_events
    3. open WS with HMAC headers
    4. read ready frame; reset backoff
    5. stream events → for each: decide + maybe trade
    6. on disconnect: sleep with exponential backoff (5s → 60s), then loop

If WS errors out the poller fallback (`ambush_poller.py`) still keeps the
cursor moving forward as long as it's running.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time

import websockets

from ambush import ambush_client
from ambush import live_database as db
from ambush import event_handler


logger = logging.getLogger("ambush.ws")


async def run_ws(
    client: ambush_client.AmbushClient,
    db_path: str,
    direction: str,
    handler: event_handler.EventHandler,
    stop_event: asyncio.Event,
) -> None:
    """Main WS loop. Returns when stop_event is set.

    Errors fall through to exponential backoff. The bootstrap fetch on each
    iteration doubles as the catch-up mechanism for events missed during the
    disconnect window — server's bootstrap returns events with id > cursor
    up to a 24h / 500-row cap.
    """
    loop = asyncio.get_event_loop()
    backoff = 5.0
    while not stop_event.is_set():
        try:
            # 1. Bootstrap snapshot + replay missed events.
            since = db.get_last_event_id_seen(db_path)
            logger.info("ambush ws: fetching bootstrap since=%d", since)
            boot = await loop.run_in_executor(
                None, lambda: client.get_bootstrap(since_event_id=since)
            )
            server_seq = int(boot.get("server_event_sequence", 0))
            events = boot.get("recent_events") or []
            logger.info(
                "ambush ws: bootstrap ok, server_event_sequence=%d, replay_events=%d",
                server_seq, len(events),
            )

            # 2. Replay missed events through the same handler.
            for ev in events:
                await loop.run_in_executor(
                    None,
                    event_handler.process_event,
                    handler, db_path, direction, ev, "ws_bootstrap",
                )

            # 3. WS connect with HMAC headers.
            ws_url = client.build_ws_url()
            headers = client.build_ws_headers()
            logger.info("ambush ws: connecting %s ...", ws_url)
            async with websockets.connect(
                ws_url,
                additional_headers=headers,
                ping_interval=20,
                ping_timeout=10,
            ) as ws:
                # 4. Ready frame first.
                raw_ready = await ws.recv()
                ready = json.loads(raw_ready)
                if ready.get("type") != "ready":
                    logger.warning(
                        "ambush ws: expected ready frame, got type=%s",
                        ready.get("type"),
                    )
                    continue
                logger.info(
                    "ambush ws: ready bot=%s server_event_sequence=%d",
                    ready.get("bot_id"), ready.get("server_event_sequence", 0),
                )
                backoff = 5.0  # reset on successful connect

                # 5. Event stream.
                async for raw in ws:
                    if stop_event.is_set():
                        break
                    try:
                        msg = json.loads(raw)
                    except json.JSONDecodeError:
                        logger.warning("ambush ws: malformed JSON, skipping")
                        continue
                    if msg.get("type") != "ambush_event":
                        continue
                    ev = msg.get("event") or {}
                    if not ev:
                        continue
                    await loop.run_in_executor(
                        None,
                        event_handler.process_event,
                        handler, db_path, direction, ev, "ws_live",
                    )

        except asyncio.CancelledError:
            logger.info("ambush ws: cancelled")
            return
        except (websockets.ConnectionClosed, OSError) as e:
            if stop_event.is_set():
                return
            logger.warning("ambush ws: connection lost (%s), retry in %.1fs", e, backoff)
            await _sleep(backoff, stop_event)
            backoff = min(backoff * 2, 60.0)
        except Exception as e:
            if stop_event.is_set():
                return
            logger.exception("ambush ws: unexpected error: %s, retry in %.1fs", e, backoff)
            await _sleep(backoff, stop_event)
            backoff = min(backoff * 2, 60.0)


async def _sleep(seconds: float, stop_event: asyncio.Event) -> None:
    """Cancellable sleep — wakes on stop_event."""
    try:
        await asyncio.wait_for(stop_event.wait(), timeout=seconds)
    except asyncio.TimeoutError:
        pass
