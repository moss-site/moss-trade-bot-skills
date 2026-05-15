"""
Ambush-specific extensions on top of the shared TradingClient HMAC core.

Provides:
  - get_bootstrap(since=...) — initial snapshot + recent events catch-up
  - get_events_since(from_ts=..., since=...) — REST poller fallback
  - build_ws_url() + build_ws_headers() — auth headers for ws://.../ambush-events/ws

The HMAC signing (X-API-KEY / X-TS / X-NONCE / X-SIGNATURE) is the same as
TradingClient's regular agent-mode auth. Order placement (open_long /
open_short / close_position) reuses TradingClient as-is — see
`live_runner.py`.
"""

from __future__ import annotations

import urllib.parse

from trading_client import TradingClient


class AmbushClient:
    """Thin façade over TradingClient that adds 3 ambush-events endpoints.

    Holds a reference to a TradingClient (does NOT subclass it) so callers
    can keep using `tc.open_long(...)` / `tc.close_position(...)` directly
    while the AmbushClient handles the new event endpoints.
    """

    def __init__(self, tc: TradingClient, bot_id: str):
        self._tc = tc
        self._bot_id = bot_id
        if not bot_id:
            raise ValueError("AmbushClient requires bot_id (set after CreateBot)")

    @property
    def bot_id(self) -> str:
        return self._bot_id

    # ── Event endpoints (HMAC auth) ────────────────────────────────────────

    def get_bootstrap(
        self, since_event_id: int = 0, since_exit_signal_id: int = 0,
    ) -> dict:
        """GET /api/v2/moss/agent/realtime/bots/:bot_id/ambush-events/bootstrap.

        Returns: {
          "bot_id": str,
          "server_time": rfc3339,
          "server_event_sequence":        int,   # max ambush_event.id at query time
          "server_exit_signal_sequence":  int,   # max ambush_exit_signal.id at query time
          "watchlist_snapshot":           [...], # current observation set
          "recent_events":                [...], # detected events with id > since (24h, ≤500)
          "recent_exit_signals":          [...], # exit signals with id > since_exit (24h, ≤500)
        }

        `since_exit_signal_id` cursors the exit-signal stream independently
        from the open-event cursor (they come from different sequences).
        """
        path = f"/agent/realtime/bots/{self._bot_id}/ambush-events/bootstrap"
        query: dict[str, str] = {}
        if since_event_id and since_event_id > 0:
            query["since"] = str(since_event_id)
        if since_exit_signal_id and since_exit_signal_id > 0:
            query["since_exit"] = str(since_exit_signal_id)
        return self._tc._request("GET", path, query=query)

    def get_events_since(
        self,
        from_ts: str = "",
        since_event_id: int = 0,
        since_exit_signal_id: int = 0,
        limit: int = 100,
    ) -> dict:
        """GET /agent/realtime/bots/:bot_id/ambush-events?from_ts=&since=&since_exit=&limit=.

        Returns the parsed response dict:
          {
            "bot_id":            str,
            "items":             [...]  # detected events
            "exit_signal_items": [...]  # exit signals
          }

        The dict shape (vs. previous list-only return) is a small contract
        widening — the poller is the only caller; it now reads both
        streams.
        """
        path = f"/agent/realtime/bots/{self._bot_id}/ambush-events"
        query: dict[str, str] = {}
        if from_ts:
            query["from_ts"] = from_ts
        if since_event_id and since_event_id > 0:
            query["since"] = str(since_event_id)
        if since_exit_signal_id and since_exit_signal_id > 0:
            query["since_exit"] = str(since_exit_signal_id)
        if limit and limit > 0:
            query["limit"] = str(limit)
        resp = self._tc._request("GET", path, query=query)
        if not isinstance(resp, dict):
            return {"items": [], "exit_signal_items": []}
        # Normalize missing keys so callers don't need to check.
        resp.setdefault("items", [])
        resp.setdefault("exit_signal_items", [])
        return resp

    # ── WebSocket: URL + headers ───────────────────────────────────────────

    def build_ws_url(self) -> str:
        """ws:// or wss:// URL for the ambush events stream. No query."""
        base = self._tc.base_url.rstrip("/")
        path = f"/api/v2/moss/agent/realtime/bots/{self._bot_id}/ambush-events/ws"
        if base.startswith("https://"):
            base = "wss://" + base[len("https://"):]
        elif base.startswith("http://"):
            base = "ws://" + base[len("http://"):]
        return f"{base}{path}"

    def build_ws_headers(self) -> dict:
        """HMAC signature headers for the WS handshake. Same scheme as REST."""
        path = f"/api/v2/moss/agent/realtime/bots/{self._bot_id}/ambush-events/ws"
        # canonical query is empty for the WS upgrade GET
        ts, nonce, sig = self._tc._sign("GET", path, "", "")
        return {
            "X-API-KEY":   self._tc.api_key,
            "X-TS":        ts,
            "X-NONCE":     nonce,
            "X-SIGNATURE": sig,
        }
