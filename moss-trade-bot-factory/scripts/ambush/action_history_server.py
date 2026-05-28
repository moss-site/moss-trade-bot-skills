"""Optional local read-only HTTP surface for Ambush symbol action history."""

from __future__ import annotations

import json
import logging
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

from ambush import live_database as db


logger = logging.getLogger("ambush.action_history")


def _first(params: dict[str, list[str]], key: str, default: str = "") -> str:
    values = params.get(key)
    if not values:
        return default
    return str(values[0] or "").strip()


def _parse_limit(raw: str) -> int:
    try:
        return max(1, min(int(raw), 500))
    except (TypeError, ValueError):
        return 100


def _write_json(handler: BaseHTTPRequestHandler, status: HTTPStatus, body: dict) -> None:
    payload = json.dumps(body, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    handler.send_response(int(status))
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(payload)))
    handler.end_headers()
    handler.wfile.write(payload)


def _make_handler(db_path: str) -> type[BaseHTTPRequestHandler]:
    class ActionHistoryHandler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            if parsed.path == "/healthz":
                _write_json(self, HTTPStatus.OK, {"ok": True})
                return
            if parsed.path != "/symbol-action-history":
                _write_json(self, HTTPStatus.NOT_FOUND, {"error": "not_found"})
                return

            params = parse_qs(parsed.query)
            symbol = _first(params, "symbol")
            action = _first(params, "action")
            limit = _parse_limit(_first(params, "limit", "100"))
            if not symbol:
                symbol = _first(params, "hl_symbol")
            if action == "":
                action = None

            try:
                items = db.list_symbol_action_history(
                    db_path,
                    hl_symbol=symbol or None,
                    action=action,
                    limit=limit,
                )
            except ValueError as e:
                _write_json(self, HTTPStatus.BAD_REQUEST, {"error": str(e)})
                return
            _write_json(
                self,
                HTTPStatus.OK,
                {
                    "items": items,
                    "count": len(items),
                    "symbol": symbol.upper() if symbol else None,
                    "action": action,
                    "limit": limit,
                },
            )

        def log_message(self, fmt: str, *args) -> None:
            logger.debug("action-history http: " + fmt, *args)

    return ActionHistoryHandler


def start(db_path: str, *, host: str, port: int) -> ThreadingHTTPServer:
    server = ThreadingHTTPServer((host, int(port)), _make_handler(db_path))
    thread = threading.Thread(
        target=server.serve_forever,
        name="ambush-action-history-http",
        daemon=True,
    )
    thread.start()
    actual_host, actual_port = server.server_address[:2]
    logger.info(
        "ambush action-history REST listening on http://%s:%s/symbol-action-history",
        actual_host,
        actual_port,
    )
    return server
