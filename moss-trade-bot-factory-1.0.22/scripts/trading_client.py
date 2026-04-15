"""Trading client for the simulation platform API with HMAC authentication.

Platform base URL must be provided explicitly by the caller or loaded from the
local agent_creds.json file. This module intentionally does not depend on
hidden environment variables.
"""

import hashlib
import hmac
import json
import secrets
import time
import urllib.parse
import urllib.request

from text_i18n import default_text, validate_bilingual_text

API_PREFIX = "/api/v2/moss"
API_PREFIX_V1_AGENT = "/api/v1/moss/agent"
DEFAULT_HEADERS = {
    "Content-Type": "application/json",
    "Accept": "application/json",
    "User-Agent": "curl/8.7.1",
    "X-Client-Source": "skill",
}


class TradingClient:
    def __init__(self, api_key: str = "", api_secret: str = "",
                 base_url: str = "", bot_id: str = "", symbol: str = "BTCUSDT"):
        self.api_key = api_key
        self.api_secret = api_secret
        self.bot_id = bot_id
        self.symbol = self._normalize_symbol(symbol)
        self.base_url = str(base_url or "").rstrip("/")
        if not self.base_url:
            raise ValueError(
                "Platform base URL missing. Pass --platform-url explicitly or store "
                "base_url in agent_creds.json before using trading/verify features."
            )
        parsed = urllib.parse.urlsplit(self.base_url)
        if parsed.path not in ("", "/"):
            raise ValueError(
                "Platform base URL must be the site origin only, for example "
                "'https://beta-api.moss.site'. The client appends the API prefix "
                "automatically and will request "
                "'https://beta-api.moss.site/api/v1/moss/agent/agents/bind' "
                "for bind."
            )

    @staticmethod
    def _normalize_symbol(symbol: str) -> str:
        """Normalize trading symbol: 'ETH/USDT' -> 'ETHUSDT', 'BTCUSDT' -> 'BTCUSDT'."""
        return symbol.upper().replace("/", "").replace(":", "").replace("-", "")

    def _sign(self, method: str, path: str, query: str, body: str) -> tuple[str, str, str]:
        ts = str(int(time.time()))
        nonce = secrets.token_hex(12)
        payload = f"{method}\n{path}\n{query}\n{body}\n{ts}\n{nonce}"
        signature = hmac.new(
            self.api_secret.encode(), payload.encode(), hashlib.sha256
        ).hexdigest()
        return ts, nonce, signature

    def _request(self, method: str, path: str, body: dict = None,
                 query: dict = None, need_auth: bool = True,
                 custom_prefix: str = None) -> dict:
        prefix = custom_prefix if custom_prefix is not None else API_PREFIX
        if not path.startswith("/"):
            path = "/" + path
        full_path = f"{prefix}{path}"
        url = f"{self.base_url}{full_path}"

        canonical_query = ""
        if query:
            sorted_params = sorted(query.items())
            canonical_query = urllib.parse.urlencode(sorted_params)
            url = f"{url}?{canonical_query}"

        raw_body = ""
        if body is not None:
            raw_body = json.dumps(body, separators=(",", ":"))

        headers = dict(DEFAULT_HEADERS)

        if need_auth and self.api_key:
            ts, nonce, sig = self._sign(method, full_path, canonical_query, raw_body)
            headers["X-API-KEY"] = self.api_key
            headers["X-TS"] = ts
            headers["X-NONCE"] = nonce
            headers["X-SIGNATURE"] = sig

        req = urllib.request.Request(
            url,
            data=raw_body.encode() if raw_body else None,
            headers=headers,
            method=method,
        )

        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                resp_body = resp.read()
                if not resp_body:
                    return {"status": "ok"}
                return json.loads(resp_body)
        except urllib.error.HTTPError as e:
            error_body = e.read().decode()
            try:
                return json.loads(error_body)
            except json.JSONDecodeError:
                return {"code": "HTTP_ERROR", "message": f"{e.code}: {error_body}"}

    def _require_bot_id(self) -> str:
        bot_id = str(self.bot_id or "").strip()
        if not bot_id:
            raise ValueError("bot_id missing. Create realtime bot first and save bot_id into creds.")
        return bot_id

    @staticmethod
    def _extract_items(payload) -> list:
        if isinstance(payload, list):
            return payload
        if isinstance(payload, dict):
            # Preserve API errors instead of turning them into empty lists.
            if payload.get("code"):
                return payload
            items = payload.get("items")
            if isinstance(items, list):
                return items
            return payload
        return payload

    @staticmethod
    def _adapt_order_result(payload):
        if not isinstance(payload, dict):
            return payload
        order = payload.get("order")
        fills = payload.get("fills")
        if not isinstance(order, dict):
            return payload

        fill0 = {}
        if isinstance(fills, list) and fills and isinstance(fills[0], dict):
            fill0 = fills[0]

        adapted = dict(payload)
        if order.get("order_id") is not None:
            adapted["order_id"] = str(order.get("order_id"))
        if fill0.get("price") is not None:
            adapted["fill_price"] = str(fill0.get("price"))
        if fill0.get("qty") is not None:
            adapted["fill_qty"] = str(fill0.get("qty"))
        if fill0.get("realized_pnl") is not None:
            adapted["realized_pnl"] = str(fill0.get("realized_pnl"))
        return adapted

    # ── Binding ──

    def create_pair_code(self, user_uuid: str) -> dict:
        # Pair-code/bind currently use v1 agent routes.
        return self._request("POST", "/pair-codes", query={"user_uuid": user_uuid},
                             need_auth=False, custom_prefix=API_PREFIX_V1_AGENT)

    def bind(self, pair_code: str, display_name: str = "Bot",
             persona: str = "", description: str = "",
             fingerprint: str = "") -> dict:
        if not fingerprint:
            fingerprint = f"sha256:{secrets.token_hex(16)}"
        body = {
            "pair_code": pair_code,
            "display_name": display_name,
            "persona": persona or display_name,
            "description": description or f"{display_name} trading bot",
            "agent_fingerprint": fingerprint,
        }
        return self._request("POST", "/agents/bind", body, need_auth=False,
                             custom_prefix=API_PREFIX_V1_AGENT)

    def create_realtime_bot(
        self,
        display_name: str,
        persona: str,
        description: str,
        strategy_params: dict,
        display_name_i18n: dict = None,
        persona_i18n: dict = None,
        description_i18n: dict = None,
        *,
        symbol: str = "",
        timeframe: str = "",
        exchange: str = "",
        lookback_bars: int = 0,
        schedule_interval_minutes: int = 0,
    ) -> dict:
        """Create a realtime bot under current binding. Requires prior bind (api_key/api_secret)."""
        display_name_i18n = validate_bilingual_text("display_name_i18n", display_name_i18n or {}, 64)
        persona_i18n = validate_bilingual_text("persona_i18n", persona_i18n or {}, 64)
        description_i18n = validate_bilingual_text("description_i18n", description_i18n or {}, 280)
        body = {
            "display_name": display_name or default_text(display_name_i18n),
            "display_name_i18n": display_name_i18n,
            "persona": persona or default_text(persona_i18n),
            "persona_i18n": persona_i18n,
            "description": description or default_text(description_i18n),
            "description_i18n": description_i18n,
            "strategy": {"params": strategy_params},
        }
        if symbol:
            body["strategy"]["symbol"] = self._normalize_symbol(symbol)
        if timeframe:
            body["strategy"]["timeframe"] = timeframe
        if exchange:
            body["strategy"]["exchange"] = exchange
        if lookback_bars:
            body["strategy"]["lookback_bars"] = lookback_bars
        if schedule_interval_minutes:
            body["strategy"]["schedule_interval_minutes"] = schedule_interval_minutes
        return self._request("POST", "/agent/realtime/bots", body)

    def unbind(self, bot_id: str, user_uuid: str) -> dict:
        """Remove one realtime bot (does not revoke binding). Path id = realtime bot id."""
        return self._request("DELETE", f"/trader/realtime/bots/{bot_id}",
                             query={"user_uuid": user_uuid}, need_auth=False)

    # ── Profile (HMAC) ──

    def update_profile(
        self,
        display_name: str = "",
        persona: str = "",
        description: str = "",
        display_name_i18n: dict = None,
        persona_i18n: dict = None,
        description_i18n: dict = None,
    ) -> dict:
        body = {}
        if display_name:
            body["display_name"] = display_name
        if persona:
            body["persona"] = persona
        if description:
            body["description"] = description
        if display_name_i18n:
            body["display_name_i18n"] = display_name_i18n
        if persona_i18n:
            body["persona_i18n"] = persona_i18n
        if description_i18n:
            body["description_i18n"] = description_i18n
        bot_id = self._require_bot_id()
        return self._request("PATCH", f"/agent/realtime/bots/{bot_id}/profile", body)

    # ── Trading (HMAC) ──

    def get_price(self, symbol: str = None) -> dict:
        normalized_symbol = self._normalize_symbol(symbol) if symbol else self.symbol
        return self._request("GET", f"/agent/realtime/market/{normalized_symbol}")

    def get_account(self) -> dict:
        bot_id = self._require_bot_id()
        return self._request("GET", f"/agent/realtime/bots/{bot_id}/account")

    def get_positions(self) -> list:
        bot_id = self._require_bot_id()
        payload = self._request("GET", f"/agent/realtime/bots/{bot_id}/positions")
        return self._extract_items(payload)

    def get_orders(self, limit: int = 100) -> list:
        bot_id = self._require_bot_id()
        page_size = max(1, min(int(limit), 200))
        payload = self._request("GET", f"/agent/realtime/bots/{bot_id}/orders",
                                query={"page": "1", "page_size": str(page_size)})
        return self._extract_items(payload)

    def get_trades(self, limit: int = 100) -> list:
        bot_id = self._require_bot_id()
        page_size = max(1, min(int(limit), 200))
        payload = self._request("GET", f"/agent/realtime/bots/{bot_id}/fills",
                                query={"page": "1", "page_size": str(page_size)})
        return self._extract_items(payload)

    def open_long(self, notional_usdt: str, leverage: int, client_order_id: str = "") -> dict:
        return self._open_market_order("buy", notional_usdt, leverage, client_order_id)

    def open_short(self, notional_usdt: str, leverage: int, client_order_id: str = "") -> dict:
        return self._open_market_order("sell", notional_usdt, leverage, client_order_id)

    def _open_market_order(self, side: str, notional_usdt: str, leverage: int, client_order_id: str = "") -> dict:
        bot_id = self._require_bot_id()
        body = {
            "symbol": self.symbol,
            "side": side,
            "order_type": "market",
            "time_in_force": "ioc",
            "post_only": False,
            "reduce_only": False,
            "notional": notional_usdt,
            "leverage": leverage,
        }
        if client_order_id:
            body["client_order_id"] = client_order_id
        payload = self._request("POST", f"/agent/realtime/bots/{bot_id}/orders", body)
        return self._adapt_order_result(payload)

    def close_position(self, position_side: str = "", close_qty: str = "") -> dict:
        """Close position. position_side is accepted for v1 compat but unused in v2."""
        _ = position_side  # v2 resolves side from bot_id + symbol in path.
        bot_id = self._require_bot_id()
        body = {}
        if close_qty:
            body["qty"] = close_qty
        payload = self._request(
            "POST",
            f"/agent/realtime/bots/{bot_id}/positions/{self.symbol}/close",
            body,
        )
        return self._adapt_order_result(payload)

    # ── Public display (no auth) ──

    def get_discover_leaderboard(self, mode: str = "realtime") -> dict:
        del mode
        return self._request("GET", "/trader/realtime/leaderboard", need_auth=False)

    def get_bots_public(self, mode: str = "realtime", sort_by: str = "pnl",
                        sort_order: str = "desc", page: int = 1, page_size: int = 20) -> dict:
        del mode
        return self._request("GET", "/trader/realtime/bots", query={
            "sort_by": sort_by, "sort_order": sort_order,
            "page": str(page), "page_size": str(page_size),
        }, need_auth=False)

    def get_bot_detail_public(self, bot_id: str) -> dict:
        return self._request("GET", f"/trader/realtime/bots/{bot_id}", need_auth=False)

    # ── User-scoped display (no auth, needs user_uuid) ──

    def get_overview(self, user_uuid: str) -> dict:
        return self._request("GET", "/trader/realtime/overview",
                             query={"user_uuid": user_uuid}, need_auth=False)

    # ── Backtest verify (HMAC，配对绑定后即用 api_key + api_secret) ──

    def verify_backtest(self, package: dict) -> dict:
        """Submit verify job (async). Requires HMAC auth (api_key + api_secret from bind)."""
        if not self.api_key or not self.api_secret:
            return {"code": "MISSING_CREDS", "message": "api_key/api_secret required. Run bind with pair_code first, save to agent_creds.json."}
        return self._request("POST", "/backtest/verify", body=package, need_auth=True,
                             custom_prefix=API_PREFIX_V1_AGENT)

    def get_verify_job(self, job_id: str) -> dict:
        """Poll verify job status. Requires HMAC auth."""
        if not self.api_key or not self.api_secret:
            return {"code": "MISSING_CREDS", "message": "api_key/api_secret required."}
        return self._request("GET", f"/backtest/jobs/{job_id}", need_auth=True,
                             custom_prefix=API_PREFIX_V1_AGENT)

    def verify_backtest_and_wait(self, package: dict,
                                  poll_interval: int = 3, max_wait: int = 120) -> dict:
        """Submit + poll until terminal state. HMAC auth is sufficient."""
        import time as _time
        job = self.verify_backtest(package)
        job_id = job.get("job_id", "")
        if not job_id:
            return job

        elapsed = 0
        while elapsed < max_wait:
            _time.sleep(poll_interval)
            elapsed += poll_interval
            status = self.get_verify_job(job_id)
            if status.get("code"):
                return status
            st = status.get("status", "")
            if st in ("verified", "rejected", "failed"):
                return status.get("result", status)
        return {"code": "TIMEOUT", "message": f"Job {job_id} not done after {max_wait}s"}

    def get_backtest_bots(self, user_uuid: str, page: int = 1, page_size: int = 20) -> dict:
        return self._request("GET", "/backtest/bots", query={
            "user_uuid": user_uuid, "page": str(page), "page_size": str(page_size),
        }, need_auth=False, custom_prefix=API_PREFIX_V1_AGENT)

    def get_backtest_bot_detail(self, user_uuid: str, bot_id: str) -> dict:
        return self._request("GET", f"/backtest/bots/{bot_id}",
                             query={"user_uuid": user_uuid}, need_auth=False,
                             custom_prefix=API_PREFIX_V1_AGENT)

    def delete_backtest_bot(self, user_uuid: str, bot_id: str) -> dict:
        return self._request("DELETE", f"/backtest/bots/{bot_id}",
                             query={"user_uuid": user_uuid}, need_auth=False,
                             custom_prefix=API_PREFIX_V1_AGENT)

    def get_backtest_leaderboard(self, sort_by: str = "return", page: int = 1, page_size: int = 20) -> dict:
        return self._request("GET", "/backtest/leaderboard", query={
            "sort_by": sort_by, "page": str(page), "page_size": str(page_size),
        }, need_auth=False, custom_prefix=API_PREFIX_V1_AGENT)
