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
import urllib.request
import urllib.parse

from text_i18n import default_text, validate_bilingual_text
API_PREFIX = "/api/v1/moss/agent"
DEFAULT_HEADERS = {
    "Content-Type": "application/json",
    "Accept": "application/json",
    "User-Agent": "curl/8.7.1",
    "X-Client-Source": "skill",
}

# 多 realtime bot 时以下接口需带 X-BOT-ID；单 bot 可省略
ACCOUNT_BOUND_PATHS = (
    "/account", "/positions", "/orders", "/trades", "/profile",
    "/orders/market", "/positions/close",
)


class TradingClient:
    def __init__(self, api_key: str = "", api_secret: str = "",
                 base_url: str = "", bot_id: str = ""):
        self.api_key = api_key
        self.api_secret = api_secret
        self.bot_id = bot_id
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
                "'https://ai.moss.site'. The client appends the API prefix "
                "automatically and will request "
                "'https://ai.moss.site/api/v1/moss/agent/agents/bind' "
                "for bind."
            )

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
            if self.bot_id and path in ACCOUNT_BOUND_PATHS:
                headers["X-BOT-ID"] = self.bot_id

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

    # ── Binding ──

    def create_pair_code(self, user_uuid: str) -> dict:
        return self._request("POST", "/pair-codes", query={"user_uuid": user_uuid}, need_auth=False)

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
        return self._request("POST", "/agents/bind", body, need_auth=False)

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
            body["strategy"]["symbol"] = symbol
        if timeframe:
            body["strategy"]["timeframe"] = timeframe
        if exchange:
            body["strategy"]["exchange"] = exchange
        if lookback_bars:
            body["strategy"]["lookback_bars"] = lookback_bars
        if schedule_interval_minutes:
            body["strategy"]["schedule_interval_minutes"] = schedule_interval_minutes
        return self._request("POST", "/realtime/bots", body)

    def unbind(self, bot_id: str, user_uuid: str) -> dict:
        """Remove one realtime bot (does not revoke binding). Path id = realtime bot id."""
        return self._request("POST", f"/agents/{bot_id}/unbind",
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
        return self._request("PATCH", "/profile", body)

    # ── Trading (HMAC) ──

    def get_price(self) -> dict:
        return self._request("GET", "/market/price")

    def get_account(self) -> dict:
        return self._request("GET", "/account")

    def get_positions(self) -> list:
        return self._request("GET", "/positions")

    def get_orders(self, limit: int = 100) -> list:
        return self._request("GET", "/orders", query={"limit": str(limit)})

    def get_trades(self, limit: int = 100) -> list:
        return self._request("GET", "/trades", query={"limit": str(limit)})

    def open_long(self, notional_usdt: str, leverage: int, client_order_id: str = "") -> dict:
        body = {
            "symbol": "BTCUSDT",
            "side": "BUY",
            "position_side": "LONG",
            "notional_usdt": notional_usdt,
            "leverage": leverage,
        }
        if client_order_id:
            body["client_order_id"] = client_order_id
        return self._request("POST", "/orders/market", body)

    def open_short(self, notional_usdt: str, leverage: int, client_order_id: str = "") -> dict:
        body = {
            "symbol": "BTCUSDT",
            "side": "SELL",
            "position_side": "SHORT",
            "notional_usdt": notional_usdt,
            "leverage": leverage,
        }
        if client_order_id:
            body["client_order_id"] = client_order_id
        return self._request("POST", "/orders/market", body)

    def close_position(self, position_side: str, close_qty_btc: str = "") -> dict:
        body = {
            "symbol": "BTCUSDT",
            "position_side": position_side,
        }
        if close_qty_btc:
            body["close_qty_btc"] = close_qty_btc
        return self._request("POST", "/positions/close", body)

    # ── Public display (no auth) ──

    def get_discover_leaderboard(self, mode: str = "realtime") -> dict:
        return self._request("GET", "/trader/discover/leaderboard",
                             query={"mode": mode}, need_auth=False)

    def get_bots_public(self, mode: str = "hell", sort_by: str = "pnl",
                        sort_order: str = "desc", page: int = 1, page_size: int = 20) -> dict:
        return self._request("GET", "/trader/bots", query={
            "mode": mode, "sort_by": sort_by, "sort_order": sort_order,
            "page": str(page), "page_size": str(page_size),
        }, need_auth=False)

    def get_bot_detail_public(self, bot_id: str) -> dict:
        return self._request("GET", f"/trader/bots/{bot_id}", need_auth=False)

    # ── User-scoped display (no auth, needs user_uuid) ──

    def get_overview(self, user_uuid: str) -> dict:
        return self._request("GET", "/trader/overview",
                             query={"user_uuid": user_uuid}, need_auth=False)

    # ── Backtest verify (HMAC，配对绑定后即用 api_key + api_secret) ──

    def verify_backtest(self, package: dict) -> dict:
        """Submit verify job (async). Requires HMAC auth (api_key + api_secret from bind)."""
        if not self.api_key or not self.api_secret:
            return {"code": "MISSING_CREDS", "message": "api_key/api_secret required. Run bind with pair_code first, save to agent_creds.json."}
        return self._request("POST", "/backtest/verify", body=package, need_auth=True)

    def get_verify_job(self, job_id: str) -> dict:
        """Poll verify job status. Requires HMAC auth."""
        if not self.api_key or not self.api_secret:
            return {"code": "MISSING_CREDS", "message": "api_key/api_secret required."}
        return self._request("GET", f"/backtest/jobs/{job_id}", need_auth=True)

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
        }, need_auth=False)

    def get_backtest_bot_detail(self, user_uuid: str, bot_id: str) -> dict:
        return self._request("GET", f"/backtest/bots/{bot_id}",
                             query={"user_uuid": user_uuid}, need_auth=False)

    def delete_backtest_bot(self, user_uuid: str, bot_id: str) -> dict:
        return self._request("DELETE", f"/backtest/bots/{bot_id}",
                             query={"user_uuid": user_uuid}, need_auth=False)

    def get_backtest_leaderboard(self, sort_by: str = "return", page: int = 1, page_size: int = 20) -> dict:
        return self._request("GET", "/backtest/leaderboard", query={
            "sort_by": sort_by, "page": str(page), "page_size": str(page_size),
        }, need_auth=False)
