from __future__ import annotations

ASSET_MAX_LEVERAGES = {
    "BTC": 40,
    "ETH": 25,
    "SOL": 20,
    "XRP": 20,
    "ADA": 10,
    "APT": 10,
    "ARB": 10,
    "AVAX": 10,
    "BCH": 10,
    "BNB": 10,
    "DOGE": 10,
    "DOT": 10,
    "LINK": 10,
    "LTC": 10,
    "NEAR": 10,
    "SUI": 10,
    "TRX": 10,
    "UNI": 10,
    "ATOM": 5,
    "FIL": 5,
    "HBAR": 5,
    "OP": 5,
}


def normalize_perp_base_asset(symbol: str | None) -> str:
    raw = (symbol or "").upper().strip()
    raw = raw.replace("/", "").replace("-", "").replace("_", "").replace(":", "")
    for suffix in ("USDC", "USDT", "BUSD"):
        if raw.endswith(suffix):
            raw = raw[: -len(suffix)]
            break
    return raw


def max_leverage_for_symbol(symbol: str | None) -> int:
    return ASSET_MAX_LEVERAGES.get(normalize_perp_base_asset(symbol), 0)


def cap_params_for_symbol(params: dict | None, symbol: str | None) -> dict:
    out = dict(params or {})
    cap = max_leverage_for_symbol(symbol)
    if cap <= 0:
        return out
    base = float(out.get("base_leverage", 10.0))
    max_lev = float(out.get("max_leverage", 40.0))
    base = max(1.0, min(base, float(cap)))
    max_lev = max(base, min(max_lev, float(cap)))
    out["base_leverage"] = base
    out["max_leverage"] = max_lev
    return out
