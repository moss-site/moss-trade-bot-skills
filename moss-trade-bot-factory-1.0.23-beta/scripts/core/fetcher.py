"""
数据采集模块

- 回测 / 上传验证：使用预置的 Hyperliquid 固定数据集 CSV。
- 实盘信号：使用 Hyperliquid 永续合约拉取实时 K 线数据。
"""

import os
import time
from datetime import datetime, timedelta, timezone
from typing import Optional

import pandas as pd

EXCHANGE_ID = "hyperliquid"
LIVE_ALLOWED_EXCHANGE_IDS = {"hyperliquid"}

try:
    import ccxt
except ImportError:
    ccxt = None


DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data_cache")
SCRIPTS_DIR = os.path.dirname(os.path.dirname(__file__))


def normalize_symbol_for_exchange(symbol: str, exchange_id: str) -> str:
    symbol = symbol.strip().upper().replace("-", "/")
    if exchange_id == "hyperliquid":
        # Hyperliquid 使用 USDC 永续：BTC/USDT → BTC/USDC:USDC
        base = symbol.split("/")[0].split(":")[0]
        return f"{base}/USDC:USDC"
    return symbol


def get_exchange(exchange_id: str = EXCHANGE_ID):
    if ccxt is None:
        raise ImportError("ccxt is required for live data fetching. Install: pip install ccxt")
    if not hasattr(ccxt, exchange_id):
        raise ValueError(f"Unsupported exchange: {exchange_id}")
    return getattr(ccxt, exchange_id)({"enableRateLimit": True})


def _fetch_ohlcv(
    symbol: str,
    timeframe: str,
    days: int,
    exchange_id: str,
    use_cache: bool,
    since_date: str = None,
) -> pd.DataFrame:
    os.makedirs(DATA_DIR, exist_ok=True)
    symbol = normalize_symbol_for_exchange(symbol, exchange_id)
    cache_tag = f"{since_date}_{days}d" if since_date else f"{days}d"
    cache_file = os.path.join(DATA_DIR, f"{exchange_id}_{symbol.replace('/', '_')}_{timeframe}_{cache_tag}.csv")

    # Cache check
    if use_cache and os.path.exists(cache_file):
        mod_time = os.path.getmtime(cache_file)
        if time.time() - mod_time < 86400:
            return pd.read_csv(cache_file, parse_dates=["timestamp"])

    exchange = get_exchange(exchange_id)

    if since_date:
        start_dt = datetime.fromisoformat(since_date).replace(tzinfo=timezone.utc)
        end_dt = start_dt + timedelta(days=days)
        since = exchange.parse8601(start_dt.isoformat())
        end_ms = exchange.parse8601(end_dt.isoformat())
    else:
        since = exchange.parse8601((datetime.now(timezone.utc) - timedelta(days=days)).isoformat())
        end_ms = None

    all_data = []
    limit = 5000
    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    target_end = end_ms if end_ms else now_ms

    while True:
        try:
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=limit)
        except Exception as e:
            print(f"Error fetching {symbol} {timeframe}: {e}")
            break

        if not ohlcv:
            break

        if end_ms:
            ohlcv = [c for c in ohlcv if c[0] < end_ms]

        all_data.extend(ohlcv)

        if not ohlcv:
            break

        last_ts = ohlcv[-1][0]
        if last_ts >= target_end - 1:
            break
        if last_ts <= since:
            break

        since = last_ts + 1
        time.sleep(exchange.rateLimit / 1000)

    if not all_data:
        raise ValueError(f"No data fetched for {symbol} {timeframe}")

    df = pd.DataFrame(all_data, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df = df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)

    df.to_csv(cache_file, index=False)
    return df


def fetch_live_ohlcv(
    symbol: str = "BTC/USDT",
    timeframe: str = "1h",
    days: int = 14,
    data_source: str = "hyperliquid",
    use_cache: bool = True,
) -> pd.DataFrame:
    """下载 live mode 所需 K 线（Hyperliquid 永续合约）。"""
    if data_source not in LIVE_ALLOWED_EXCHANGE_IDS:
        raise ValueError(f"unsupported live data source: {data_source}")
    return _fetch_ohlcv(
        symbol=symbol,
        timeframe=timeframe,
        days=days,
        exchange_id=data_source,
        use_cache=use_cache,
        since_date=None,
    )
