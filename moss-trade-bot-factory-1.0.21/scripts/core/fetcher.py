"""
数据采集模块

回测 / 上传验证：
- 严格限制：仅从 Binance 期货 (binanceusdm) 拉取 K 线数据。

实盘自动运行：
- 默认仍使用 Binance 期货数据。
- 若美区用户无法访问 Binance API，可改用 Coinbase 现货 K 线作为
  signal 输入数据源。
- Coinbase 仅允许用于 live mode，不允许用于回测或上传验证。
"""

import os
import time
from datetime import datetime, timedelta, timezone
from typing import Optional

import pandas as pd

BACKTEST_EXCHANGE_ID = "binanceusdm"
EXCHANGE_ID = BACKTEST_EXCHANGE_ID
LIVE_ALLOWED_EXCHANGE_IDS = {"binanceusdm", "coinbase"}

try:
    import ccxt
except ImportError:
    ccxt = None


DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data_cache")


def normalize_symbol_for_exchange(symbol: str, exchange_id: str) -> str:
    symbol = symbol.strip().upper().replace("-", "/")
    if exchange_id == BACKTEST_EXCHANGE_ID:
        if "/" in symbol and ":USDT" not in symbol:
            symbol = f"{symbol}:USDT"
        return symbol
    if exchange_id == "coinbase":
        if ":USDT" in symbol:
            symbol = symbol.split(":")[0]
        if symbol.endswith("/USDT"):
            base = symbol.split("/")[0]
            return f"{base}/USD"
        return symbol
    return symbol


def get_ohlcv_cache_path(
    symbol: str,
    timeframe: str,
    days: int,
    since_date: str = None,
    exchange_id: str = EXCHANGE_ID,
) -> str:
    """Return the cache file path used by fetch_ohlcv (same logic for consistency)."""
    symbol = normalize_symbol_for_exchange(symbol, exchange_id)
    if since_date:
        cache_tag = f"{since_date}_{days}d"
    else:
        cache_tag = f"{days}d"
    return os.path.join(DATA_DIR, f"{exchange_id}_{symbol.replace('/', '_')}_{timeframe}_{cache_tag}.csv")


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
    cache_file = get_ohlcv_cache_path(
        symbol=symbol,
        timeframe=timeframe,
        days=days,
        since_date=since_date,
        exchange_id=exchange_id,
    )

    if use_cache and os.path.exists(cache_file):
        mod_time = os.path.getmtime(cache_file)
        if time.time() - mod_time < 86400:
            print(f"Using cached file: {os.path.basename(cache_file)}", file=__import__('sys').stderr)
            return pd.read_csv(cache_file, parse_dates=["timestamp"])

    if not since_date:
        if use_cache:
            import glob
            prefix = f"{exchange_id}_{symbol.replace('/', '_')}_{timeframe}_"
            pattern = os.path.join(DATA_DIR, f"{prefix}{days}d.csv")
            matches = sorted(glob.glob(pattern), key=os.path.getmtime, reverse=True)
            if matches:
                best = matches[0]
                print(f"Using cached file: {os.path.basename(best)}", file=__import__('sys').stderr)
                return pd.read_csv(best, parse_dates=["timestamp"])

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
    limit = 1000
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

    # 缓存
    df.to_csv(cache_file, index=False)
    print(f"Fetched {len(df)} candles for {symbol} {timeframe}, cached to {cache_file}", file=__import__('sys').stderr)

    return df


def fetch_ohlcv(
    symbol: str = "BTC/USDT",
    timeframe: str = "1h",
    days: int = 120,
    exchange_id: str = None,  # 兼容旧签名，仍只允许 binanceusdm
    use_cache: bool = True,
    since_date: str = None,
) -> pd.DataFrame:
    """
    下载 Binance 期货 K线数据。

    这是回测 / 上传验证专用入口，固定使用 Binance USDT-M 期货数据。
    """
    if exchange_id not in (None, BACKTEST_EXCHANGE_ID):
        raise ValueError("fetch_ohlcv only supports Binance USDT-M for backtest/verify")
    return _fetch_ohlcv(
        symbol=symbol,
        timeframe=timeframe,
        days=days,
        exchange_id=BACKTEST_EXCHANGE_ID,
        use_cache=use_cache,
        since_date=since_date,
    )


def fetch_live_ohlcv(
    symbol: str = "BTC/USDT",
    timeframe: str = "1h",
    days: int = 14,
    data_source: str = BACKTEST_EXCHANGE_ID,
    use_cache: bool = True,
) -> pd.DataFrame:
    """
    下载 live mode 所需 K 线。

    - `binanceusdm`: Binance 全球站 USDT-M 期货
    - `coinbase`: Coinbase 现货 K 线，仅用于实盘信号输入
    """
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


def fetch_multi_symbol(
    symbols: list[str] = None,
    timeframe: str = "1h",
    days: int = 120,
) -> dict[str, pd.DataFrame]:
    """
    批量下载多个交易对数据（仅 Binance 期货）。
    """
    if symbols is None:
        symbols = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT"]

    result = {}
    for symbol in symbols:
        try:
            result[symbol] = fetch_ohlcv(symbol, timeframe, days)
        except Exception as e:
            print(f"Failed to fetch {symbol}: {e}")

    return result


def summarize_dataframe(df: pd.DataFrame, symbol: str = "") -> None:
    """打印DataFrame摘要信息。"""
    prefix = f"[{symbol}] " if symbol else ""
    print(f"{prefix}{len(df)}根K线, "
          f"{df['timestamp'].iloc[0]} ~ {df['timestamp'].iloc[-1]}, "
          f"价格 {df['close'].min():.0f} ~ {df['close'].max():.0f}")


def fetch_multi_timeframe(
    symbol: str = "BTC/USDT",
    timeframes: list[str] = None,
    days: int = 120,
) -> dict[str, pd.DataFrame]:
    """
    下载同一交易对的多个时间周期数据（仅 Binance 期货）。
    """
    if timeframes is None:
        timeframes = ["5m", "15m", "1h", "4h", "1d"]

    result = {}
    for tf in timeframes:
        try:
            result[tf] = fetch_ohlcv(symbol, tf, days)
        except Exception as e:
            print(f"Failed to fetch {symbol} {tf}: {e}")

    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="下载交易所K线数据")
    parser.add_argument("--symbols", nargs="+", default=["BTC/USDT"],
                        help="交易对列表 (默认: BTC/USDT)")
    parser.add_argument("--timeframe", default="1h",
                        help="K线周期 (默认: 1h)")
    parser.add_argument("--days", type=int, default=148,
                        help="下载天数 (默认: 148，约2025-10-01至今)")
    parser.add_argument("--no-cache", action="store_true",
                        help="忽略缓存强制重新下载")
    args = parser.parse_args()

    for symbol in args.symbols:
        df = fetch_ohlcv(symbol, args.timeframe, args.days,
                         use_cache=not args.no_cache)
        summarize_dataframe(df, symbol)
