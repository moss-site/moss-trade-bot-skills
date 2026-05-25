"""Parity tests for core.replay_baseline against backend Go realtime replay."""

from __future__ import annotations

import glob
import os
import sys
import unittest
from decimal import Decimal

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPTS = os.path.dirname(HERE)
if SCRIPTS not in sys.path:
    sys.path.insert(0, SCRIPTS)

from core.data_cache_archive import ensure_data_cache
from core.leverage_caps import cap_params_for_symbol, max_leverage_for_symbol
from core.backtest import _aggregate_trades_backend_style, _apply_backend_style_trade_metrics
from core.engine import BacktestResult
from core.replay_baseline import (
    FIXED_REPLAY_FUNDING_RATE,
    FIXED_REPLAY_LOT_SIZE,
    FIXED_REPLAY_TAKER_FEE_RATE,
    REPLAY_BASELINE_PROFILE,
    build_fixed_replay_depth_book,
    build_fixed_replay_funding_events,
    floor_qty_to_lot_size,
    infer_replay_symbol_from_path,
    normalize_replay_coin,
    replay_baseline_assumptions_text,
    simulate_replay_baseline_fill,
    simulate_replay_baseline_fills_per_level,
    synthesize_replay_minute_candles,
)
from core.replay_depth_templates import FIXED_REPLAY_DEPTH_BOOKS


EXPECTED_COINS = [
    # original 22 (HyperCore main board, USDC)
    "ADA", "APT", "ARB", "ATOM", "AVAX", "BCH", "BNB", "BTC", "DOGE", "DOT", "ETH",
    "FIL", "HBAR", "LINK", "LTC", "NEAR", "OP", "SOL", "SUI", "TRX", "UNI", "XRP",
    # added 2026-05-18 — HYPE (main board) + 20 xyz HIP-3 builder assets
    # xyz live trading is gated on backend HIP-3 client integration (follow-up PR);
    # depth + csv staged here so backtest/verify accept these coins
    "HYPE",
    "AMD", "BRENTOIL", "CBRS", "CL", "COIN", "CRCL", "GOLD", "GOOGL", "INTC", "META",
    "MSTR", "MU", "NVDA", "ORCL", "SILVER", "SKHX", "SNDK", "SP500", "TSLA", "XYZ100",
]
EXPECTED_COINS = sorted(EXPECTED_COINS)


def _D(v) -> Decimal:
    return Decimal(str(v))


class ProfileAndConstantsTest(unittest.TestCase):
    def test_profile_name_is_replay_baseline_v1(self):
        self.assertEqual(REPLAY_BASELINE_PROFILE, "replay_baseline_v1")

    def test_taker_fee_rate(self):
        self.assertEqual(FIXED_REPLAY_TAKER_FEE_RATE, 0.00045)

    def test_funding_rate_per_settlement(self):
        self.assertEqual(FIXED_REPLAY_FUNDING_RATE, 0.0000125)

    def test_lot_size_is_zero_in_replay(self):
        self.assertEqual(FIXED_REPLAY_LOT_SIZE, 0.0)

    def test_assumptions_text_mentions_symbol_aware_depth(self):
        text = replay_baseline_assumptions_text()
        self.assertIn("4.5bps", text)
        self.assertIn("0.00125", text)
        self.assertIn("按币种冻结", text)


class ReplayDepthTemplatesTest(unittest.TestCase):
    def test_template_coin_set_matches_backend_depth_universe(self):
        self.assertEqual(sorted(FIXED_REPLAY_DEPTH_BOOKS), EXPECTED_COINS)

    def test_every_template_has_20_ordered_levels_per_side(self):
        for coin, book in FIXED_REPLAY_DEPTH_BOOKS.items():
            with self.subTest(coin=coin):
                self.assertEqual(len(book["bids"]), 20)
                self.assertEqual(len(book["asks"]), 20)
                bid_prices = [_D(price) for price, _ in book["bids"]]
                ask_prices = [_D(price) for price, _ in book["asks"]]
                self.assertEqual(bid_prices, sorted(bid_prices, reverse=True))
                self.assertEqual(ask_prices, sorted(ask_prices))

    def test_top_of_book_pinned_for_sui_and_btc(self):
        self.assertEqual(FIXED_REPLAY_DEPTH_BOOKS["SUI"]["bids"][0], ["1.2324", "2732.3"])
        self.assertEqual(FIXED_REPLAY_DEPTH_BOOKS["SUI"]["asks"][0], ["1.2325", "2546.7"])
        self.assertEqual(FIXED_REPLAY_DEPTH_BOOKS["BTC"]["bids"][0], ["80876.0", "12.19958"])
        self.assertEqual(FIXED_REPLAY_DEPTH_BOOKS["BTC"]["asks"][0], ["80877.0", "5.49057"])

    def test_normalize_replay_coin_accepts_common_symbol_shapes(self):
        for raw in ("SUI", "SUIUSDC", "SUIUSDT", "SUI/USDC", "SUI-USDC", "sui_usdc"):
            self.assertEqual(normalize_replay_coin(raw), "SUI")

    def test_every_packaged_csv_has_a_matching_depth_template(self):
        data_cache = ensure_data_cache()
        paths = sorted(glob.glob(os.path.join(str(data_cache), "hyperliquid_*_15m_*.csv")))
        coins = [normalize_replay_coin(infer_replay_symbol_from_path(path)) for path in paths]
        self.assertEqual(sorted(coins), EXPECTED_COINS)
        for coin in coins:
            self.assertIn(coin, FIXED_REPLAY_DEPTH_BOOKS)


class LeverageCapsTest(unittest.TestCase):
    def test_known_asset_caps_match_platform_caps(self):
        self.assertEqual(max_leverage_for_symbol("BTCUSDC"), 40)
        self.assertEqual(max_leverage_for_symbol("ETHUSDC"), 25)
        self.assertEqual(max_leverage_for_symbol("SOLUSDC"), 20)
        self.assertEqual(max_leverage_for_symbol("SUIUSDC"), 10)
        self.assertEqual(max_leverage_for_symbol("OPUSDC"), 5)

    def test_cap_params_for_symbol_preserves_btc_default_max(self):
        params = cap_params_for_symbol({"base_leverage": 10.0}, "BTCUSDC")
        self.assertEqual(params["base_leverage"], 10.0)
        self.assertEqual(params["max_leverage"], 40.0)

    def test_cap_params_for_symbol_caps_alt_defaults_and_explicit_values(self):
        sui = cap_params_for_symbol({"base_leverage": 10.0, "max_leverage": 40.0}, "SUIUSDC")
        self.assertEqual(sui["base_leverage"], 10.0)
        self.assertEqual(sui["max_leverage"], 10.0)

        op = cap_params_for_symbol({"base_leverage": 10.0, "max_leverage": 40.0}, "OPUSDC")
        self.assertEqual(op["base_leverage"], 5.0)
        self.assertEqual(op["max_leverage"], 5.0)


class BackendStyleTradeMetricTest(unittest.TestCase):
    def test_fill_aggregation_counts_closed_lifecycles_not_partial_reductions(self):
        fills = [
            {"side": "buy", "qty": 10.0, "price": 1.0},
            {"side": "sell", "qty": 4.0, "price": 1.1},
            {"side": "sell", "qty": 6.0, "price": 0.9},
            {"side": "sell", "qty": 5.0, "price": 0.8},
            {"side": "buy", "qty": 6.0, "price": 0.7},
        ]

        agg = _aggregate_trades_backend_style(fills)

        self.assertEqual(agg["total_trades"], 2)
        self.assertEqual(agg["wins"], 1)
        self.assertAlmostEqual(agg["win_rate"], 0.5, places=12)
        self.assertAlmostEqual(agg["profit_factor"], 2.5, places=12)

    def test_result_metrics_are_overridden_with_backend_fill_aggregation(self):
        result = BacktestResult(total_trades=5, win_rate=1.0, profit_factor=99.0)
        fills = [
            {"side": "buy", "qty": 1.0, "price": 100.0},
            {"side": "sell", "qty": 1.0, "price": 90.0},
        ]

        _apply_backend_style_trade_metrics(result, fills)

        self.assertEqual(result.total_trades, 1)
        self.assertEqual(result.win_rate, 0.0)
        self.assertEqual(result.profit_factor, 0.0)


class BuildFixedReplayDepthBookTest(unittest.TestCase):
    def test_symbol_variants_use_same_template(self):
        mark = 10.0
        for coin in EXPECTED_COINS:
            template = FIXED_REPLAY_DEPTH_BOOKS[coin]
            expected_bid = float(_D(mark) * _D(template["bids"][0][0]) / _D(template["mid"]))
            expected_ask = float(_D(mark) * _D(template["asks"][0][0]) / _D(template["mid"]))
            for symbol in (coin, coin + "USDC", coin + "USDT", coin + "/USDC", coin + "-USDC"):
                with self.subTest(symbol=symbol):
                    book = build_fixed_replay_depth_book(mark, symbol)
                    self.assertEqual(len(book["bids"]), 20)
                    self.assertEqual(len(book["asks"]), 20)
                    self.assertAlmostEqual(book["bids"][0][0], expected_bid, places=12)
                    self.assertAlmostEqual(book["asks"][0][0], expected_ask, places=12)
                    self.assertEqual(book["bids"][0][1], float(template["bids"][0][1]))
                    self.assertEqual(book["asks"][0][1], float(template["asks"][0][1]))

    def test_unknown_symbol_falls_back_to_btc(self):
        btc = build_fixed_replay_depth_book(10.0, "BTC")
        unknown = build_fixed_replay_depth_book(10.0, "NOTACOIN")
        self.assertEqual(unknown, btc)

    def test_non_positive_mark_returns_empty_book(self):
        self.assertEqual(build_fixed_replay_depth_book(0.0, "SUI"), {"bids": [], "asks": []})
        self.assertEqual(build_fixed_replay_depth_book(-1.0, "SUI"), {"bids": [], "asks": []})


class FloorQtyToLotSizeTest(unittest.TestCase):
    def test_default_lot_size_zero_is_passthrough(self):
        self.assertEqual(floor_qty_to_lot_size(0.5), 0.5)
        self.assertEqual(floor_qty_to_lot_size(123.456789), 123.456789)

    def test_negative_qty_made_positive(self):
        self.assertEqual(floor_qty_to_lot_size(-2.5), 2.5)

    def test_lot_size_floors_to_multiple(self):
        self.assertEqual(floor_qty_to_lot_size(1.23, lot_size=0.1), 1.2000000000000002)


class SimulateReplayBaselineFillTest(unittest.TestCase):
    MARK = 2.5
    SYMBOL = "SUIUSDC"

    def test_buy_within_first_sui_ask_level(self):
        avg, filled, notional = simulate_replay_baseline_fill(+1, 1000.0, self.MARK, self.SYMBOL)
        self.assertAlmostEqual(avg, 2.50010142399286, places=12)
        self.assertEqual(filled, 1000.0)
        self.assertAlmostEqual(notional, avg * filled, places=9)

    def test_sell_within_first_sui_bid_level(self):
        avg, filled, notional = simulate_replay_baseline_fill(-1, 1000.0, self.MARK, self.SYMBOL)
        self.assertAlmostEqual(avg, 2.49989857600714, places=12)
        self.assertEqual(filled, 1000.0)
        self.assertAlmostEqual(notional, avg * filled, places=9)

    def test_buy_walks_sui_levels_when_requested_qty_exceeds_best_ask(self):
        fills = simulate_replay_baseline_fills_per_level(+1, 5000.0, self.MARK, self.SYMBOL)
        self.assertGreater(len(fills), 1)
        prices = [p for p, _ in fills]
        self.assertEqual(prices, sorted(prices))
        avg, filled, notional = simulate_replay_baseline_fill(+1, 5000.0, self.MARK, self.SYMBOL)
        self.assertAlmostEqual(sum(q for _, q in fills), filled, places=9)
        self.assertAlmostEqual(sum(p * q for p, q in fills), notional, places=6)
        self.assertAlmostEqual(avg, notional / filled, places=9)

    def test_zero_qty_or_mark_returns_zeros(self):
        self.assertEqual(simulate_replay_baseline_fill(+1, 0.0, self.MARK, self.SYMBOL), (0.0, 0.0, 0.0))
        self.assertEqual(simulate_replay_baseline_fill(+1, 1.0, 0.0, self.SYMBOL), (0.0, 0.0, 0.0))


class SynthesizeReplayMinuteCandlesTest(unittest.TestCase):
    def test_two_15m_bars_expand_to_30_1m_bars(self):
        df = pd.DataFrame([
            {"timestamp": "2025-10-06T00:00:00Z", "open": 100.0, "high": 116.0, "low": 99.0, "close": 115.0, "volume": 15.0},
            {"timestamp": "2025-10-06T00:15:00Z", "open": 115.0, "high": 131.0, "low": 114.0, "close": 130.0, "volume": 30.0},
        ])
        out = synthesize_replay_minute_candles(df)
        self.assertEqual(len(out), 30)
        self.assertEqual(out.iloc[0]["open"], 115.0)
        self.assertEqual(out.iloc[15]["close"], 130.0)
        self.assertAlmostEqual(out.iloc[0]["volume"], 1.0, places=9)
        self.assertAlmostEqual(out.iloc[15]["volume"], 2.0, places=9)

    def test_already_1m_data_passes_through(self):
        df = pd.DataFrame([
            {"timestamp": "2025-10-06T00:00:00Z", "open": 100.0, "high": 101.0, "low": 99.5, "close": 100.5, "volume": 1.0},
            {"timestamp": "2025-10-06T00:01:00Z", "open": 100.5, "high": 101.5, "low": 100.0, "close": 101.0, "volume": 1.0},
        ])
        out = synthesize_replay_minute_candles(df)
        self.assertEqual(len(out), 2)
        self.assertEqual(float(out.iloc[0]["close"]), 100.5)

    def test_empty_df_returns_empty(self):
        out = synthesize_replay_minute_candles(pd.DataFrame())
        self.assertEqual(len(out), 0)
        for col in ("timestamp", "open", "high", "low", "close", "volume"):
            self.assertIn(col, out.columns)


class BuildFixedReplayFundingEventsTest(unittest.TestCase):
    def test_three_hour_window_emits_three_events(self):
        df = pd.DataFrame({
            "timestamp": pd.date_range("2026-01-01T00:30:00Z", "2026-01-01T03:30:00Z", freq="15min", tz="UTC"),
            "open": [100.0] * 13,
            "high": [100.0] * 13,
            "low": [100.0] * 13,
            "close": list(range(100, 113)),
            "volume": [1.0] * 13,
        })
        events = build_fixed_replay_funding_events(df)
        self.assertEqual(len(events), 3)
        self.assertEqual(events[0]["timestamp"], pd.Timestamp("2026-01-01T01:00:00Z"))
        self.assertEqual(events[-1]["timestamp"], pd.Timestamp("2026-01-01T03:00:00Z"))
        for event in events:
            self.assertEqual(event["funding_rate"], FIXED_REPLAY_FUNDING_RATE)
            self.assertGreater(event["oracle_price"], 0)

    def test_subhour_window_emits_no_events(self):
        df = pd.DataFrame({
            "timestamp": pd.date_range("2026-01-01T00:05:00Z", "2026-01-01T00:55:00Z", freq="15min", tz="UTC"),
            "open": [100.0] * 4,
            "high": [100.0] * 4,
            "low": [100.0] * 4,
            "close": [100.0] * 4,
            "volume": [1.0] * 4,
        })
        self.assertEqual(build_fixed_replay_funding_events(df), [])


if __name__ == "__main__":
    unittest.main()
