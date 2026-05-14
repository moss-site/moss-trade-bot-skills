"""Parity tests for core.replay_baseline against backend Go realtime replay.

`replay_baseline_v1` is the local-backtest execution profile that must produce
byte-for-byte identical fills, fees, and funding to the Go replay verifier
(internal/service/realtime_boot_replay_baseline.go +
 internal/service/replay_mem_store.go). If any constant or helper drifts the
backend's stitched-replay verifier (run during release gating) rejects the
upload with a parity error. We catch the drift here, in a unit test, before
the upload ever leaves the dev box.

The numeric baselines pinned below are the same values that
internal/service/realtime_boot_replay_baseline.go has in Go (Decimal
literals). Changing either side without changing the other is the bug this
test exists to prevent.

Run from repo root:
    python3 -m unittest discover -s skill/production/scripts/tests
"""

from __future__ import annotations

import math
import os
import sys
import unittest
from datetime import timedelta

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPTS = os.path.dirname(HERE)
if SCRIPTS not in sys.path:
    sys.path.insert(0, SCRIPTS)

from core.replay_baseline import (
    FIXED_REPLAY_ASK_DEPTH_LEVELS,
    FIXED_REPLAY_BID_DEPTH_LEVELS,
    FIXED_REPLAY_DEPTH_SNAPSHOT_CAPTURED_AT,
    FIXED_REPLAY_DEPTH_SNAPSHOT_MID_PRICE,
    FIXED_REPLAY_FUNDING_RATE,
    FIXED_REPLAY_LOT_SIZE,
    FIXED_REPLAY_TAKER_FEE_RATE,
    REPLAY_BASELINE_PROFILE,
    build_fixed_replay_depth_book,
    build_fixed_replay_funding_events,
    floor_qty_to_lot_size,
    replay_baseline_assumptions_text,
    simulate_replay_baseline_fill,
    simulate_replay_baseline_fills_per_level,
    synthesize_replay_minute_candles,
)


class ProfileNameTest(unittest.TestCase):
    """The string the backtest output embeds as `execution_profile`."""

    def test_profile_name_is_replay_baseline_v1(self):
        # The Go verifier rejects uploads whose execution_profile does not
        # match "replay_baseline_v1". DO NOT rename without also updating
        # internal/service/replay_mem_store.go.
        self.assertEqual(REPLAY_BASELINE_PROFILE, "replay_baseline_v1")


class FixedConstantsTest(unittest.TestCase):
    """Constants that must match backend Go realtime_boot_replay_baseline.go.

    Each value here was sourced from a frozen Hyperliquid POST /info l2Book
    snapshot at 2026-04-10T10:34:16.333Z. If anyone re-captures the snapshot
    they MUST update both sides in lockstep.
    """

    def test_taker_fee_rate(self):
        # Matches Go's market.DefaultSourcePerpTakerFeeRate = "0.00045"
        # (= VIP 0 taker rate in source_fee_tiers.go).
        self.assertEqual(FIXED_REPLAY_TAKER_FEE_RATE, 0.00045)

    def test_funding_rate_per_settlement(self):
        # Matches Go's fixedReplayFundingRate = "0.0000125" (per hourly tick).
        self.assertEqual(FIXED_REPLAY_FUNDING_RATE, 0.0000125)

    def test_lot_size_is_zero_in_replay(self):
        # Go's realtime replay wires TradeService without a metadata provider,
        # so lot-size flooring is effectively disabled. Both sides must agree.
        self.assertEqual(FIXED_REPLAY_LOT_SIZE, 0.0)

    def test_depth_snapshot_mid_price(self):
        # Matches Go's fixedReplayDepthSnapshotMidPrice = "71703.5".
        self.assertEqual(FIXED_REPLAY_DEPTH_SNAPSHOT_MID_PRICE, 71703.5)

    def test_depth_snapshot_captured_at(self):
        # The Go side stores the same moment as time.Date(...) with nanosecond
        # precision; we pin the ISO-8601 string so any drift in the snapshot
        # provenance is obvious.
        self.assertEqual(
            FIXED_REPLAY_DEPTH_SNAPSHOT_CAPTURED_AT, "2026-04-10T10:34:16.333Z"
        )

    def test_assumptions_text_is_stable(self):
        text = replay_baseline_assumptions_text()
        # Pin the headline numbers users see; if any drift the docs/comments
        # they read may quietly diverge from the runtime.
        self.assertIn("4.5bps", text)
        self.assertIn("0.00125", text)


class DepthLevelTablesTest(unittest.TestCase):
    """Exact bid/ask level tables — 20 levels each, ordered, matching Go."""

    def test_bid_count_is_20(self):
        self.assertEqual(len(FIXED_REPLAY_BID_DEPTH_LEVELS), 20)

    def test_ask_count_is_20(self):
        self.assertEqual(len(FIXED_REPLAY_ASK_DEPTH_LEVELS), 20)

    def test_bids_in_descending_price_order(self):
        prices = [p for p, _ in FIXED_REPLAY_BID_DEPTH_LEVELS]
        self.assertEqual(prices, sorted(prices, reverse=True))

    def test_asks_in_ascending_price_order(self):
        prices = [p for p, _ in FIXED_REPLAY_ASK_DEPTH_LEVELS]
        self.assertEqual(prices, sorted(prices))

    def test_top_of_book_pinned(self):
        # First entries — match Go's fixedReplayBidDepthLevels[0] / Asks[0].
        self.assertEqual(FIXED_REPLAY_BID_DEPTH_LEVELS[0], (71703.0, 7.44435))
        self.assertEqual(FIXED_REPLAY_ASK_DEPTH_LEVELS[0], (71704.0, 0.19192))

    def test_asks_skip_71706(self):
        # Faithful reproduction of the raw HL snapshot: 71706 was empty in the
        # captured l2Book and the Go side keeps the skip too.
        prices = {p for p, _ in FIXED_REPLAY_ASK_DEPTH_LEVELS}
        self.assertIn(71704.0, prices)
        self.assertIn(71705.0, prices)
        self.assertNotIn(71706.0, prices)
        self.assertIn(71707.0, prices)

    def test_bid_ask_spread_is_one_dollar(self):
        # Snapshot mid 71703.5 came from (best_bid 71703 + best_ask 71704) / 2.
        best_bid = FIXED_REPLAY_BID_DEPTH_LEVELS[0][0]
        best_ask = FIXED_REPLAY_ASK_DEPTH_LEVELS[0][0]
        self.assertEqual(best_ask - best_bid, 1.0)
        self.assertEqual((best_bid + best_ask) / 2, FIXED_REPLAY_DEPTH_SNAPSHOT_MID_PRICE)


class BuildFixedReplayDepthBookTest(unittest.TestCase):
    """Linear scaling around mark; identity at the snapshot mid."""

    def test_identity_scaling_at_snapshot_mid(self):
        book = build_fixed_replay_depth_book(FIXED_REPLAY_DEPTH_SNAPSHOT_MID_PRICE)
        self.assertEqual(book["bids"], FIXED_REPLAY_BID_DEPTH_LEVELS)
        self.assertEqual(book["asks"], FIXED_REPLAY_ASK_DEPTH_LEVELS)

    def test_zero_mark_returns_empty_book(self):
        book = build_fixed_replay_depth_book(0.0)
        self.assertEqual(book, {"bids": [], "asks": []})

    def test_negative_mark_returns_empty_book(self):
        book = build_fixed_replay_depth_book(-1.0)
        self.assertEqual(book, {"bids": [], "asks": []})

    def test_doubled_mark_doubles_each_level(self):
        # mark = 143407.0 = exactly 2 * 71703.5. Every scaled price doubles.
        book = build_fixed_replay_depth_book(143407.0)
        self.assertEqual(len(book["bids"]), 20)
        self.assertEqual(len(book["asks"]), 20)
        for (snap_price, _), (scaled_price, _) in zip(
            FIXED_REPLAY_BID_DEPTH_LEVELS, book["bids"]
        ):
            self.assertAlmostEqual(scaled_price, snap_price * 2.0, places=6)
        for (snap_price, _), (scaled_price, _) in zip(
            FIXED_REPLAY_ASK_DEPTH_LEVELS, book["asks"]
        ):
            self.assertAlmostEqual(scaled_price, snap_price * 2.0, places=6)

    def test_qty_unaffected_by_scaling(self):
        # Only price scales; qty must pass through unchanged on every level.
        book = build_fixed_replay_depth_book(50000.0)
        for (_, snap_qty), (_, scaled_qty) in zip(
            FIXED_REPLAY_BID_DEPTH_LEVELS, book["bids"]
        ):
            self.assertEqual(scaled_qty, snap_qty)
        for (_, snap_qty), (_, scaled_qty) in zip(
            FIXED_REPLAY_ASK_DEPTH_LEVELS, book["asks"]
        ):
            self.assertEqual(scaled_qty, snap_qty)


class FloorQtyToLotSizeTest(unittest.TestCase):
    """LotSize=0 is the replay default → flooring is a no-op."""

    def test_default_lot_size_zero_is_passthrough(self):
        self.assertEqual(floor_qty_to_lot_size(0.5), 0.5)
        self.assertEqual(floor_qty_to_lot_size(123.456789), 123.456789)

    def test_zero_qty_returns_zero(self):
        self.assertEqual(floor_qty_to_lot_size(0.0), 0.0)

    def test_negative_qty_made_positive(self):
        # Spec from the code: returns max(0, abs(qty)).
        self.assertEqual(floor_qty_to_lot_size(-2.5), 2.5)

    def test_lot_size_floors_to_multiple(self):
        self.assertEqual(floor_qty_to_lot_size(1.23, lot_size=0.1), 1.2000000000000002)

    def test_qty_below_one_lot_returns_zero(self):
        self.assertEqual(floor_qty_to_lot_size(0.05, lot_size=0.1), 0.0)

    def test_exact_lot_multiple_unchanged(self):
        self.assertAlmostEqual(floor_qty_to_lot_size(2.0, lot_size=0.5), 2.0)


class SimulateReplayBaselineFillTest(unittest.TestCase):
    """Multi-level walk parity with Go executeAcrossBookLevels."""

    MARK = 71703.5  # use snapshot mid so price scaling is identity

    def test_buy_within_first_ask_level(self):
        # qty=0.1 < ask[0].qty=0.19192 → single-level fill at 71704.0.
        avg, filled, notional = simulate_replay_baseline_fill(+1, 0.1, self.MARK)
        self.assertAlmostEqual(avg, 71704.0, places=9)
        self.assertAlmostEqual(filled, 0.1, places=9)
        self.assertAlmostEqual(notional, 71704.0 * 0.1, places=6)

    def test_sell_within_first_bid_level(self):
        # qty=0.5 < bid[0].qty=7.44435 → single-level fill at 71703.0.
        avg, filled, notional = simulate_replay_baseline_fill(-1, 0.5, self.MARK)
        self.assertAlmostEqual(avg, 71703.0, places=9)
        self.assertAlmostEqual(filled, 0.5, places=9)
        self.assertAlmostEqual(notional, 71703.0 * 0.5, places=6)

    def test_buy_exact_first_level_qty(self):
        # qty equals exactly ask[0].qty → single-level fill, no remainder.
        avg, filled, notional = simulate_replay_baseline_fill(+1, 0.19192, self.MARK)
        self.assertAlmostEqual(avg, 71704.0, places=9)
        self.assertAlmostEqual(filled, 0.19192, places=9)

    def test_buy_walks_multiple_levels(self):
        # qty=1.0 walks at least 6 ask levels (cumulative ~0.52573 → 1.57156).
        avg, filled, notional = simulate_replay_baseline_fill(+1, 1.0, self.MARK)
        self.assertAlmostEqual(filled, 1.0, places=9)
        # Avg price strictly between first ask (best) and the walked-into level.
        self.assertGreater(avg, 71704.0)
        self.assertLess(avg, 71710.5)
        self.assertAlmostEqual(notional, avg * 1.0, places=6)

    def test_buy_exhausts_book_when_requested_too_large(self):
        # Sum the captured ask qty; request more than that.
        total_ask_qty = sum(q for _, q in FIXED_REPLAY_ASK_DEPTH_LEVELS)
        avg, filled, notional = simulate_replay_baseline_fill(
            +1, total_ask_qty * 2.0, self.MARK
        )
        # Filled == total available; the rest is silently dropped (matches Go).
        self.assertAlmostEqual(filled, total_ask_qty, places=5)
        # Avg price falls inside the walked range.
        self.assertGreater(avg, 71704.0)
        self.assertLess(avg, FIXED_REPLAY_ASK_DEPTH_LEVELS[-1][0] + 1.0)

    def test_zero_qty_returns_zeros(self):
        avg, filled, notional = simulate_replay_baseline_fill(+1, 0.0, self.MARK)
        self.assertEqual((avg, filled, notional), (0.0, 0.0, 0.0))

    def test_zero_mark_returns_zeros(self):
        avg, filled, notional = simulate_replay_baseline_fill(+1, 1.0, 0.0)
        self.assertEqual((avg, filled, notional), (0.0, 0.0, 0.0))

    def test_buy_at_doubled_mark_scales_prices(self):
        # mark = 2 * snapshot_mid → all ask prices double, fills double too.
        avg_base, _, _ = simulate_replay_baseline_fill(+1, 0.1, self.MARK)
        avg_2x, _, _ = simulate_replay_baseline_fill(+1, 0.1, self.MARK * 2.0)
        self.assertAlmostEqual(avg_2x, avg_base * 2.0, places=6)


class SimulateReplayBaselineFillsPerLevelTest(unittest.TestCase):
    """Per-level emission parity with Go's one-SourceFill-per-depth-level rule."""

    MARK = 71703.5

    def test_single_level_emits_one_fill(self):
        fills = simulate_replay_baseline_fills_per_level(+1, 0.1, self.MARK)
        self.assertEqual(len(fills), 1)
        self.assertAlmostEqual(fills[0][0], 71704.0, places=9)
        self.assertAlmostEqual(fills[0][1], 0.1, places=9)

    def test_walks_levels_in_order(self):
        # qty=1.0 → walk asks in ascending price order.
        fills = simulate_replay_baseline_fills_per_level(+1, 1.0, self.MARK)
        self.assertGreater(len(fills), 1)
        prices = [p for p, _ in fills]
        self.assertEqual(prices, sorted(prices))
        # Final aggregate qty equals requested.
        total = sum(q for _, q in fills)
        self.assertAlmostEqual(total, 1.0, places=9)

    def test_sell_walks_bids_in_descending_order(self):
        fills = simulate_replay_baseline_fills_per_level(-1, 12.0, self.MARK)
        self.assertGreater(len(fills), 1)
        prices = [p for p, _ in fills]
        self.assertEqual(prices, sorted(prices, reverse=True))

    def test_zero_qty_emits_no_fills(self):
        self.assertEqual(simulate_replay_baseline_fills_per_level(+1, 0.0, self.MARK), [])

    def test_per_level_sum_matches_aggregate(self):
        # The per-level walk and the aggregate-avg walk MUST agree on total
        # filled qty and total notional (Go enforces this via shared helper).
        per_level = simulate_replay_baseline_fills_per_level(+1, 0.8, self.MARK)
        avg, filled, notional = simulate_replay_baseline_fill(+1, 0.8, self.MARK)
        per_qty = sum(q for _, q in per_level)
        per_notional = sum(p * q for p, q in per_level)
        self.assertAlmostEqual(per_qty, filled, places=9)
        self.assertAlmostEqual(per_notional, notional, places=4)
        if filled > 0:
            self.assertAlmostEqual(avg, per_notional / per_qty, places=6)


class SynthesizeReplayMinuteCandlesTest(unittest.TestCase):
    """15m bar → 15 synthetic 1m bars (open=high=low=close=row.close)."""

    def test_two_15m_bars_expand_to_30_1m_bars(self):
        df = pd.DataFrame(
            [
                {
                    "timestamp": "2025-10-06T00:00:00Z",
                    "open": 100.0, "high": 116.0, "low": 99.0, "close": 115.0,
                    "volume": 15.0,
                },
                {
                    "timestamp": "2025-10-06T00:15:00Z",
                    "open": 115.0, "high": 131.0, "low": 114.0, "close": 130.0,
                    "volume": 30.0,
                },
            ]
        )
        out = synthesize_replay_minute_candles(df)
        self.assertEqual(len(out), 30)

    def test_first_minute_uses_bar_close_for_all_ohlc(self):
        # 1.0.23.1's classic parity case: at the 15m boundary, the synthetic
        # 1m bar's open == bar.close (not bar.open). Pins the same behavior
        # the Go verifier uses for quote-at-time lookups.
        df = pd.DataFrame(
            [
                {
                    "timestamp": "2025-10-06T00:00:00Z",
                    "open": 100.0, "high": 116.0, "low": 99.0, "close": 115.0,
                    "volume": 15.0,
                },
                {
                    "timestamp": "2025-10-06T00:15:00Z",
                    "open": 115.0, "high": 131.0, "low": 114.0, "close": 130.0,
                    "volume": 30.0,
                },
            ]
        )
        out = synthesize_replay_minute_candles(df)
        first = out.iloc[0]
        self.assertEqual(first["open"], 115.0)
        self.assertEqual(first["high"], 115.0)
        self.assertEqual(first["low"], 115.0)
        self.assertEqual(first["close"], 115.0)

    def test_minute_timestamps_increment_by_one_minute(self):
        df = pd.DataFrame(
            [
                {
                    "timestamp": "2025-10-06T00:00:00Z",
                    "open": 100.0, "high": 100.0, "low": 100.0, "close": 100.0,
                    "volume": 15.0,
                },
                {
                    "timestamp": "2025-10-06T00:15:00Z",
                    "open": 100.0, "high": 100.0, "low": 100.0, "close": 100.0,
                    "volume": 15.0,
                },
            ]
        )
        out = synthesize_replay_minute_candles(df)
        deltas = out["timestamp"].diff().dropna()
        # 15 deltas of 1m within bar 0, then jump from 00:14→00:15, then
        # 14 deltas of 1m within bar 1 = 29 total — all = 1 minute.
        for delta in deltas:
            self.assertEqual(delta, pd.Timedelta(minutes=1))

    def test_second_bar_minute_uses_second_bar_close(self):
        df = pd.DataFrame(
            [
                {
                    "timestamp": "2025-10-06T00:00:00Z",
                    "open": 100.0, "high": 116.0, "low": 99.0, "close": 115.0,
                    "volume": 15.0,
                },
                {
                    "timestamp": "2025-10-06T00:15:00Z",
                    "open": 115.0, "high": 131.0, "low": 114.0, "close": 130.0,
                    "volume": 30.0,
                },
            ]
        )
        out = synthesize_replay_minute_candles(df)
        # bar 1 starts at row index 15 (00:15).
        row_15 = out.iloc[15]
        self.assertEqual(row_15["close"], 130.0)
        self.assertEqual(row_15["open"], 130.0)
        # Last minute of bar 1: row index 29 (00:29).
        row_29 = out.iloc[29]
        self.assertEqual(row_29["close"], 130.0)

    def test_volume_split_evenly_across_minutes(self):
        df = pd.DataFrame(
            [
                {
                    "timestamp": "2025-10-06T00:00:00Z",
                    "open": 100.0, "high": 116.0, "low": 99.0, "close": 115.0,
                    "volume": 15.0,
                },
                {
                    "timestamp": "2025-10-06T00:15:00Z",
                    "open": 115.0, "high": 131.0, "low": 114.0, "close": 130.0,
                    "volume": 30.0,
                },
            ]
        )
        out = synthesize_replay_minute_candles(df)
        # bar 0 volume 15 / 15 = 1.0 per minute
        for i in range(15):
            self.assertAlmostEqual(out.iloc[i]["volume"], 1.0, places=9)
        # bar 1 volume 30 / 15 = 2.0 per minute
        for i in range(15, 30):
            self.assertAlmostEqual(out.iloc[i]["volume"], 2.0, places=9)

    def test_already_1m_data_passes_through(self):
        df = pd.DataFrame(
            [
                {
                    "timestamp": "2025-10-06T00:00:00Z",
                    "open": 100.0, "high": 101.0, "low": 99.5, "close": 100.5,
                    "volume": 1.0,
                },
                {
                    "timestamp": "2025-10-06T00:01:00Z",
                    "open": 100.5, "high": 101.5, "low": 100.0, "close": 101.0,
                    "volume": 1.0,
                },
            ]
        )
        out = synthesize_replay_minute_candles(df)
        self.assertEqual(len(out), 2)
        # Step is already 1m → returns original frame.
        self.assertEqual(float(out.iloc[0]["open"]), 100.0)
        self.assertEqual(float(out.iloc[0]["close"]), 100.5)
        self.assertEqual(float(out.iloc[1]["close"]), 101.0)

    def test_single_row_passes_through(self):
        df = pd.DataFrame(
            [
                {
                    "timestamp": "2025-10-06T00:00:00Z",
                    "open": 100.0, "high": 116.0, "low": 99.0, "close": 115.0,
                    "volume": 15.0,
                },
            ]
        )
        out = synthesize_replay_minute_candles(df)
        self.assertEqual(len(out), 1)

    def test_empty_df_returns_empty(self):
        out = synthesize_replay_minute_candles(pd.DataFrame())
        self.assertEqual(len(out), 0)
        for col in ("timestamp", "open", "high", "low", "close", "volume"):
            self.assertIn(col, out.columns)


class BuildFixedReplayFundingEventsTest(unittest.TestCase):
    """Hourly funding settlement timeline parity with Go's funding worker."""

    def test_three_hour_window_emits_three_events(self):
        # Start 00:30, end 03:30 → settle at 01:00, 02:00, 03:00 = 3 events.
        df = pd.DataFrame(
            {
                "timestamp": pd.date_range(
                    "2026-01-01T00:30:00Z", "2026-01-01T03:30:00Z", freq="15min", tz="UTC"
                ),
                "open": [100.0] * 13,
                "high": [100.0] * 13,
                "low": [100.0] * 13,
                "close": list(range(100, 113)),
                "volume": [1.0] * 13,
            }
        )
        events = build_fixed_replay_funding_events(df)
        self.assertEqual(len(events), 3)
        self.assertEqual(events[0]["timestamp"], pd.Timestamp("2026-01-01T01:00:00Z"))
        self.assertEqual(events[-1]["timestamp"], pd.Timestamp("2026-01-01T03:00:00Z"))
        for event in events:
            self.assertEqual(event["funding_rate"], FIXED_REPLAY_FUNDING_RATE)
            self.assertGreater(event["oracle_price"], 0)

    def test_subhour_window_emits_no_events(self):
        # Window 00:05 → 00:55 has no integer-hour boundary inside.
        df = pd.DataFrame(
            {
                "timestamp": pd.date_range(
                    "2026-01-01T00:05:00Z", "2026-01-01T00:55:00Z", freq="15min", tz="UTC"
                ),
                "open": [100.0] * 4,
                "high": [100.0] * 4,
                "low": [100.0] * 4,
                "close": [100.0] * 4,
                "volume": [1.0] * 4,
            }
        )
        events = build_fixed_replay_funding_events(df)
        self.assertEqual(events, [])

    def test_oracle_price_uses_close_at_or_after_settlement(self):
        # Start 00:00, end 02:30 → settle at 01:00 + 02:00. Pin that the
        # oracle price for the 01:00 event uses the row at or after 01:00.
        df = pd.DataFrame(
            {
                "timestamp": pd.date_range(
                    "2026-01-01T00:00:00Z", "2026-01-01T02:30:00Z", freq="15min", tz="UTC"
                ),
                # 11 rows: 00:00, 00:15, 00:30, 00:45, 01:00, 01:15, 01:30, 01:45, 02:00, 02:15, 02:30
                "open": [100.0] * 11,
                "high": [100.0] * 11,
                "low": [100.0] * 11,
                # close indexed by row: 100 at 00:00, 104 at 01:00, 108 at 02:00 ...
                "close": [100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0, 109.0, 110.0],
                "volume": [1.0] * 11,
            }
        )
        events = build_fixed_replay_funding_events(df)
        self.assertEqual(len(events), 2)
        # 01:00 settlement → oracle close = 104 (row index 4 → timestamp 01:00)
        self.assertEqual(events[0]["oracle_price"], 104.0)
        self.assertEqual(events[1]["oracle_price"], 108.0)

    def test_empty_df_returns_no_events(self):
        events = build_fixed_replay_funding_events(pd.DataFrame())
        self.assertEqual(events, [])


if __name__ == "__main__":
    unittest.main()
