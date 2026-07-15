from __future__ import annotations

import os
import sys
import unittest
from decimal import Decimal

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPTS = os.path.dirname(HERE)
if SCRIPTS not in sys.path:
    sys.path.insert(0, SCRIPTS)

from core.backtest import _AccountState, _apply_fill


def _fill(state, open_trade, side, qty, price, action):
    return _apply_fill(
        state, open_trade, side=side, qty=qty, fill_price=price,
        requested_leverage=10, fee_rate=0.0, fill_idx=0, fill_time=None,
        action=action, df=None,
    )


class PnlPctMarginBasisTest(unittest.TestCase):
    """pnl_pct must be realized PnL over the margin the trade actually employed.

    Regression: a partial close shrinks entry_margin to the *remaining* sliver
    while gross_realized_pnl keeps accumulating over the whole position, so the
    final pnl_pct was numerator(full size) / denominator(last sliver) — inflated
    by the scale-down ratio (observed ~10x in a live dataset).
    """

    def test_partial_close_then_full_close_uses_employed_margin(self):
        # Open 100 @ 100, leverage 10 -> margin employed = 100*100/10 = 1000
        state = _AccountState(wallet=Decimal(10000))
        ot, _ = _fill(state, None, "buy", 100, 100, "open_long")
        self.assertEqual(ot.entry_margin, Decimal(1000))

        # Partially close 90 @ 110 -> realized = (110-100)*90 = 900, position still open
        ot, done = _fill(state, ot, "sell", 90, 110, "reduce_long")
        self.assertEqual(done, [], "partial close must not finalize the trade")
        self.assertEqual(ot.gross_realized_pnl, Decimal(900))

        # Close the remaining 10 @ 110 -> realized += (110-100)*10 = 100 => 1000 total
        ot, done = _fill(state, ot, "sell", 10, 110, "close_long")
        self.assertEqual(len(done), 1)
        trade = done[0]

        # Price moved +10% at 10x on a position whose peak margin was 1000.
        # Correct pnl_pct = 1000 / 1000 = 1.0 (i.e. +100% on margin).
        # Buggy behaviour divided by the last sliver's margin (100) -> 10.0.
        self.assertAlmostEqual(trade.pnl_pct, 1.0, places=9)

    def test_no_partial_close_is_unchanged(self):
        # Guard the common path: open then fully close in one fill.
        state = _AccountState(wallet=Decimal(10000))
        ot, _ = _fill(state, None, "buy", 100, 100, "open_long")
        ot, done = _fill(state, ot, "sell", 100, 110, "close_long")
        self.assertEqual(len(done), 1)
        # realized = 1000, margin = 1000 -> 1.0
        self.assertAlmostEqual(done[0].pnl_pct, 1.0, places=9)

    def test_margin_times_pnl_pct_reconstructs_gross_pnl(self):
        """Contract with the Go consumer: it rebuilds absolute PnL as
        margin * pnl_pct (repository/backtest_canonical_persistence.go). That
        product must equal gross_pnl regardless of how the margin basis is
        chosen — this is why the old inflated pnl_pct still produced the right
        absolute PnL, and why the fix must not break it.
        """
        state = _AccountState(wallet=Decimal(10000))
        ot, _ = _fill(state, None, "buy", 100, 100, "open_long")
        ot, _ = _fill(state, ot, "sell", 90, 110, "reduce_long")
        ot, done = _fill(state, ot, "sell", 10, 110, "close_long")
        t = done[0]
        self.assertAlmostEqual(t.margin * t.pnl_pct, t.gross_pnl, places=6)
        self.assertAlmostEqual(t.gross_pnl, 1000.0, places=6)

    def test_increase_then_close_uses_peak_margin(self):
        # Increase path already grows entry_margin; lock that it stays correct.
        # NOTE: this case never partially closes, so it does NOT pin the
        # max_margin bookkeeping on the increase branch — see
        # test_increase_then_partial_close_then_close for that.
        state = _AccountState(wallet=Decimal(10000))
        ot, _ = _fill(state, None, "buy", 50, 100, "open_long")
        ot, _ = _fill(state, ot, "buy", 50, 100, "increase_long")
        self.assertEqual(ot.entry_margin, Decimal(1000))  # 100 @ 100 / 10x
        ot, done = _fill(state, ot, "sell", 100, 110, "close_long")
        self.assertAlmostEqual(done[0].pnl_pct, 1.0, places=9)

    def test_increase_then_partial_close_then_close(self):
        """Pins the max_margin update on the INCREASE branch.

        Regression guard: deleting `open_trade.max_margin = max(...)` from the
        same-direction branch leaves every other test green, but makes this case
        report pnl_pct 2.0 instead of 1.0 — the peak (1000) is forgotten and the
        post-open margin (500) becomes the divisor.
        """
        state = _AccountState(wallet=Decimal(10000))
        ot, _ = _fill(state, None, "buy", 50, 100, "open_long")     # margin 500
        ot, _ = _fill(state, ot, "buy", 50, 100, "increase_long")    # peak margin 1000
        self.assertEqual(ot.max_margin, Decimal(1000))
        ot, _ = _fill(state, ot, "sell", 90, 110, "reduce_long")     # entry_margin shrinks to 100
        ot, done = _fill(state, ot, "sell", 10, 110, "close_long")
        self.assertEqual(len(done), 1)
        # realized = (110-100)*100 = 1000 on a peak margin of 1000 -> 1.0
        self.assertAlmostEqual(done[0].pnl_pct, 1.0, places=9)
        self.assertAlmostEqual(done[0].margin, 1000.0, places=6)

    def test_short_partial_close_then_close(self):
        """Same defect, short side. The realized-PnL branch differs by sign, and
        every other test in this file is a long."""
        state = _AccountState(wallet=Decimal(10000))
        ot, _ = _fill(state, None, "sell", 100, 100, "open_short")   # margin 1000
        self.assertEqual(ot.entry_margin, Decimal(1000))
        ot, done = _fill(state, ot, "buy", 90, 90, "reduce_short")   # realized (100-90)*90 = 900
        self.assertEqual(done, [])
        self.assertEqual(ot.gross_realized_pnl, Decimal(900))
        ot, done = _fill(state, ot, "buy", 10, 90, "close_short")    # +100 -> 1000 total
        self.assertAlmostEqual(done[0].pnl_pct, 1.0, places=9)
        self.assertAlmostEqual(done[0].margin, 1000.0, places=6)

    def test_multiple_partial_closes(self):
        """Peak must survive repeated scale-downs, not just one."""
        state = _AccountState(wallet=Decimal(10000))
        ot, _ = _fill(state, None, "buy", 100, 100, "open_long")     # margin 1000
        for _ in range(3):
            ot, done = _fill(state, ot, "sell", 30, 110, "reduce_long")
            self.assertEqual(done, [])
        ot, done = _fill(state, ot, "sell", 10, 110, "close_long")
        # realized = (110-100)*100 = 1000 over peak margin 1000
        self.assertAlmostEqual(done[0].pnl_pct, 1.0, places=9)


if __name__ == "__main__":
    unittest.main()
