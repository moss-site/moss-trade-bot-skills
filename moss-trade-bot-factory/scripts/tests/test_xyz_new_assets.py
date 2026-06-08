from __future__ import annotations

import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPTS = os.path.dirname(HERE)
if SCRIPTS not in sys.path:
    sys.path.insert(0, SCRIPTS)

from core.leverage_caps import max_leverage_for_symbol
from core.local_costs import normalize_coin


class NewXYZAssetsTest(unittest.TestCase):
    """2026-06-08: AAPL/TSM/SPCX added as xyz HIP-3 equities. They must route to
    the xyz: namespace for HL coin lookups and carry their live leverage caps."""

    def test_normalize_coin_prefixes_xyz(self):
        self.assertEqual(normalize_coin("AAPLUSDC"), "xyz:AAPL")
        self.assertEqual(normalize_coin("TSMUSDC"), "xyz:TSM")
        self.assertEqual(normalize_coin("SPCXUSDC"), "xyz:SPCX")

    def test_leverage_caps(self):
        self.assertEqual(max_leverage_for_symbol("AAPLUSDC"), 20)
        self.assertEqual(max_leverage_for_symbol("TSMUSDC"), 10)
        self.assertEqual(max_leverage_for_symbol("SPCXUSDC"), 5)


if __name__ == "__main__":
    unittest.main()
