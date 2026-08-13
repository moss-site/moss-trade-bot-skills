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
    """2026-06-08: AAPL/TSM/SPCX; 2026-06-15: MSFT/MRVL/AVGO; 2026-06-22: ZHIPU
    added as xyz HIP-3 equities. They must route to the xyz: namespace for HL
    coin lookups and carry their live leverage caps."""

    def test_normalize_coin_prefixes_xyz(self):
        self.assertEqual(normalize_coin("AAPLUSDC"), "xyz:AAPL")
        self.assertEqual(normalize_coin("TSMUSDC"), "xyz:TSM")
        self.assertEqual(normalize_coin("SPCXUSDC"), "xyz:SPCX")
        self.assertEqual(normalize_coin("MSFTUSDC"), "xyz:MSFT")
        self.assertEqual(normalize_coin("MRVLUSDC"), "xyz:MRVL")
        self.assertEqual(normalize_coin("AVGOUSDC"), "xyz:AVGO")
        self.assertEqual(normalize_coin("ZHIPUUSDC"), "xyz:ZHIPU")
        # 2026-06-23: DRAM is xyz HIP-3 (Roundhill Memory ETF)
        self.assertEqual(normalize_coin("DRAMUSDC"), "xyz:DRAM")
        # 2026-06-23: ZEC/WLD are main-board crypto -> bare base, NOT xyz:
        self.assertEqual(normalize_coin("ZECUSDC"), "ZEC")
        self.assertEqual(normalize_coin("WLDUSDC"), "WLD")
        # 2026-07-15: SMSN + SKHY (SK Hynix ADS). SKHY != SKHX (common).
        self.assertEqual(normalize_coin("SMSNUSDC"), "xyz:SMSN")
        self.assertEqual(normalize_coin("SKHYUSDC"), "xyz:SKHY")
        self.assertEqual(normalize_coin("SKHXUSDC"), "xyz:SKHX")
        self.assertEqual(normalize_coin("CXMTUSDC"), "xyz:CXMT")
        self.assertEqual(normalize_coin("CRWVUSDC"), "xyz:CRWV")
        self.assertEqual(normalize_coin("NBISUSDC"), "xyz:NBIS")

    def test_leverage_caps(self):
        self.assertEqual(max_leverage_for_symbol("AAPLUSDC"), 20)
        self.assertEqual(max_leverage_for_symbol("TSMUSDC"), 10)
        self.assertEqual(max_leverage_for_symbol("SPCXUSDC"), 20)  # 2026-06-23: HL 10->20
        self.assertEqual(max_leverage_for_symbol("MSFTUSDC"), 20)
        self.assertEqual(max_leverage_for_symbol("MRVLUSDC"), 10)
        self.assertEqual(max_leverage_for_symbol("AVGOUSDC"), 10)
        # 2026-06-16: HL-sync — TSLA/META/GOOGL raised 10->20 (were stale)
        self.assertEqual(max_leverage_for_symbol("TSLAUSDC"), 20)
        self.assertEqual(max_leverage_for_symbol("METAUSDC"), 20)
        self.assertEqual(max_leverage_for_symbol("GOOGLUSDC"), 20)
        # 2026-06-22: ZHIPU
        self.assertEqual(max_leverage_for_symbol("ZHIPUUSDC"), 10)
        # 2026-06-23: ZEC/WLD main-board (10x), DRAM xyz ETF (20x)
        self.assertEqual(max_leverage_for_symbol("ZECUSDC"), 10)
        self.assertEqual(max_leverage_for_symbol("WLDUSDC"), 10)
        self.assertEqual(max_leverage_for_symbol("DRAMUSDC"), 20)
        # 2026-07-15: SMSN / SKHY both 10x
        self.assertEqual(max_leverage_for_symbol("SMSNUSDC"), 10)
        self.assertEqual(max_leverage_for_symbol("SKHYUSDC"), 10)
        # 2026-08-13: CXMT 5x -> 10x (HL 转换后上调)
        self.assertEqual(max_leverage_for_symbol("CXMTUSDC"), 10)
        # 2026-08-13: CRWV / NBIS 10x
        self.assertEqual(max_leverage_for_symbol("CRWVUSDC"), 10)
        self.assertEqual(max_leverage_for_symbol("NBISUSDC"), 10)


if __name__ == "__main__":
    unittest.main()
