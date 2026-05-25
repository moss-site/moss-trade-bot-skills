from __future__ import annotations

import csv
import os
import sys
import tempfile
import unittest


HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPTS = os.path.dirname(HERE)
if SCRIPTS not in sys.path:
    sys.path.insert(0, SCRIPTS)

from dataset_catalog import choose, discover


def _write_csv(path: str, rows: list[str]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["timestamp", "open", "high", "low", "close", "volume"])
        for ts in rows:
            writer.writerow([ts, "1", "1", "1", "1", "1"])


class DatasetCatalogTest(unittest.TestCase):
    def test_discovers_actual_btc_304d_dataset(self):
        items = discover(os.path.join(SCRIPTS, "data_cache"), "15m")
        selected = choose(items, "BTC/USDT")
        self.assertIsNotNone(selected)
        self.assertEqual(selected["compact"], "BTCUSDC")
        self.assertEqual(selected["filename"], "hyperliquid_BTCUSDC_15m_2025-07-01_304d.csv")
        self.assertEqual(selected["start"], "2025-07-01 00:00:00")
        self.assertEqual(selected["end"], "2026-04-30 23:45:00")
        self.assertEqual(selected["bars"], 29184)

    def test_prefers_longer_dataset_when_legacy_and_new_both_exist(self):
        with tempfile.TemporaryDirectory() as tmp:
            _write_csv(
                os.path.join(tmp, "hyperliquid_BTCUSDC_15m_2025-10-06_148d.csv"),
                ["2025-10-06 00:00:00", "2025-10-06 00:15:00"],
            )
            _write_csv(
                os.path.join(tmp, "hyperliquid_BTCUSDC_15m_2025-07-01_304d.csv"),
                ["2025-07-01 00:00:00", "2025-07-01 00:15:00", "2025-07-01 00:30:00"],
            )
            selected = choose(discover(tmp, "15m"), "BTC/USDC")
            self.assertEqual(selected["dataset_version"], "2025-07-01_304d")
            self.assertEqual(selected["bars"], 3)

    def test_unknown_symbol_returns_none(self):
        items = discover(os.path.join(SCRIPTS, "data_cache"), "15m")
        self.assertIsNone(choose(items, "NOTREAL/USDC"))


if __name__ == "__main__":
    unittest.main()
