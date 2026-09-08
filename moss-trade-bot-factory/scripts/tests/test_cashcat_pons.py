from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
SCRIPTS = HERE.parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from core.leverage_caps import max_leverage_for_symbol
from core.local_costs import normalize_coin


class CashcatAndPonsTest(unittest.TestCase):
    def test_main_board_coin_routing(self):
        for symbol in ("CASHCAT", "CASHCATUSDC", "PONS", "PONSUSDC"):
            with self.subTest(symbol=symbol):
                expected = "CASHCAT" if symbol.startswith("CASHCAT") else "PONS"
                self.assertEqual(normalize_coin(symbol), expected)

    def test_three_x_leverage_caps_and_aliases(self):
        for symbol in ("CASHCAT", "CASHCATUSDC", "CASHCATUSDT", "PONS", "PONSUSDC", "PONSUSDT"):
            with self.subTest(symbol=symbol):
                self.assertEqual(max_leverage_for_symbol(symbol), 3)

    def test_release_manifest_contains_exact_native_datasets(self):
        manifest = json.loads((SCRIPTS / "data_cache_manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["version"], "1.0.28")
        by_path = {item["path"]: item for item in manifest["files"]}
        expected = {
            "data_cache/hyperliquid_CASHCATUSDC_15m_2026-07-17_53d.csv": {
                "sha256": "7182ccbdc0ece4d72517b1571575b2bb3431d81b0d8c6c74328a899a1dafbad2",
                "row_count": 5081,
                "last_timestamp": "2026-09-08 02:30:00",
            },
            "data_cache/hyperliquid_PONSUSDC_15m_2026-08-31_8d.csv": {
                "sha256": "ef3230e05aa46cb3e0a57fdd73126ad4ef04ed3e3f4f7ff9753ce49fa75f923b",
                "row_count": 765,
                "last_timestamp": "2026-09-08 02:30:00",
            },
        }
        for path, fingerprint in expected.items():
            with self.subTest(path=path):
                self.assertIn(path, by_path)
                self.assertEqual(
                    {key: by_path[path][key] for key in fingerprint},
                    fingerprint,
                )


if __name__ == "__main__":
    unittest.main()
