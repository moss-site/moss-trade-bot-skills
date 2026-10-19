from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
SCRIPTS = HERE.parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from core.leverage_caps import cap_params_for_symbol, max_leverage_for_symbol
from core.local_costs import normalize_coin


class PaperTokenTest(unittest.TestCase):
    def test_paragon_coin_routing(self):
        for symbol in ("PAPER", "PAPERUSDC", "PAPERUSDT", "paper/usdc", "para:PAPER"):
            with self.subTest(symbol=symbol):
                self.assertEqual(normalize_coin(symbol), "para:PAPER")

    def test_three_x_leverage_cap_and_aliases(self):
        for symbol in ("PAPER", "PAPERUSDC", "PAPERUSDT", "paper/usdt", "para:PAPER"):
            with self.subTest(symbol=symbol):
                self.assertEqual(max_leverage_for_symbol(symbol), 3)
        self.assertEqual(
            cap_params_for_symbol(
                {"base_leverage": 10, "max_leverage": 40},
                "PAPERUSDC",
            ),
            {"base_leverage": 3.0, "max_leverage": 3.0},
        )

    def test_release_manifest_contains_native_dataset(self):
        manifest = json.loads((SCRIPTS / "data_cache_manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["version"], "1.0.28")
        by_path = {item["path"]: item for item in manifest["files"]}
        path = "data_cache/hyperliquid_PAPERUSDC_15m_2026-10-18_1d.csv"
        self.assertIn(path, by_path)
        self.assertEqual(
            {
                key: by_path[path][key]
                for key in ("sha256", "row_count", "last_timestamp")
            },
            {
                "sha256": "a56a0c75b16d6265da33e5b39d9f6ba5861fcb93af792958f17ffedf5df72bd8",
                "row_count": 85,
                "last_timestamp": "2026-10-19 07:45:00",
            },
        )


if __name__ == "__main__":
    unittest.main()
