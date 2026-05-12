"""Regression test: data_cache must hold the 22 expected USDC perp CSVs.

Background — 2026-05-11 R1 review §3 confirmed 22/22 dataset coverage after
the d64c018 backfill (XRP / ADA / ARB). This test pins that invariant: if a
CSV is removed, renamed, or truncated below 148 days × 96 bars (= 14208 rows
+ 1 header), the test fails before tip-line bugs reach prod.

Run from repo root:
    python3 -m unittest discover -s skill/production/scripts/tests
"""

from __future__ import annotations

import csv
import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPTS = os.path.dirname(HERE)
if SCRIPTS not in sys.path:
    sys.path.insert(0, SCRIPTS)

DATA_CACHE = os.path.join(SCRIPTS, "data_cache")

EXPECTED_SYMBOLS = sorted(
    [
        "ADA", "APT", "ARB", "ATOM", "AVAX", "BCH", "BNB", "BTC",
        "DOGE", "DOT", "ETH", "FIL", "HBAR", "LINK", "LTC", "NEAR",
        "OP", "SOL", "SUI", "TRX", "UNI", "XRP",
    ]
)
EXPECTED_HEADER = ["timestamp", "open", "high", "low", "close", "volume"]
EXPECTED_ROW_COUNT = 14209  # 148 days * 96 bars + 1 header
EXPECTED_LAST_TIMESTAMP = "2026-03-02 23:45:00"


def _csv_path(symbol):
    return os.path.join(DATA_CACHE, f"hyperliquid_{symbol}USDC_15m_2025-10-06_148d.csv")


class DataCacheCompleteTest(unittest.TestCase):
    def test_data_cache_dir_exists(self):
        self.assertTrue(os.path.isdir(DATA_CACHE), f"missing {DATA_CACHE}")

    def test_all_22_csv_files_present(self):
        missing = [s for s in EXPECTED_SYMBOLS if not os.path.isfile(_csv_path(s))]
        self.assertEqual(missing, [], f"missing CSV files: {missing}")

    def test_no_extraneous_csv_files(self):
        present = sorted(
            name for name in os.listdir(DATA_CACHE)
            if name.endswith(".csv") and name.startswith("hyperliquid_")
        )
        expected_files = sorted(
            f"hyperliquid_{s}USDC_15m_2025-10-06_148d.csv" for s in EXPECTED_SYMBOLS
        )
        self.assertEqual(
            present,
            expected_files,
            "data_cache CSV set drifted from the 22-symbol whitelist",
        )

    def test_each_csv_header_and_row_count(self):
        failures = []
        for symbol in EXPECTED_SYMBOLS:
            path = _csv_path(symbol)
            with open(path, encoding="utf-8") as f:
                reader = csv.reader(f)
                header = next(reader)
                if header != EXPECTED_HEADER:
                    failures.append(f"{symbol}: header={header}")
                    continue
                row_count = sum(1 for _ in reader) + 1  # +1 for header
                if row_count != EXPECTED_ROW_COUNT:
                    failures.append(
                        f"{symbol}: row_count={row_count} (expected {EXPECTED_ROW_COUNT})"
                    )
        if failures:
            self.fail("data_cache CSV structure drift:\n  - " + "\n  - ".join(failures))

    def test_each_csv_last_timestamp_aligned(self):
        failures = []
        for symbol in EXPECTED_SYMBOLS:
            path = _csv_path(symbol)
            with open(path, encoding="utf-8") as f:
                rows = list(csv.reader(f))
            last_ts = rows[-1][0]
            # Accept either "2026-03-02 23:45:00" or "2026-03-02 23:45:00+00:00" / ISO variants
            if not last_ts.startswith(EXPECTED_LAST_TIMESTAMP):
                failures.append(f"{symbol}: last_ts={last_ts}")
        if failures:
            self.fail(
                "data_cache CSV last timestamps drifted from the expected fixed "
                f"snapshot ({EXPECTED_LAST_TIMESTAMP}):\n  - "
                + "\n  - ".join(failures)
            )


if __name__ == "__main__":
    unittest.main()
