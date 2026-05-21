"""Regression test: data_cache must hold the expected USDC perp CSVs.

Background — 2026-05-11 R1 review §3 confirmed 22/22 dataset coverage after
the d64c018 backfill (XRP / ADA / ARB). This test pins that invariant: if a
CSV is removed, renamed, or truncated below its pinned coverage, the test fails
before tip-line bugs reach prod.

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
        "OP", "SOL", "SUI", "TRX", "UNI", "XRP", "HYPE",
        "AMD", "BRENTOIL", "CBRS", "CL", "COIN", "CRCL", "GOLD",
        "GOOGL", "INTC", "META", "MSTR", "MU", "NVDA", "ORCL",
        "SILVER", "SKHX", "SNDK", "SP500", "TSLA", "XYZ100",
    ]
)
EXPECTED_HEADER = ["timestamp", "open", "high", "low", "close", "volume"]
DEFAULT_DATASET_SUFFIX = "2025-10-06_148d"
DEFAULT_ROW_COUNT = 14209  # 148 days * 96 bars + 1 header
DEFAULT_LAST_TIMESTAMP = "2026-03-02 23:45:00"
SYMBOL_DATASET_OVERRIDES = {
    "BTC": {
        "suffix": "2025-07-01_304d",
        "row_count": 29185,
        "last_timestamp": "2026-04-30 23:45:00",
    },
    "AMD": {"row_count": 8489, "last_timestamp": "2026-03-03 00:15:00"},
    "BRENTOIL": {"row_count": 5004, "last_timestamp": "2026-05-18 09:00:00"},
    "CBRS": {"row_count": 1664, "last_timestamp": "2026-05-18 09:00:00"},
    "CL": {"row_count": 5321, "last_timestamp": "2026-03-03 00:15:00"},
    "COIN": {"row_count": 9343, "last_timestamp": "2026-03-03 00:15:00"},
    "CRCL": {"row_count": 7429, "last_timestamp": "2026-03-03 00:15:00"},
    "GOLD": {"row_count": 6761, "last_timestamp": "2026-03-03 00:15:00"},
    "GOOGL": {"row_count": 10018, "last_timestamp": "2026-03-03 00:15:00"},
    "HYPE": {"row_count": 14210, "last_timestamp": "2026-03-03 00:00:00"},
    "INTC": {"row_count": 8585, "last_timestamp": "2026-03-03 00:15:00"},
    "META": {"row_count": 9833, "last_timestamp": "2026-03-03 00:15:00"},
    "MSTR": {"row_count": 8679, "last_timestamp": "2026-03-03 00:15:00"},
    "MU": {"row_count": 7041, "last_timestamp": "2026-03-03 00:15:00"},
    "NVDA": {"row_count": 10579, "last_timestamp": "2026-03-03 00:15:00"},
    "ORCL": {"row_count": 8393, "last_timestamp": "2026-03-03 00:15:00"},
    "SILVER": {"row_count": 6435, "last_timestamp": "2026-03-03 00:15:00"},
    "SKHX": {"row_count": 1062, "last_timestamp": "2026-03-03 00:15:00"},
    "SNDK": {"row_count": 4745, "last_timestamp": "2026-03-03 00:15:00"},
    "SP500": {"row_count": 5003, "last_timestamp": "2026-05-18 09:00:00"},
    "TSLA": {"row_count": 10505, "last_timestamp": "2026-03-03 00:15:00"},
    "XYZ100": {"row_count": 13490, "last_timestamp": "2026-03-03 00:15:00"},
}


def _csv_path(symbol):
    meta = SYMBOL_DATASET_OVERRIDES.get(symbol, {})
    suffix = meta.get("suffix", DEFAULT_DATASET_SUFFIX)
    return os.path.join(DATA_CACHE, f"hyperliquid_{symbol}USDC_15m_{suffix}.csv")


class DataCacheCompleteTest(unittest.TestCase):
    def test_data_cache_dir_exists(self):
        self.assertTrue(os.path.isdir(DATA_CACHE), f"missing {DATA_CACHE}")

    def test_all_csv_files_present(self):
        missing = [s for s in EXPECTED_SYMBOLS if not os.path.isfile(_csv_path(s))]
        self.assertEqual(missing, [], f"missing CSV files: {missing}")

    def test_no_extraneous_csv_files(self):
        present = sorted(
            name for name in os.listdir(DATA_CACHE)
            if name.endswith(".csv") and name.startswith("hyperliquid_")
        )
        expected_files = sorted(
            os.path.basename(_csv_path(s)) for s in EXPECTED_SYMBOLS
        )
        self.assertEqual(
            present,
            expected_files,
            "data_cache CSV set drifted from the expected symbol whitelist",
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
                expected = SYMBOL_DATASET_OVERRIDES.get(symbol, {}).get("row_count", DEFAULT_ROW_COUNT)
                if row_count != expected:
                    failures.append(
                        f"{symbol}: row_count={row_count} (expected {expected})"
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
            expected = SYMBOL_DATASET_OVERRIDES.get(symbol, {}).get("last_timestamp", DEFAULT_LAST_TIMESTAMP)
            # Accept either plain timestamp or timezone / ISO variants
            if not last_ts.startswith(expected):
                failures.append(f"{symbol}: last_ts={last_ts}")
        if failures:
            self.fail(
                "data_cache CSV last timestamps drifted from the expected fixed "
                f"snapshot ({DEFAULT_LAST_TIMESTAMP}, with per-symbol overrides):\n  - "
                + "\n  - ".join(failures)
            )


if __name__ == "__main__":
    unittest.main()
