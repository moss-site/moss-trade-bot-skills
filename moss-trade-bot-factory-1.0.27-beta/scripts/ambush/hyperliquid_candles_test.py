"""Unit tests for ambush.hyperliquid_candles.fetch — HTTP wrapper around
HL public info candleSnapshot."""
from __future__ import annotations

import json
import os
import sys
import unittest
from unittest.mock import patch, MagicMock

_HERE = os.path.dirname(os.path.abspath(__file__))
_SCRIPTS = os.path.dirname(_HERE)
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

from ambush import hyperliquid_candles as hc


def _fake_hl_response(n: int = 3) -> bytes:
    """Build n entries matching HL's actual candleSnapshot wire shape."""
    base_ts_ms = 1_700_000_000_000
    entries = []
    for i in range(n):
        entries.append({
            "t": base_ts_ms + i * 15 * 60 * 1000,   # bar open ts
            "T": base_ts_ms + (i + 1) * 15 * 60 * 1000 - 1,  # bar close ts
            "s": "SAGA",
            "i": "15m",
            "o": str(100 + i),
            "c": str(100 + i + 0.5),
            "h": str(100 + i + 1),
            "l": str(100 + i - 1),
            "v": str(1000 + i),
            "n": 10 + i,
        })
    return json.dumps(entries).encode()


class TestFetch(unittest.TestCase):
    @patch("ambush.hyperliquid_candles._urlopen")
    def test_happy_path_normalizes_payload(self, mock_urlopen):
        mock_resp = MagicMock()
        mock_resp.read.return_value = _fake_hl_response(n=3)
        mock_urlopen.return_value.__enter__.return_value = mock_resp
        bars = hc.fetch("SAGA", interval="15m", limit=3)
        self.assertEqual(len(bars), 3)
        first = bars[0]
        # Numbers parsed from string to float
        self.assertIsInstance(first["open"], float)
        self.assertEqual(first["open"], 100.0)
        self.assertEqual(first["close"], 100.5)
        # Timestamp parsed from ms to datetime
        from datetime import datetime, timezone
        self.assertIsInstance(first["ts"], datetime)
        self.assertEqual(first["ts"].tzinfo, timezone.utc)

    @patch("ambush.hyperliquid_candles._urlopen")
    def test_http_error_raises(self, mock_urlopen):
        import urllib.error
        mock_urlopen.side_effect = urllib.error.HTTPError(
            "url", 503, "Service Unavailable", {}, None,
        )
        with self.assertRaises(hc.HyperliquidCandleError):
            hc.fetch("SAGA", interval="15m", limit=3)

    @patch("ambush.hyperliquid_candles._urlopen")
    def test_malformed_response_raises(self, mock_urlopen):
        mock_resp = MagicMock()
        mock_resp.read.return_value = b"not json"
        mock_urlopen.return_value.__enter__.return_value = mock_resp
        with self.assertRaises(hc.HyperliquidCandleError):
            hc.fetch("SAGA", interval="15m", limit=3)

    @patch("ambush.hyperliquid_candles._urlopen")
    def test_missing_field_raises(self, mock_urlopen):
        mock_resp = MagicMock()
        # Missing "c" (close) field
        mock_resp.read.return_value = json.dumps([
            {"t": 1, "T": 2, "s": "X", "i": "15m", "o": "1", "h": "1", "l": "1", "v": "1"}
        ]).encode()
        mock_urlopen.return_value.__enter__.return_value = mock_resp
        with self.assertRaises(hc.HyperliquidCandleError):
            hc.fetch("SAGA", interval="15m", limit=3)

    @patch("ambush.hyperliquid_candles._urlopen")
    def test_bars_are_chronological(self, mock_urlopen):
        # HL returns oldest-first already; verify we preserve that order.
        mock_resp = MagicMock()
        mock_resp.read.return_value = _fake_hl_response(n=5)
        mock_urlopen.return_value.__enter__.return_value = mock_resp
        bars = hc.fetch("SAGA", interval="15m", limit=5)
        for i in range(1, len(bars)):
            self.assertGreater(bars[i]["ts"], bars[i - 1]["ts"])

    @patch("ambush.hyperliquid_candles._urlopen")
    def test_limit_trims_when_hl_returns_more(self, mock_urlopen):
        # HL returns 10 bars but caller only wants 5 — must get the most recent 5.
        mock_resp = MagicMock()
        mock_resp.read.return_value = _fake_hl_response(n=10)
        mock_urlopen.return_value.__enter__.return_value = mock_resp
        bars = hc.fetch("SAGA", interval="15m", limit=5)
        self.assertEqual(len(bars), 5)
        # After sort+trim we keep indices 5..9 (most recent); index 5 has open=105.0.
        self.assertEqual(bars[0]["open"], 105.0)

    @patch("ambush.hyperliquid_candles._urlopen")
    def test_url_error_raises(self, mock_urlopen):
        import urllib.error
        mock_urlopen.side_effect = urllib.error.URLError("DNS failure")
        with self.assertRaises(hc.HyperliquidCandleError):
            hc.fetch("SAGA", interval="15m", limit=3)


if __name__ == "__main__":
    unittest.main()
