from __future__ import annotations

import hashlib
import io
import json
import os
import shutil
import sys
import tarfile
import tempfile
import threading
import unittest
from pathlib import Path

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPTS = os.path.dirname(HERE)
if SCRIPTS not in sys.path:
    sys.path.insert(0, SCRIPTS)

from core import data_cache_archive as dca


CSV_BYTES = b"timestamp,open,high,low,close,volume\n2026-01-01 00:00:00,1,1,1,1,10\n"


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _write_archive(path: Path, member_name: str = "data_cache/sample.csv", data: bytes = CSV_BYTES) -> str:
    with tarfile.open(path, "w:gz") as tf:
        info = tarfile.TarInfo("data_cache")
        info.type = tarfile.DIRTYPE
        tf.addfile(info)
        info = tarfile.TarInfo(member_name)
        info.size = len(data)
        tf.addfile(info, io.BytesIO(data))
    return dca._sha256_file(path)


def _write_manifest(tmp: Path, archive_sha: str, file_sha: str | None = None) -> Path:
    manifest = {
        "version": "test",
        "repo": "moss-site/moss-trade-bot-skills",
        "tag": "vtest",
        "asset_name": "data_cache-test.tar.gz",
        "archive_sha256": archive_sha,
        "files": [
            {
                "path": "data_cache/sample.csv",
                "sha256": file_sha or _sha256_bytes(CSV_BYTES),
                "row_count": 2,
                "last_timestamp": "2026-01-01 00:00:00",
            }
        ],
    }
    path = tmp / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path


class DataCacheArchiveTest(unittest.TestCase):
    def test_manifest_parse_reports_missing_fields(self):
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / "manifest.json"
            path.write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(dca.DataCacheError, "missing fields"):
                dca.load_manifest(path)

    def test_archive_checksum_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as raw:
            tmp = Path(raw)
            archive = tmp / "data_cache-test.tar.gz"
            _write_archive(archive)
            manifest_path = _write_manifest(tmp, "0" * 64)
            cache_root = tmp / "cache"
            old_manifest_path = dca.MANIFEST_PATH
            old_env = os.environ.get("MOSS_TRADE_BOT_CACHE_DIR")
            old_download = dca.download_archive
            try:
                dca.MANIFEST_PATH = manifest_path
                os.environ["MOSS_TRADE_BOT_CACHE_DIR"] = str(cache_root)
                dca.download_archive = lambda destination, manifest: shutil.copyfile(archive, destination)
                with self.assertRaisesRegex(dca.DataCacheError, "archive checksum mismatch"):
                    dca.ensure_data_cache()
            finally:
                dca.MANIFEST_PATH = old_manifest_path
                dca.download_archive = old_download
                if old_env is None:
                    os.environ.pop("MOSS_TRADE_BOT_CACHE_DIR", None)
                else:
                    os.environ["MOSS_TRADE_BOT_CACHE_DIR"] = old_env

    def test_csv_checksum_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as raw:
            tmp = Path(raw)
            archive = tmp / "data_cache-test.tar.gz"
            _write_archive(archive)
            manifest = dca.load_manifest(_write_manifest(tmp, dca._sha256_file(archive), "0" * 64))
            with self.assertRaisesRegex(dca.DataCacheError, "file checksum mismatch"):
                dca.extract_archive(archive, tmp / "out", manifest)

    def test_archive_path_traversal_is_rejected(self):
        with tempfile.TemporaryDirectory() as raw:
            tmp = Path(raw)
            archive = tmp / "bad.tar.gz"
            _write_archive(archive, "../escape.csv")
            manifest = dca.load_manifest(_write_manifest(tmp, dca._sha256_file(archive)))
            with self.assertRaisesRegex(dca.DataCacheError, "unsafe archive member path"):
                dca.extract_archive(archive, tmp / "out", manifest)

    def test_concurrent_hydrate_produces_complete_cache(self):
        with tempfile.TemporaryDirectory() as raw:
            tmp = Path(raw)
            archive = tmp / "data_cache-test.tar.gz"
            archive_sha = _write_archive(archive)
            manifest_path = _write_manifest(tmp, archive_sha)
            cache_root = tmp / "cache"
            old_manifest_path = dca.MANIFEST_PATH
            old_env = os.environ.get("MOSS_TRADE_BOT_CACHE_DIR")
            old_download = dca.download_archive
            try:
                dca.MANIFEST_PATH = manifest_path
                os.environ["MOSS_TRADE_BOT_CACHE_DIR"] = str(cache_root)
                dca.download_archive = lambda destination, manifest: shutil.copyfile(archive, destination)
                results = []
                errors = []

                def run():
                    try:
                        results.append(dca.ensure_data_cache())
                    except Exception as exc:  # pragma: no cover - surfaced below
                        errors.append(exc)

                threads = [threading.Thread(target=run) for _ in range(5)]
                for thread in threads:
                    thread.start()
                for thread in threads:
                    thread.join()

                self.assertEqual(errors, [])
                self.assertEqual(len(results), 5)
                self.assertTrue((results[0] / "sample.csv").is_file())
                dca.validate_data_cache(results[0], dca.load_manifest(manifest_path))
            finally:
                dca.MANIFEST_PATH = old_manifest_path
                dca.download_archive = old_download
                if old_env is None:
                    os.environ.pop("MOSS_TRADE_BOT_CACHE_DIR", None)
                else:
                    os.environ["MOSS_TRADE_BOT_CACHE_DIR"] = old_env


class NestedPathTest(unittest.TestCase):
    """Ambush data_cache uses nested layout (data_cache/ambush/klines/<BASE>.csv).
    Verify the helper preserves subdir structure during validate + extract."""

    def test_validate_preserves_subdir(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            data_dir = tmp / "data_cache"
            (data_dir / "ambush" / "klines").mkdir(parents=True)
            csv_path = data_dir / "ambush" / "klines" / "SAGA.csv"
            csv_path.write_bytes(CSV_BYTES)
            # CSV_BYTES has 1 header + 1 data row → 2 rows total
            row_count = CSV_BYTES.count(b"\n")
            last_ts = CSV_BYTES.splitlines()[-1].split(b",", 1)[0].decode()
            manifest = {
                "version": "test",
                "repo": "x/y",
                "tag": "t",
                "asset_name": "n.tar.gz",
                "archive_sha256": "x" * 64,
                "files": [
                    {
                        "path": "data_cache/ambush/klines/SAGA.csv",
                        "sha256": _sha256_bytes(CSV_BYTES),
                        "row_count": row_count,
                        "last_timestamp": last_ts,
                    }
                ],
            }
            dca.validate_data_cache(data_dir, manifest)

    def test_validate_skips_csv_check_for_json(self):
        """JSON sidecars (supply.json etc.) have sha256 but no row_count/last_ts."""
        json_bytes = b'{"foo": "bar"}'
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            data_dir = tmp / "data_cache" / "ambush"
            data_dir.mkdir(parents=True)
            json_path = data_dir / "supply.json"
            json_path.write_bytes(json_bytes)
            manifest = {
                "version": "test",
                "repo": "x/y",
                "tag": "t",
                "asset_name": "n.tar.gz",
                "archive_sha256": "x" * 64,
                "files": [
                    {
                        "path": "data_cache/ambush/supply.json",
                        "sha256": _sha256_bytes(json_bytes),
                        "row_count": None,
                        "last_timestamp": None,
                    }
                ],
            }
            dca.validate_data_cache(tmp / "data_cache", manifest)

    def test_safe_member_name_subdir(self):
        expected = {"data_cache/ambush/klines/SAGA.csv"}
        # File entry
        self.assertEqual(
            dca._safe_member_name("data_cache/ambush/klines/SAGA.csv", expected),
            "ambush/klines/SAGA.csv",
        )
        # Intermediate dir prefix
        self.assertEqual(
            dca._safe_member_name("data_cache/ambush", expected),
            "ambush",
        )
        self.assertEqual(
            dca._safe_member_name("data_cache/ambush/klines", expected),
            "ambush/klines",
        )
        # Unexpected file should raise
        with self.assertRaises(dca.DataCacheError):
            dca._safe_member_name("data_cache/ambush/klines/EVIL.csv", expected)
        # Path traversal must raise
        with self.assertRaises(dca.DataCacheError):
            dca._safe_member_name("../etc/passwd", expected)


if __name__ == "__main__":
    unittest.main()
