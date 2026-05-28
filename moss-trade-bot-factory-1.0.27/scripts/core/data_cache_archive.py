from __future__ import annotations

import csv
import hashlib
import json
import os
import posixpath
import shutil
import subprocess
import tarfile
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path, PurePosixPath
from typing import Any


SCRIPTS_DIR = Path(__file__).resolve().parents[1]
MANIFEST_PATH = SCRIPTS_DIR / "data_cache_manifest.json"
LOCAL_DATA_DIR = SCRIPTS_DIR / "data_cache"
LOCK_TIMEOUT_SECONDS = 600


class DataCacheError(RuntimeError):
    pass


def load_manifest(path: str | os.PathLike[str] | None = None) -> dict[str, Any]:
    manifest_path = Path(path or MANIFEST_PATH)
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise DataCacheError(f"failed to read data cache manifest: {manifest_path}: {exc}") from exc
    required = {"version", "repo", "tag", "asset_name", "archive_sha256", "files"}
    missing = sorted(required - set(manifest))
    if missing:
        raise DataCacheError(f"data cache manifest missing fields: {missing}")
    if not isinstance(manifest["files"], list) or not manifest["files"]:
        raise DataCacheError("data cache manifest has no files")
    return manifest


def cache_dir_for_manifest(manifest: dict[str, Any]) -> Path:
    root = Path(
        os.environ.get("MOSS_TRADE_BOT_CACHE_DIR", "~/.cache/moss-trade-bot-factory")
    ).expanduser()
    return root / f"v{manifest['version']}" / "data_cache"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _csv_fingerprint(path: Path) -> tuple[int, str]:
    try:
        with path.open(newline="", encoding="utf-8") as f:
            rows = list(csv.reader(f))
    except Exception as exc:
        raise DataCacheError(f"failed to read CSV {path}: {exc}") from exc
    if not rows:
        raise DataCacheError(f"empty CSV: {path}")
    return len(rows), rows[-1][0]


def validate_data_cache(data_dir: Path, manifest: dict[str, Any]) -> None:
    """Verify every manifest entry has a matching file in `data_dir`.

    Manifest paths are tarball-relative (e.g. "data_cache/hyperliquid_BTCUSDC...csv"
    for majors, "data_cache/ambush/klines/SAGA.csv" for ambush). We strip the
    leading "data_cache/" component so we land under our local `data_dir`,
    BUT preserve any deeper subdirectory structure — the ambush dataset relies
    on `ambush/klines/<BASE>.csv` etc. (Previous flat-only behaviour would
    collapse `ambush/klines/SAGA.csv` to `SAGA.csv`, breaking colocation.)

    CSV-specific row_count/last_timestamp checks only run on .csv entries.
    Ambush ships JSON sidecars (supply.json, market_cap_snapshot.json,
    sanity_events.json) — those are sha256-only.
    """
    for item in manifest["files"]:
        rel_path = item.get("path")
        if not rel_path:
            raise DataCacheError("manifest file entry missing path")
        rel = PurePosixPath(rel_path)
        if rel.parts and rel.parts[0] == "data_cache":
            rel = PurePosixPath(*rel.parts[1:])
        path = data_dir / Path(*rel.parts)
        if not path.is_file():
            raise DataCacheError(f"missing cached file: {path}")
        actual_sha = _sha256_file(path)
        if actual_sha != item.get("sha256"):
            raise DataCacheError(f"file checksum mismatch: {path}")
        if path.suffix.lower() != ".csv":
            continue
        row_count, last_ts = _csv_fingerprint(path)
        if row_count != item.get("row_count"):
            raise DataCacheError(f"CSV row_count mismatch: {path}")
        if last_ts != item.get("last_timestamp"):
            raise DataCacheError(f"CSV last_timestamp mismatch: {path}")


def _github_token() -> str | None:
    for env_name in ("GITHUB_TOKEN", "GH_TOKEN"):
        value = os.environ.get(env_name)
        if value:
            return value.strip()
    try:
        proc = subprocess.run(
            ["gh", "auth", "token"],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=10,
        )
    except Exception:
        return None
    token = proc.stdout.strip()
    return token or None


def _request(url: str, token: str | None = None, accept: str | None = None) -> urllib.request.Request:
    headers = {"User-Agent": "moss-trade-bot-factory-data-cache"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    if accept:
        headers["Accept"] = accept
    return urllib.request.Request(url, headers=headers)


def _release_asset_api_url(manifest: dict[str, Any], token: str) -> str:
    api_url = f"https://api.github.com/repos/{manifest['repo']}/releases/tags/{manifest['tag']}"
    with urllib.request.urlopen(
        _request(api_url, token=token, accept="application/vnd.github+json"),
        timeout=60,
    ) as resp:
        release = json.loads(resp.read().decode("utf-8"))
    for asset in release.get("assets", []):
        if asset.get("name") == manifest["asset_name"]:
            return asset["url"]
    raise DataCacheError(f"release asset not found: {manifest['asset_name']}")


def _download_help(manifest: dict[str, Any]) -> str:
    return (
        f"failed to download data cache release asset {manifest['asset_name']} "
        f"from {manifest['repo']} {manifest['tag']}. Check that the GitHub "
        "Release and asset exist. If the repository or asset is private, make "
        "sure your GitHub account has access and set GITHUB_TOKEN/GH_TOKEN or "
        "run `gh auth login`."
    )


def download_archive(destination: Path, manifest: dict[str, Any]) -> None:
    token = _github_token()
    try:
        if token:
            url = _release_asset_api_url(manifest, token)
            request = _request(url, token=token, accept="application/octet-stream")
        else:
            url = manifest.get(
                "asset_url",
                f"https://github.com/{manifest['repo']}/releases/download/{manifest['tag']}/{manifest['asset_name']}",
            )
            request = _request(url, accept="application/octet-stream")
        with urllib.request.urlopen(request, timeout=300) as resp, destination.open("wb") as out:
            shutil.copyfileobj(resp, out)
    except urllib.error.HTTPError as exc:
        raise DataCacheError(f"{_download_help(manifest)} HTTP {exc.code}: {exc.reason}") from exc
    except urllib.error.URLError as exc:
        raise DataCacheError(f"{_download_help(manifest)} {exc.reason}") from exc
    except OSError as exc:
        raise DataCacheError(f"{_download_help(manifest)} {exc}") from exc


def _safe_member_name(member_name: str, expected: set[str]) -> str | None:
    """Return the target-relative path (preserves subdir) or None to skip.

    Subdir support is required because the ambush dataset uses nested paths
    like `data_cache/ambush/klines/<BASE>.csv`. Behaviour:

    - Root entries (`.`, `data_cache`) and intermediate directory entries
      that are prefixes of expected file paths (e.g. `data_cache/ambush/`):
      return None — caller will recognise these aren't files and just create
      the dir.
    - File member that matches a manifest path: return the path relative to
      `data_cache/` so caller can join under target_dir.
    - Anything else (path traversal, absolute path, unlisted file): raise.
    """
    normalized = posixpath.normpath(member_name.replace("\\", "/"))
    if normalized.startswith("/") or normalized == ".." or normalized.startswith("../"):
        raise DataCacheError(f"unsafe archive member path: {member_name}")
    if normalized in (".", "data_cache"):
        return None
    if not normalized.startswith("data_cache/"):
        raise DataCacheError(f"unsafe archive member path: {member_name}")
    if normalized in expected:
        # File entry that matches the manifest.
        return str(PurePosixPath(*PurePosixPath(normalized).parts[1:]))
    # Not a manifest file — allow only if it's an intermediate dir prefix of
    # some expected entry (extract_archive will mkdir + skip non-files).
    prefix = normalized + "/"
    if any(item.startswith(prefix) for item in expected):
        return str(PurePosixPath(*PurePosixPath(normalized).parts[1:]))
    raise DataCacheError(f"unexpected archive member: {member_name}")


def extract_archive(archive_path: Path, target_dir: Path, manifest: dict[str, Any]) -> None:
    expected = {item["path"] for item in manifest["files"]}
    target_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive_path, "r:gz") as tf:
        for member in tf.getmembers():
            out_name = _safe_member_name(member.name, expected)
            if out_name is None:
                continue
            if not member.isfile():
                # Could be an intermediate directory entry (e.g. ambush/klines/).
                # Create it eagerly so file extraction below succeeds.
                (target_dir / out_name).mkdir(parents=True, exist_ok=True)
                continue
            source = tf.extractfile(member)
            if source is None:
                raise DataCacheError(f"failed to extract archive member: {member.name}")
            out_path = target_dir / out_name
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with source, out_path.open("wb") as out:
                shutil.copyfileobj(source, out)
    validate_data_cache(target_dir, manifest)


class _DirectoryLock:
    def __init__(self, path: Path):
        self.path = path
        self.fd: int | None = None

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        start = time.time()
        while True:
            try:
                self.fd = os.open(str(self.path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.write(self.fd, str(os.getpid()).encode("ascii"))
                return self
            except FileExistsError:
                if time.time() - start > LOCK_TIMEOUT_SECONDS:
                    try:
                        self.path.unlink()
                    except FileNotFoundError:
                        pass
                time.sleep(0.1)

    def __exit__(self, exc_type, exc, tb):
        if self.fd is not None:
            os.close(self.fd)
        try:
            self.path.unlink()
        except FileNotFoundError:
            pass


def ensure_data_cache() -> Path:
    manifest = load_manifest()
    data_dir = cache_dir_for_manifest(manifest)
    lock_path = data_dir.parent / ".hydrate.lock"
    with _DirectoryLock(lock_path):
        if data_dir.is_dir():
            try:
                validate_data_cache(data_dir, manifest)
                return data_dir
            except DataCacheError:
                shutil.rmtree(data_dir, ignore_errors=True)

        parent = data_dir.parent
        parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=".hydrate-", dir=str(parent)) as tmp:
            tmp_path = Path(tmp)
            archive_path = tmp_path / manifest["asset_name"]
            staged_dir = tmp_path / "data_cache"
            download_archive(archive_path, manifest)
            if _sha256_file(archive_path) != manifest["archive_sha256"]:
                raise DataCacheError("data cache archive checksum mismatch")
            extract_archive(archive_path, staged_dir, manifest)
            os.replace(staged_dir, data_dir)
    return data_dir


def _local_data_root_has_content() -> bool:
    """True iff <skill>/scripts/data_cache/ holds at least one expected file.

    Used by `resolve_data_root()` to detect a dev-mode pre-populated cache
    (where `.gitignore` keeps the dir empty in fresh checkouts).
    """
    if not LOCAL_DATA_DIR.is_dir():
        return False
    # Cheap heuristics: any top-level CSV (majors) or the ambush/ subdir.
    if any(LOCAL_DATA_DIR.glob("*.csv")):
        return True
    if (LOCAL_DATA_DIR / "ambush").is_dir():
        return True
    return False


def resolve_data_root() -> Path:
    """Return the active data_cache root, hydrating on first call if needed.

    Resolution precedence:

    1. ``$MOSS_TRADE_BOT_DATA_DIR`` — explicit dev override (any path).
    2. ``<skill>/scripts/data_cache/`` — used as-is when it already contains
       data (dev workflow: `git clone` + manually drop CSVs in).
    3. ``ensure_data_cache()`` — downloads + verifies + extracts the GitHub
       Release asset into ``~/.cache/moss-trade-bot-factory/v<version>/data_cache/``.
       This is the production code path for fresh user checkouts where the
       in-repo `scripts/data_cache/` is `.gitignore`d empty.

    Callers should treat the return value as **read-only**.
    """
    override = os.environ.get("MOSS_TRADE_BOT_DATA_DIR")
    if override:
        return Path(override).expanduser()
    if _local_data_root_has_content():
        return LOCAL_DATA_DIR
    return ensure_data_cache()


def _manifest_relative_paths(manifest: dict[str, Any]) -> list[PurePosixPath]:
    """Return each manifest entry's `path` stripped of the leading `data_cache/`."""
    out: list[PurePosixPath] = []
    for item in manifest["files"]:
        entry = PurePosixPath(item["path"])
        if entry.parts and entry.parts[0] == "data_cache":
            entry = PurePosixPath(*entry.parts[1:])
        out.append(entry)
    return out


def ensure_data_file(path: str | os.PathLike[str]) -> str:
    """Resolve a single data file by its in-cache path or its basename.

    Supports nested layouts (e.g. `ambush/klines/SAGA.csv`): the caller may
    pass either the full relative path (with or without the leading
    `data_cache/`) or a bare basename. The bare-basename form is kept for
    back-compat with the original flat layout (majors only); for ambush
    files a full path is preferred since multiple subtrees may share names.
    """
    candidate = Path(path)
    if candidate.is_file():
        return str(candidate)

    manifest = load_manifest()
    entries = _manifest_relative_paths(manifest)

    # Normalise the candidate: strip a leading "data_cache/" if present.
    parts = candidate.parts
    if parts and parts[0] == "data_cache":
        parts = parts[1:]
    candidate_rel = PurePosixPath(*parts) if parts else PurePosixPath(candidate.name)

    # Prefer exact relative-path match (handles nested correctly).
    match = next((e for e in entries if str(e) == str(candidate_rel)), None)
    # Fall back to basename match for the flat-layout call sites.
    if match is None:
        match = next((e for e in entries if e.name == candidate.name), None)
    if match is None:
        raise FileNotFoundError(f"CSV file not found: {candidate}")

    data_dir = ensure_data_cache()
    cached = data_dir / Path(*match.parts)
    if not cached.is_file():
        raise FileNotFoundError(f"CSV file not found after data cache hydrate: {cached}")
    return str(cached)
