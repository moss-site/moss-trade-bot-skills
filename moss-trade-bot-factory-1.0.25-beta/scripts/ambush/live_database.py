"""
Ambush live runner 本地状态库 (SQLite)。

镜像 hyperliquid-copy-trade `follow_service/database.py` 模式：
- 三张表：events / decisions / ambush_state
- events 表 UNIQUE(event_id) 强制去重 — WS + poller 双通道任一先收都不会重复处理
- decisions 表记录每个事件的判决 + 下单结果（审计 + 重启幂等性兜底）
- ambush_state 是 key/value 游标存储 — 主要存 last_event_id_seen

Server 端**完全不感知**这个本地状态：游标 / 决策记录 / 去重全在 skill 端。
"""

from __future__ import annotations

import contextlib
import json
import sqlite3
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


_lock = threading.RLock()
_conn_cache: dict[str, sqlite3.Connection] = {}


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _get_conn(db_path: str) -> sqlite3.Connection:
    """Per-path connection cache. SQLite 单连接 + threading.RLock 串行化即可。"""
    with _lock:
        conn = _conn_cache.get(db_path)
        if conn is not None:
            return conn
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(db_path, isolation_level=None, check_same_thread=False)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA foreign_keys=ON")
        _conn_cache[db_path] = conn
        return conn


@contextlib.contextmanager
def get_conn(db_path: str):
    """Context-managed handle; auto-locks for the duration."""
    with _lock:
        yield _get_conn(db_path)


def init_db(db_path: str) -> None:
    """Idempotent schema creation."""
    with get_conn(db_path) as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS events (
                event_id     INTEGER PRIMARY KEY,
                received_at  TEXT NOT NULL,
                hl_symbol    TEXT NOT NULL,
                trigger_ts   TEXT NOT NULL,
                raw_payload  TEXT NOT NULL,
                sync_status  TEXT NOT NULL DEFAULT 'pending'
            );
            CREATE INDEX IF NOT EXISTS ix_events_status_id
              ON events(sync_status, event_id);

            CREATE TABLE IF NOT EXISTS decisions (
                event_id       INTEGER PRIMARY KEY REFERENCES events(event_id),
                decided_at     TEXT NOT NULL,
                decision       TEXT NOT NULL,  -- 'long' / 'short' / 'skip'
                reason         TEXT NOT NULL,  -- rule id or skip reason
                order_id       TEXT,            -- source_order_id if placed
                order_status   TEXT,            -- 'placed' / 'rejected' / 'failed'
                error_msg      TEXT
            );

            CREATE TABLE IF NOT EXISTS ambush_state (
                key         TEXT PRIMARY KEY,
                value       TEXT NOT NULL,
                updated_at  TEXT NOT NULL
            );
            """
        )


# ───── events ─────


def upsert_event(db_path: str, event: dict) -> bool:
    """Insert one event row keyed on event_id. Returns True if newly inserted,
    False if it was already present (skill should skip duplicate decision)."""
    event_id = int(event["event_id"])
    with get_conn(db_path) as conn:
        cur = conn.execute(
            "INSERT OR IGNORE INTO events (event_id, received_at, hl_symbol, trigger_ts, raw_payload, sync_status) "
            "VALUES (?, ?, ?, ?, ?, 'pending')",
            (
                event_id,
                _now_iso(),
                event.get("hl_symbol", ""),
                event.get("trigger_ts", ""),
                json.dumps(event, ensure_ascii=False),
            ),
        )
        return cur.rowcount == 1


def mark_event_synced(db_path: str, event_id: int) -> None:
    with get_conn(db_path) as conn:
        conn.execute(
            "UPDATE events SET sync_status='synced' WHERE event_id = ?", (event_id,)
        )


def is_event_processed(db_path: str, event_id: int) -> bool:
    """Cheap dedup check: already in events AND has a decision row."""
    with get_conn(db_path) as conn:
        row = conn.execute(
            "SELECT 1 FROM decisions WHERE event_id = ? LIMIT 1", (event_id,)
        ).fetchone()
        return row is not None


def list_pending_events(db_path: str, limit: int = 100) -> list[dict]:
    """Return events that have no decision row yet (skill startup recovery)."""
    with get_conn(db_path) as conn:
        rows = conn.execute(
            """
            SELECT e.event_id, e.raw_payload
              FROM events e
         LEFT JOIN decisions d ON d.event_id = e.event_id
             WHERE d.event_id IS NULL
          ORDER BY e.event_id ASC
             LIMIT ?
            """,
            (limit,),
        ).fetchall()
        return [json.loads(row[1]) for row in rows]


# ───── decisions ─────


def record_decision(
    db_path: str,
    event_id: int,
    decision: str,
    reason: str,
    order_id: str | None = None,
    order_status: str | None = None,
    error_msg: str | None = None,
) -> None:
    """Idempotent: writing twice for the same event_id is a no-op (OR IGNORE)."""
    with get_conn(db_path) as conn:
        conn.execute(
            "INSERT OR IGNORE INTO decisions "
            "(event_id, decided_at, decision, reason, order_id, order_status, error_msg) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (event_id, _now_iso(), decision, reason, order_id, order_status, error_msg),
        )


# ───── state (cursor / misc) ─────


def get_state(db_path: str, key: str, default: str = "") -> str:
    with get_conn(db_path) as conn:
        row = conn.execute(
            "SELECT value FROM ambush_state WHERE key = ?", (key,)
        ).fetchone()
        return row[0] if row else default


def set_state(db_path: str, key: str, value: str) -> None:
    with get_conn(db_path) as conn:
        conn.execute(
            "INSERT INTO ambush_state (key, value, updated_at) VALUES (?, ?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value, updated_at=excluded.updated_at",
            (key, value, _now_iso()),
        )


def get_last_event_id_seen(db_path: str) -> int:
    raw = get_state(db_path, "last_event_id_seen", "0")
    try:
        return int(raw)
    except ValueError:
        return 0


def set_last_event_id_seen(db_path: str, event_id: int) -> None:
    set_state(db_path, "last_event_id_seen", str(event_id))
