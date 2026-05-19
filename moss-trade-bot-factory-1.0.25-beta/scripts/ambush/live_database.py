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

            -- position_state: per-open-position tracking for close_monitor's
            -- trailing / max_hold rules. The server-side stop_loss watcher is
            -- the hard floor; this table is the skill's own drift state
            -- (peak ratchet + opened_at) that the server doesn't own. On
            -- restart, the monitor bootstraps missing rows from the live
            -- positions list so peak gets reset to current mark — never
            -- closes prematurely, just loses the historical peak.
            CREATE TABLE IF NOT EXISTS position_state (
                symbol         TEXT PRIMARY KEY,
                side           TEXT NOT NULL,
                entry_price    TEXT NOT NULL,
                opened_at      TEXT NOT NULL,
                peak_price     TEXT NOT NULL,
                last_seen_at   TEXT NOT NULL
            );

            -- exit_signal_processed: audit log of platform exit signals the
            -- skill has acted on. Keyed by the backend's ambush_exit_signal.id
            -- (a separate ID space from ambush_event.id) so the WS bootstrap
            -- replay + REST poller fallback can both dedupe via INSERT OR
            -- IGNORE. action ∈ {'closed', 'no_position', 'failed'}; reason
            -- carries the backend rule ('oi_revert' / 'max_hold') for audit.
            CREATE TABLE IF NOT EXISTS exit_signal_processed (
                exit_signal_id    INTEGER PRIMARY KEY,
                opening_event_id  INTEGER NOT NULL,
                hl_symbol         TEXT NOT NULL,
                reason            TEXT NOT NULL,
                received_at       TEXT NOT NULL,
                action            TEXT NOT NULL,
                order_id          TEXT,
                error_msg         TEXT
            );
            CREATE INDEX IF NOT EXISTS ix_exit_signal_symbol
              ON exit_signal_processed(hl_symbol, received_at);

            -- symbol_action_history: per-symbol open/close audit used by
            -- `same_coin_dedup_days` (7d window) + `cooldown_bars` (post-close
            -- N-bar cool-off) + `max_trades_per_event` gates. Records every
            -- open/close that this skill instance executed; queries look at
            -- the latest open/close per symbol to decide whether the next
            -- event should be acted on. event_id is the originating ambush
            -- event (NULL for closes triggered by exit_signal/close_monitor
            -- without a clear linkage).
            CREATE TABLE IF NOT EXISTS symbol_action_history (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                hl_symbol     TEXT NOT NULL,
                action        TEXT NOT NULL,    -- 'open' | 'close'
                happened_at   TEXT NOT NULL,    -- ISO-8601 UTC
                event_id      INTEGER,          -- ambush_event.id (NULL OK for close)
                order_id      TEXT              -- audit linkage to server order
            );
            CREATE INDEX IF NOT EXISTS ix_sym_action_symbol_time
              ON symbol_action_history(hl_symbol, action, happened_at DESC);
            CREATE INDEX IF NOT EXISTS ix_sym_action_event_id
              ON symbol_action_history(event_id);

            -- deferred_opens: crash-safe queue for long momentum_bars /
            -- short entry_delay_bars. A decision row with
            -- order_status='deferred_open' dedupes event processing; this
            -- table preserves the due time so a restarted live_runner can
            -- re-spawn the delayed open instead of losing it with the old
            -- daemon-thread-only implementation.
            CREATE TABLE IF NOT EXISTS deferred_opens (
                event_id          INTEGER PRIMARY KEY REFERENCES events(event_id),
                due_at            TEXT NOT NULL,
                delay_param_name  TEXT NOT NULL,
                delay_bars        INTEGER NOT NULL,
                status            TEXT NOT NULL DEFAULT 'pending',
                created_at        TEXT NOT NULL,
                updated_at        TEXT NOT NULL,
                last_error        TEXT
            );
            CREATE INDEX IF NOT EXISTS ix_deferred_opens_status_due
              ON deferred_opens(status, due_at);
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


def update_decision_outcome(
    db_path: str,
    event_id: int,
    *,
    order_id: str | None = None,
    order_status: str | None = None,
    error_msg: str | None = None,
) -> None:
    """Update the order outcome columns on an existing decision row.

    Used by the deferred-open path: the initial record_decision call writes
    `order_status='deferred_open'` for dedup, then this UPDATE fills in the
    real order_id / order_status after the deferred order actually fires.
    UPDATE only — no insert; if no row exists this is a no-op silently
    (caller is responsible for having INSERT'd first)."""
    with get_conn(db_path) as conn:
        conn.execute(
            "UPDATE decisions SET order_id = ?, order_status = ?, error_msg = ? "
            "WHERE event_id = ?",
            (order_id, order_status, error_msg, event_id),
        )


# ───── deferred opens ─────


def enqueue_deferred_open(
    db_path: str,
    *,
    event_id: int,
    due_at: str,
    delay_param_name: str,
    delay_bars: int,
) -> None:
    now = _now_iso()
    with get_conn(db_path) as conn:
        conn.execute(
            """
            INSERT INTO deferred_opens (
                event_id, due_at, delay_param_name, delay_bars,
                status, created_at, updated_at
            )
            VALUES (?, ?, ?, ?, 'pending', ?, ?)
            ON CONFLICT(event_id) DO UPDATE SET
                due_at=excluded.due_at,
                delay_param_name=excluded.delay_param_name,
                delay_bars=excluded.delay_bars,
                status='pending',
                updated_at=excluded.updated_at,
                last_error=NULL
            """,
            (int(event_id), due_at, delay_param_name, int(delay_bars), now, now),
        )


def mark_deferred_open_running(db_path: str, event_id: int) -> None:
    with get_conn(db_path) as conn:
        conn.execute(
            """
            UPDATE deferred_opens
               SET status='running', updated_at=?
             WHERE event_id=?
            """,
            (_now_iso(), int(event_id)),
        )


def mark_deferred_open_done(
    db_path: str,
    event_id: int,
    *,
    status: str,
    error_msg: str | None = None,
) -> None:
    with get_conn(db_path) as conn:
        conn.execute(
            """
            UPDATE deferred_opens
               SET status=?, updated_at=?, last_error=?
             WHERE event_id=?
            """,
            (status, _now_iso(), error_msg, int(event_id)),
        )


def list_pending_deferred_opens(db_path: str, limit: int = 100) -> list[dict]:
    """Return pending/running deferred opens that still have a deferred decision.

    `running` is included intentionally: after a process crash while the
    daemon thread was sleeping, the row may be left running forever. Startup
    recovery should treat it as resumable.
    """
    with get_conn(db_path) as conn:
        rows = conn.execute(
            """
            SELECT q.event_id, q.due_at, q.delay_param_name, q.delay_bars,
                   e.raw_payload, d.decision, d.reason
              FROM deferred_opens q
              JOIN events e ON e.event_id = q.event_id
              JOIN decisions d ON d.event_id = q.event_id
             WHERE q.status IN ('pending', 'running')
               AND d.order_status = 'deferred_open'
          ORDER BY q.due_at ASC, q.event_id ASC
             LIMIT ?
            """,
            (int(limit),),
        ).fetchall()
    return [
        {
            "event_id": int(r[0]),
            "due_at": r[1],
            "delay_param_name": r[2],
            "delay_bars": int(r[3]),
            "event": json.loads(r[4]),
            "decision": r[5],
            "reason": r[6],
        }
        for r in rows
    ]


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


def get_last_exit_signal_id_seen(db_path: str) -> int:
    """Cursor for the platform exit-signal stream. Separate from
    last_event_id_seen because ambush_event and ambush_exit_signal use
    distinct ID sequences on the backend."""
    raw = get_state(db_path, "last_exit_signal_id_seen", "0")
    try:
        return int(raw)
    except ValueError:
        return 0


def set_last_exit_signal_id_seen(db_path: str, exit_signal_id: int) -> None:
    set_state(db_path, "last_exit_signal_id_seen", str(exit_signal_id))


# ───── exit_signal_processed ─────


def is_exit_signal_processed(db_path: str, exit_signal_id: int) -> bool:
    """True when we've already acted on this exit_signal_id. Skill uses
    this to short-circuit WS bootstrap replay (server may re-deliver the
    same signal after a disconnect)."""
    with get_conn(db_path) as conn:
        row = conn.execute(
            "SELECT 1 FROM exit_signal_processed WHERE exit_signal_id = ? LIMIT 1",
            (exit_signal_id,),
        ).fetchone()
        return row is not None


def record_exit_signal_processed(
    db_path: str,
    exit_signal_id: int,
    opening_event_id: int,
    hl_symbol: str,
    reason: str,
    action: str,
    order_id: str | None = None,
    error_msg: str | None = None,
) -> None:
    """Idempotent (INSERT OR IGNORE) — a second call for the same
    exit_signal_id is a no-op. action ∈ {'closed','no_position','failed'}."""
    with get_conn(db_path) as conn:
        conn.execute(
            "INSERT OR IGNORE INTO exit_signal_processed "
            "(exit_signal_id, opening_event_id, hl_symbol, reason, received_at, "
            " action, order_id, error_msg) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                exit_signal_id, opening_event_id, hl_symbol, reason,
                _now_iso(), action, order_id, error_msg,
            ),
        )


# ───── position_state (close_monitor uses these) ─────


def upsert_position_state(
    db_path: str,
    symbol: str,
    side: str,
    entry_price: str,
    opened_at: str,
    peak_price: str,
) -> None:
    """Insert a new active position. ON CONFLICT keeps the original
    opened_at + entry_price (a re-open after partial flatten + reopen
    on same symbol should be rare given single_position_lock; if it
    happens, treat as continuation of the original position)."""
    with get_conn(db_path) as conn:
        conn.execute(
            "INSERT INTO position_state "
            "(symbol, side, entry_price, opened_at, peak_price, last_seen_at) "
            "VALUES (?, ?, ?, ?, ?, ?) "
            "ON CONFLICT(symbol) DO UPDATE SET "
            "  last_seen_at=excluded.last_seen_at",
            (symbol, side, entry_price, opened_at, peak_price, _now_iso()),
        )


def update_position_peak(db_path: str, symbol: str, peak_price: str) -> None:
    with get_conn(db_path) as conn:
        conn.execute(
            "UPDATE position_state SET peak_price = ?, last_seen_at = ? WHERE symbol = ?",
            (peak_price, _now_iso(), symbol),
        )


def get_position_state(db_path: str, symbol: str) -> dict | None:
    with get_conn(db_path) as conn:
        row = conn.execute(
            "SELECT symbol, side, entry_price, opened_at, peak_price, last_seen_at "
            "FROM position_state WHERE symbol = ?", (symbol,)
        ).fetchone()
        if not row:
            return None
        return {
            "symbol": row[0], "side": row[1], "entry_price": row[2],
            "opened_at": row[3], "peak_price": row[4], "last_seen_at": row[5],
        }


def list_position_states(db_path: str) -> list[dict]:
    with get_conn(db_path) as conn:
        rows = conn.execute(
            "SELECT symbol, side, entry_price, opened_at, peak_price, last_seen_at "
            "FROM position_state"
        ).fetchall()
        return [
            {
                "symbol": r[0], "side": r[1], "entry_price": r[2],
                "opened_at": r[3], "peak_price": r[4], "last_seen_at": r[5],
            }
            for r in rows
        ]


def delete_position_state(db_path: str, symbol: str) -> None:
    with get_conn(db_path) as conn:
        conn.execute("DELETE FROM position_state WHERE symbol = ?", (symbol,))


# ───── symbol_action_history (dedup / cooldown / per-event trade count) ─────


def record_symbol_action(
    db_path: str,
    hl_symbol: str,
    action: str,
    *,
    event_id: int | None = None,
    order_id: str | None = None,
) -> None:
    """Append an open/close audit row. Called by event_handler after a
    successful order placement (open) and by close_monitor/process_exit_signal
    after a successful close. action ∈ {'open', 'close'}."""
    if action not in ("open", "close"):
        raise ValueError(f"invalid action: {action!r}")
    with get_conn(db_path) as conn:
        conn.execute(
            "INSERT INTO symbol_action_history "
            "(hl_symbol, action, happened_at, event_id, order_id) "
            "VALUES (?, ?, ?, ?, ?)",
            (hl_symbol.upper(), action, _now_iso(), event_id, order_id),
        )


def last_action_at(
    db_path: str, hl_symbol: str, action: str,
) -> datetime | None:
    """Return the most recent timestamp the given action happened on the
    symbol, or None if never. action ∈ {'open', 'close'}."""
    if action not in ("open", "close"):
        raise ValueError(f"invalid action: {action!r}")
    with get_conn(db_path) as conn:
        row = conn.execute(
            "SELECT happened_at FROM symbol_action_history "
            "WHERE hl_symbol = ? AND action = ? "
            "ORDER BY happened_at DESC LIMIT 1",
            (hl_symbol.upper(), action),
        ).fetchone()
    if not row or not row[0]:
        return None
    try:
        return datetime.fromisoformat(row[0])
    except (TypeError, ValueError):
        return None


def count_event_trades(db_path: str, event_id: int) -> int:
    """Number of 'open' actions recorded against this event_id. Used by
    `max_trades_per_event` gate. Returns 0 if event_id is None/0."""
    if not event_id:
        return 0
    with get_conn(db_path) as conn:
        row = conn.execute(
            "SELECT COUNT(*) FROM symbol_action_history "
            "WHERE event_id = ? AND action = 'open'",
            (event_id,),
        ).fetchone()
    return int(row[0]) if row else 0


def list_symbol_action_history(
    db_path: str,
    *,
    hl_symbol: str | None = None,
    action: str | None = None,
    limit: int = 100,
) -> list[dict]:
    """Return recent local open/close audit rows for operational inspection."""
    if action is not None and action not in ("open", "close"):
        raise ValueError(f"invalid action: {action!r}")
    limit = max(1, min(int(limit or 100), 500))

    where: list[str] = []
    args: list[Any] = []
    if hl_symbol:
        where.append("hl_symbol = ?")
        args.append(str(hl_symbol).upper())
    if action:
        where.append("action = ?")
        args.append(action)
    where_sql = ""
    if where:
        where_sql = "WHERE " + " AND ".join(where)
    args.append(limit)

    with get_conn(db_path) as conn:
        rows = conn.execute(
            f"""
            SELECT id, hl_symbol, action, happened_at, event_id, order_id
              FROM symbol_action_history
              {where_sql}
          ORDER BY happened_at DESC, id DESC
             LIMIT ?
            """,
            tuple(args),
        ).fetchall()
    return [
        {
            "id": int(r[0]),
            "hl_symbol": r[1],
            "action": r[2],
            "happened_at": r[3],
            "event_id": int(r[4]) if r[4] is not None else None,
            "order_id": r[5],
        }
        for r in rows
    ]
