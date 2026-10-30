"""SQLite helpers. The DB file is gitignored; the schema is committed."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from importlib import resources
from pathlib import Path
from typing import Any, Iterable

from .config import db_path


def utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def connect(path: Path | str | None = None) -> sqlite3.Connection:
    path = Path(path) if path is not None else db_path()
    if str(path) != ":memory:":
        path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    conn.executescript(resources.files("epl_prediction").joinpath("schema.sql").read_text(encoding="utf-8"))
    _migrate(conn)
    return conn


# Columns added after a table was first created (CREATE TABLE IF NOT EXISTS won't add them).
ADDED_COLUMNS = {"fixtures": {"xg_home": "REAL", "xg_away": "REAL"}}


def _migrate(conn: sqlite3.Connection) -> None:
    for table, cols in ADDED_COLUMNS.items():
        existing = {r[1] for r in conn.execute(f"PRAGMA table_info({table})")}
        for col, typ in cols.items():
            if col not in existing:
                conn.execute(f"ALTER TABLE {table} ADD COLUMN {col} {typ}")
    conn.commit()


def upsert(conn: sqlite3.Connection, table: str, rows: Iterable[dict[str, Any]], key: str = "id",
           keep_existing: Iterable[str] = ()) -> int:
    """Insert or update. Columns in ``keep_existing`` are only overwritten by non-NULL values."""
    rows = list(rows)
    if not rows:
        return 0
    cols = list(rows[0].keys())
    keep = set(keep_existing)
    updates = ", ".join(f"{c}=COALESCE(excluded.{c}, {c})" if c in keep else f"{c}=excluded.{c}"
                        for c in cols if c != key)
    sql = (f"INSERT INTO {table} ({', '.join(cols)}) VALUES ({', '.join('?' for _ in cols)}) "
           f"ON CONFLICT({key}) DO UPDATE SET {updates}")
    conn.executemany(sql, [tuple(r[c] for c in cols) for r in rows])
    return len(rows)


def insert(conn: sqlite3.Connection, table: str, rows: Iterable[dict[str, Any]]) -> int:
    rows = list(rows)
    if not rows:
        return 0
    cols = list(rows[0].keys())
    conn.executemany(f"INSERT INTO {table} ({', '.join(cols)}) VALUES ({', '.join('?' for _ in cols)})",
                     [tuple(r[c] for c in cols) for r in rows])
    return len(rows)


def log_sync(conn: sqlite3.Connection, command: str, detail: dict, run_at: str | None = None) -> None:
    conn.execute("INSERT INTO sync_log (run_at, command, detail_json) VALUES (?, ?, ?)",
                 (run_at or utcnow_iso(), command, json.dumps(detail, default=str)))
