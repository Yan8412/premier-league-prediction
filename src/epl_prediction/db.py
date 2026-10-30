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
    return conn


def upsert(conn: sqlite3.Connection, table: str, rows: Iterable[dict[str, Any]], key: str = "id") -> int:
    rows = list(rows)
    if not rows:
        return 0
    cols = list(rows[0].keys())
    updates = ", ".join(f"{c}=excluded.{c}" for c in cols if c != key)
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


def log_sync(conn: sqlite3.Connection, command: str, detail: dict) -> None:
    conn.execute("INSERT INTO sync_log (run_at, command, detail_json) VALUES (?, ?, ?)",
                 (utcnow_iso(), command, json.dumps(detail, default=str)))
