"""Odds snapshot scheduler: run ``epl sync due`` every ~15 minutes (cron/systemd).

Each run:
  1. refreshes upcoming EPL fixtures (kickoff times move) if the last refresh
     is older than ``fixture_refresh_hours``;
  2. fetches market 1 + 29 odds only for fixtures that are *due*:
       opening  no snapshot stored yet (fixture within ``horizon_days``)
       T-6h     kickoff in 5-7 h and no T-6h snapshot yet
       T-1h     kickoff in 0.25-1.75 h and no T-1h snapshot yet
       interim  last snapshot older than ``refresh_hours`` (kickoff > 7 h away)
Request cost per run = due fixtures x markets, well inside the hourly
per-entity limit for a 10-match round.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone

from ..config import EPL_LEAGUE_ID, MARKET_FT_1X2, MARKET_HT_FT
from ..db import log_sync
from .client import SportmonksClient
from .parse import NOT_STARTED_STATE
from .sync import SNAPSHOT_WINDOWS, sync_fixtures, sync_odds


@dataclass
class DueSnapshot:
    fixture_id: int
    reason: str
    hours_to_kickoff: float


def _parse_ts(value: str) -> datetime:
    dt = datetime.fromisoformat(value)
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def due_snapshots(conn: sqlite3.Connection, now: datetime, horizon_days: int = 14, refresh_hours: float = 24.0,
                  league_id: int = EPL_LEAGUE_ID) -> list[DueSnapshot]:
    lo, hi = now.timestamp(), (now + timedelta(days=horizon_days)).timestamp()
    rows = conn.execute(
        "SELECT id, starting_at_timestamp FROM fixtures WHERE league_id=? AND starting_at_timestamp BETWEEN ? AND ? "
        "AND (state_id IS NULL OR state_id=?) ORDER BY starting_at_timestamp",
        (league_id, int(lo), int(hi), NOT_STARTED_STATE)).fetchall()
    due = []
    for r in rows:
        hours = (r["starting_at_timestamp"] - now.timestamp()) / 3600
        snaps = conn.execute(
            "SELECT snapshot_tag, MAX(fetched_at) AS last FROM odds_snapshots WHERE fixture_id=? AND market_id=? "
            "GROUP BY snapshot_tag", (r["id"], MARKET_FT_1X2)).fetchall()
        tags = {s["snapshot_tag"] for s in snaps}
        last = max((_parse_ts(s["last"]) for s in snaps), default=None)
        reason = None
        if not snaps:
            reason = "opening"
        else:
            for tag, (w_lo, w_hi) in SNAPSHOT_WINDOWS.items():
                if w_lo <= hours <= w_hi and tag not in tags:
                    reason = tag
                    break
            if reason is None and hours > max(w for _, w in SNAPSHOT_WINDOWS.values()) and last is not None \
                    and (now - last) >= timedelta(hours=refresh_hours):
                reason = "interim"
        if reason:
            due.append(DueSnapshot(r["id"], reason, round(hours, 2)))
    return due


def last_sync(conn: sqlite3.Connection, command: str) -> datetime | None:
    row = conn.execute("SELECT MAX(run_at) AS t FROM sync_log WHERE command=?", (command,)).fetchone()
    return _parse_ts(row["t"]) if row and row["t"] else None


def run_due(client: SportmonksClient, conn: sqlite3.Connection, now: datetime | None = None, horizon_days: int = 14,
            refresh_hours: float = 24.0, fixture_refresh_hours: float = 6.0,
            markets: tuple[int, ...] = (MARKET_FT_1X2, MARKET_HT_FT)) -> dict:
    now = now or datetime.now(timezone.utc)
    out: dict = {"now": now.isoformat(timespec="seconds"), "fixtures_refreshed": None, "due": [], "odds": None}
    last = last_sync(conn, "scheduler_fixture_refresh")
    if last is None or now - last >= timedelta(hours=fixture_refresh_hours):
        today = now.date()
        out["fixtures_refreshed"] = sync_fixtures(client, conn, today, today + timedelta(days=horizon_days))
        log_sync(conn, "scheduler_fixture_refresh", {"n": out["fixtures_refreshed"]}, run_at=now.isoformat())
        conn.commit()
    due = due_snapshots(conn, now, horizon_days, refresh_hours)
    out["due"] = [d.__dict__ for d in due]
    if due:
        out["odds"] = sync_odds(client, conn, [d.fixture_id for d in due], markets, now=now)
    return out


def coverage(conn: sqlite3.Connection, since: date | None = None) -> list[dict]:
    """Per fixture: which snapshot tags exist for markets 1 and 29 (monitor sparse HT/FT)."""
    since_ts = int(datetime.combine(since or date.today() - timedelta(days=14), datetime.min.time(),
                                    tzinfo=timezone.utc).timestamp())
    rows = conn.execute(
        "SELECT f.id, f.home_team, f.away_team, f.starting_at, o.market_id, GROUP_CONCAT(DISTINCT o.snapshot_tag) AS tags "
        "FROM fixtures f LEFT JOIN odds_snapshots o ON o.fixture_id = f.id WHERE f.starting_at_timestamp >= ? "
        "GROUP BY f.id, o.market_id ORDER BY f.starting_at_timestamp", (since_ts,)).fetchall()
    by_fixture: dict[int, dict] = {}
    for r in rows:
        d = by_fixture.setdefault(r["id"], {"fixture_id": r["id"], "match": f"{r['home_team']} v {r['away_team']}",
                                            "kickoff": r["starting_at"], "market_1": [], "market_29": []})
        if r["market_id"] in (MARKET_FT_1X2, MARKET_HT_FT):
            d[f"market_{r['market_id']}"] = sorted((r["tags"] or "").split(","))
    return list(by_fixture.values())
