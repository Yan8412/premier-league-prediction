"""Sync Sportmonks data into SQLite.

Odds snapshot tags (computed at fetch time from kickoff):
  opening       first time we ever stored this fixture+market
  T-6h / T-1h   fetched within ~6h / ~1h of kickoff
  interim       any other pre-kickoff fetch
  post_kickoff  fetched after kickoff (kept for audit, ignored by predict)
Schedule ``epl sync odds`` (cron) e.g. daily + at T-6h and T-1h.
"""

from __future__ import annotations

import logging
import sqlite3
from datetime import date, datetime, timedelta, timezone
from typing import Iterable

from ..config import EPL_LEAGUE_ID, MARKET_FT_1X2, MARKET_HT_FT
from ..db import insert, log_sync, upsert, utcnow_iso
from .client import SportmonksClient, SportmonksError
from .parse import NOT_STARTED_STATE, parse_fixture, parse_odds, parse_teams

log = logging.getLogger(__name__)


def snapshot_tag(hours_to_kickoff: float | None, is_first: bool) -> str:
    if hours_to_kickoff is not None and hours_to_kickoff < 0:
        return "post_kickoff"
    if is_first:
        return "opening"
    if hours_to_kickoff is None:
        return "interim"
    if hours_to_kickoff <= 1.5:
        return "T-1h"
    if hours_to_kickoff <= 6.5:
        return "T-6h"
    return "interim"


def sync_seasons(client: SportmonksClient, conn: sqlite3.Connection, league_id: int = EPL_LEAGUE_ID) -> list[dict]:
    league = client.league(league_id)
    seasons = list(league.get("seasons") or [])
    current = league.get("currentseason") or league.get("currentSeason")
    if current and all(s.get("id") != current.get("id") for s in seasons):
        seasons.append(current)
    now = utcnow_iso()
    rows = [{
        "id": s["id"], "league_id": s.get("league_id", league_id), "name": s.get("name"),
        "starting_at": s.get("starting_at"), "ending_at": s.get("ending_at"),
        "is_current": int(bool(s.get("is_current"))), "finished": int(bool(s.get("finished"))),
        "synced_at": now,
    } for s in seasons]
    upsert(conn, "seasons", rows)
    log_sync(conn, "sync_seasons", {"league_id": league_id, "n": len(rows)})
    conn.commit()
    return rows


def current_season(conn: sqlite3.Connection, league_id: int = EPL_LEAGUE_ID) -> sqlite3.Row | None:
    return conn.execute(
        "SELECT * FROM seasons WHERE league_id=? ORDER BY is_current DESC, starting_at DESC LIMIT 1", (league_id,)
    ).fetchone()


def store_fixtures(conn: sqlite3.Connection, raw_fixtures: Iterable[dict]) -> int:
    now = utcnow_iso()
    fixtures, teams = [], {}
    for fx in raw_fixtures:
        row = parse_fixture(fx)
        row["synced_at"] = now
        fixtures.append(row)
        for t in parse_teams(fx):
            teams[t["id"]] = t
    upsert(conn, "teams", teams.values())
    return upsert(conn, "fixtures", fixtures)


def sync_fixtures(client: SportmonksClient, conn: sqlite3.Connection, start: date, end: date,
                  league_id: int = EPL_LEAGUE_ID) -> int:
    n = store_fixtures(conn, client.fixtures_between(start, end, league_id))
    log_sync(conn, "sync_fixtures", {"start": start, "end": end, "n": n})
    conn.commit()
    return n


def sync_season_fixtures(client: SportmonksClient, conn: sqlite3.Connection, season_id: int) -> int:
    row = conn.execute("SELECT starting_at, ending_at FROM seasons WHERE id=?", (season_id,)).fetchone()
    if row and row["starting_at"] and row["ending_at"]:
        start, end = date.fromisoformat(row["starting_at"][:10]), date.fromisoformat(row["ending_at"][:10])
    else:
        s = client.season(season_id)
        start, end = date.fromisoformat(s["starting_at"][:10]), date.fromisoformat(s["ending_at"][:10])
    return sync_fixtures(client, conn, start, end)


def sync_fixture(client: SportmonksClient, conn: sqlite3.Connection, fixture_id: int) -> dict:
    fx = client.fixture(fixture_id)
    store_fixtures(conn, [fx])
    conn.commit()
    return parse_fixture(fx)


def upcoming_fixture_ids(conn: sqlite3.Connection, days: int, now: datetime | None = None) -> list[int]:
    now = now or datetime.now(timezone.utc)
    lo, hi = int(now.timestamp()), int((now + timedelta(days=days)).timestamp())
    rows = conn.execute(
        "SELECT id FROM fixtures WHERE starting_at_timestamp BETWEEN ? AND ? AND (state_id IS NULL OR state_id=?) "
        "ORDER BY starting_at_timestamp", (lo, hi, NOT_STARTED_STATE)).fetchall()
    return [r["id"] for r in rows]


def sync_odds(client: SportmonksClient, conn: sqlite3.Connection, fixture_ids: Iterable[int],
              markets: Iterable[int] = (MARKET_FT_1X2, MARKET_HT_FT), now: datetime | None = None) -> dict:
    """Fetch current pre-match odds for each fixture/market and append a snapshot.
    Returns coverage stats; empty markets (e.g. sparse HT/FT) are reported, not fatal."""
    now = now or datetime.now(timezone.utc)
    fetched_at = now.isoformat(timespec="seconds")
    stats: dict = {"fixtures": 0, "rows": 0, "by_market": {}, "errors": []}
    for fid in fixture_ids:
        stats["fixtures"] += 1
        ko = conn.execute("SELECT starting_at_timestamp FROM fixtures WHERE id=?", (fid,)).fetchone()
        hours = (ko["starting_at_timestamp"] - now.timestamp()) / 3600 if ko and ko["starting_at_timestamp"] else None
        for mid in markets:
            m = stats["by_market"].setdefault(str(mid), {"fixtures_with_odds": 0, "fixtures_empty": 0, "rows": 0})
            try:
                items = client.prematch_odds(fid, mid)
            except SportmonksError as exc:
                stats["errors"].append({"fixture_id": fid, "market_id": mid, "status": exc.status, "error": str(exc)})
                if exc.status in (401, 403):
                    raise
                continue
            rows = [r for r in parse_odds(items) if r["value"] is not None]
            if not rows:
                m["fixtures_empty"] += 1
                continue
            is_first = conn.execute("SELECT 1 FROM odds_snapshots WHERE fixture_id=? AND market_id=? LIMIT 1",
                                    (fid, mid)).fetchone() is None
            tag = snapshot_tag(hours, is_first)
            books = {r["bookmaker_id"]: r["bookmaker_name"] for r in rows if r["bookmaker_id"] is not None}
            upsert(conn, "bookmakers", [{"id": b, "name": n} for b, n in books.items() if n])
            insert(conn, "odds_snapshots", [{
                "fetched_at": fetched_at, "snapshot_tag": tag, "hours_to_kickoff": hours,
                "fixture_id": fid, "market_id": mid, "bookmaker_id": r["bookmaker_id"], "odd_id": r["odd_id"],
                "label": r["label"], "name": r["name"], "value": r["value"], "stopped": r["stopped"],
                "latest_bookmaker_update": r["latest_bookmaker_update"],
            } for r in rows])
            m["fixtures_with_odds"] += 1
            m["rows"] += len(rows)
            stats["rows"] += len(rows)
    log_sync(conn, "sync_odds", stats)
    conn.commit()
    return stats
