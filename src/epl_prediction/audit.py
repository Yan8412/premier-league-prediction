"""Data-quality audit of the Sportmonks SQLite DB (``epl audit``).

Errors (exit 1): fixtures from another league, team names that don't map to
a known football-data name, seasons whose team set disagrees with the
football-data E0 CSV of the same season. Warnings: odd team / round counts,
duplicate pairings.

Note: promoted clubs are legitimate. For 2026/27 football-data's E0 file
lists Coventry, Hull and Ipswich in the Premier League, so seeing them in the
next round is expected, not a league-filter leak.
"""

from __future__ import annotations

import re
import sqlite3
from collections import Counter

from .config import EPL_LEAGUE_ID, FOOTBALL_DATA_DIR
from .data.football_data import load_all, load_season_csv
from .data.teams import canonical_team


def _season_code(name: str | None) -> str | None:
    m = re.match(r"^(\d{4})\s*/\s*(\d{4})$", (name or "").strip())
    return f"{m.group(1)[2:]}{m.group(2)[2:]}" if m else None


def audit_db(conn: sqlite3.Connection, league_id: int = EPL_LEAGUE_ID) -> dict:
    errors: list[str] = []
    warnings: list[str] = []
    known = set(load_all()[["home", "away"]].stack().unique())

    other = conn.execute(
        "SELECT id, league_id, name FROM fixtures WHERE league_id IS NULL OR league_id != ? LIMIT 20", (league_id,)
    ).fetchall()
    n_other = conn.execute("SELECT COUNT(*) FROM fixtures WHERE league_id IS NULL OR league_id != ?",
                           (league_id,)).fetchone()[0]
    if n_other:
        errors.append(f"{n_other} fixtures not in league {league_id}: "
                      + ", ".join(f"{r['id']}({r['league_id']}) {r['name']}" for r in other))

    names = {r[0] for r in conn.execute(
        "SELECT home_team FROM fixtures WHERE league_id=? UNION SELECT away_team FROM fixtures WHERE league_id=?",
        (league_id, league_id)) if r[0]}
    mapping = {n: canonical_team(n) for n in sorted(names)}
    unmapped = sorted(n for n, c in mapping.items() if c not in known)
    if unmapped:
        errors.append("team names not in team_names.csv / football-data: " + ", ".join(unmapped)
                      + " (add an alias row to src/epl_prediction/team_names.csv)")

    seasons = []
    for s in conn.execute("SELECT id, name FROM seasons WHERE league_id=? ORDER BY starting_at", (league_id,)):
        teams = {canonical_team(r[0]) for r in conn.execute(
            "SELECT home_team FROM fixtures WHERE season_id=? AND league_id=? "
            "UNION SELECT away_team FROM fixtures WHERE season_id=? AND league_id=?",
            (s["id"], league_id, s["id"], league_id)) if r[0]}
        if not teams:
            continue
        info = {"season_id": s["id"], "name": s["name"], "n_teams": len(teams), "teams": sorted(teams)}
        if len(teams) != 20:
            warnings.append(f"season {s['name']}: {len(teams)} teams (expected 20)")
        code = _season_code(s["name"])
        csv = FOOTBALL_DATA_DIR / f"E0_{code}.csv" if code else None
        if csv is not None and csv.exists():
            fd = load_season_csv(csv)
            fd_teams = set(fd["home"]) | set(fd["away"])
            info["not_in_football_data"] = sorted(teams - fd_teams)
            info["missing_vs_football_data"] = sorted(fd_teams - teams)
            if info["not_in_football_data"]:
                errors.append(f"season {s['name']}: teams not in football-data E0 {code}: "
                              + ", ".join(info["not_in_football_data"]))
        seasons.append(info)

    rounds = conn.execute(
        "SELECT round_id, round_name, COUNT(*) AS n FROM fixtures WHERE league_id=? AND round_id IS NOT NULL "
        "GROUP BY round_id HAVING n > 10", (league_id,)).fetchall()
    for r in rounds:
        warnings.append(f"round {r['round_name']} ({r['round_id']}) has {r['n']} fixtures (>10)")

    pairs = Counter((r["season_id"], r["home_team_id"], r["away_team_id"]) for r in conn.execute(
        "SELECT season_id, home_team_id, away_team_id FROM fixtures WHERE league_id=?", (league_id,)))
    dups = [k for k, v in pairs.items() if v > 1]
    if dups:
        warnings.append(f"{len(dups)} duplicate (season, home, away) pairings")

    return {"ok": not errors, "errors": errors, "warnings": warnings,
            "team_name_mapping": mapping, "seasons": seasons}
