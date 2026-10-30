"""Flatten Sportmonks v3 payloads into rows for SQLite."""

from __future__ import annotations

from typing import Any

# Sportmonks fixture states treated as final (docs: "retain finished states 5, 7 and 8").
FINISHED_STATES = {5, 7, 8}
NOT_STARTED_STATE = 1


def _participant(fx: dict, location: str) -> dict:
    for p in fx.get("participants") or []:
        if (p.get("meta") or {}).get("location") == location:
            return p
    return {}


def _score(fx: dict, description: str, side: str) -> int | None:
    for s in fx.get("scores") or []:
        sc = s.get("score") or {}
        if s.get("description") == description and sc.get("participant") == side:
            goals = sc.get("goals")
            return int(goals) if goals is not None else None
    return None


def parse_fixture(fx: dict) -> dict[str, Any]:
    """``scores`` entries: description 1ST_HALF (HT score), 2ND_HALF (score at
    end of normal time, cumulative), CURRENT (latest/final)."""
    home, away = _participant(fx, "home"), _participant(fx, "away")
    ft_h = _score(fx, "CURRENT", "home")
    ft_a = _score(fx, "CURRENT", "away")
    if ft_h is None or ft_a is None:
        ft_h, ft_a = _score(fx, "2ND_HALF", "home"), _score(fx, "2ND_HALF", "away")
    rnd = fx.get("round") if isinstance(fx.get("round"), dict) else {}
    return {
        "id": int(fx["id"]),
        "league_id": fx.get("league_id"),
        "season_id": fx.get("season_id"),
        "round_id": fx.get("round_id"),
        "round_name": rnd.get("name"),
        "name": fx.get("name"),
        "starting_at": fx.get("starting_at"),
        "starting_at_timestamp": fx.get("starting_at_timestamp"),
        "state_id": fx.get("state_id"),
        "home_team_id": home.get("id"),
        "home_team": home.get("name"),
        "away_team_id": away.get("id"),
        "away_team": away.get("name"),
        "ft_home": ft_h,
        "ft_away": ft_a,
        "ht_home": _score(fx, "1ST_HALF", "home"),
        "ht_away": _score(fx, "1ST_HALF", "away"),
        "result_info": fx.get("result_info"),
    }


def parse_teams(fx: dict) -> list[dict[str, Any]]:
    return [{"id": p["id"], "name": p.get("name"), "short_code": p.get("short_code")}
            for p in fx.get("participants") or [] if p.get("id") is not None]


def _to_float(v: Any) -> float | None:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if f > 1.0 else None


def parse_odds(items: list[dict]) -> list[dict[str, Any]]:
    rows = []
    for o in items:
        value = _to_float(o.get("value"))
        if value is None:
            value = _to_float(o.get("dp3"))
        bookmaker = o.get("bookmaker") if isinstance(o.get("bookmaker"), dict) else {}
        rows.append({
            "odd_id": o.get("id"),
            "fixture_id": o.get("fixture_id"),
            "market_id": o.get("market_id"),
            "bookmaker_id": o.get("bookmaker_id"),
            "bookmaker_name": bookmaker.get("name"),
            "label": o.get("label"),
            "name": o.get("name"),
            "value": value,
            "stopped": int(bool(o.get("stopped"))),
            "latest_bookmaker_update": o.get("latest_bookmaker_update"),
        })
    return rows
