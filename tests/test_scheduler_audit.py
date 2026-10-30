from datetime import date, datetime, timedelta, timezone

import pytest
import responses

from epl_prediction.audit import audit_db
from epl_prediction.db import connect, upsert
from epl_prediction.sportmonks import sync
from epl_prediction.sportmonks.client import SportmonksClient
from epl_prediction.sportmonks.scheduler import coverage, due_snapshots, run_due

from .conftest import load_fixture

BASE = "https://api.sportmonks.com/v3/football"
NOW = datetime(2030, 11, 1, 6, 0, tzinfo=timezone.utc)
KICKOFF = int(datetime(2030, 11, 1, 17, 30, tzinfo=timezone.utc).timestamp())  # fixture 1002


@pytest.fixture
def conn(tmp_path):
    c = connect(tmp_path / "s.db")
    yield c
    c.close()


@pytest.fixture
def client():
    return SportmonksClient(token="t", sleep=lambda s: None)


def _fixture_row(fid, kickoff_ts, league_id=8, home="Liverpool", away="Manchester City", season_id=26001,
                 round_id=390002, home_id=8, away_id=9):
    return {"id": fid, "league_id": league_id, "season_id": season_id, "round_id": round_id, "round_name": "10",
            "name": f"{home} vs {away}", "starting_at": datetime.fromtimestamp(kickoff_ts, timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
            "starting_at_timestamp": kickoff_ts, "state_id": 1, "home_team_id": home_id, "home_team": home,
            "away_team_id": away_id, "away_team": away, "ft_home": None, "ft_away": None, "ht_home": None,
            "ht_away": None, "result_info": None, "synced_at": NOW.isoformat()}


def _odds_mocks(fid=1002):
    m1 = load_fixture("odds_1002_market1.json")
    responses.get(f"{BASE}/odds/pre-match/fixtures/{fid}/markets/1", json=m1)
    responses.get(f"{BASE}/odds/pre-match/fixtures/{fid}/markets/29", json={"data": []})


@pytest.mark.parametrize("hours,first,tag", [
    (48, True, "opening"), (48, False, "interim"), (6.0, False, "T-6h"), (3.0, False, "interim"),
    (1.0, False, "T-1h"), (0.1, False, "interim"), (-0.5, False, "post_kickoff"),
])
def test_snapshot_tag_windows(hours, first, tag):
    assert sync.snapshot_tag(hours, first) == tag


@responses.activate
def test_scheduler_produces_opening_t6_t1(conn, client):
    upsert(conn, "fixtures", [_fixture_row(1002, KICKOFF)])
    conn.commit()
    _odds_mocks()
    responses.get(f"{BASE}/fixtures/between/2030-11-01/2030-11-15", json={"data": [], "pagination": {"has_more": False}})

    ticks = {}
    t = NOW
    while t < datetime.fromtimestamp(KICKOFF, timezone.utc):
        res = run_due(client, conn, now=t)
        ticks[t] = [d["reason"] for d in res["due"]]
        t += timedelta(minutes=15)

    reasons = [r for rs in ticks.values() for r in rs]
    assert reasons == ["opening", "T-6h", "T-1h"]
    tags = [r[0] for r in conn.execute("SELECT DISTINCT snapshot_tag FROM odds_snapshots ORDER BY fetched_at")]
    assert tags == ["opening", "T-6h", "T-1h"]
    # fixtures refreshed at most every 6 h across the 11.5 h of ticks
    assert conn.execute("SELECT COUNT(*) FROM sync_log WHERE command='scheduler_fixture_refresh'").fetchone()[0] == 2
    cov = coverage(conn, since=date(2030, 10, 30))
    assert cov[0]["market_1"] == ["T-1h", "T-6h", "opening"] and cov[0]["market_29"] == []


def test_interim_refresh_after_24h(conn):
    upsert(conn, "fixtures", [_fixture_row(1002, KICKOFF)])
    opened = NOW - timedelta(days=3)
    conn.execute("INSERT INTO odds_snapshots (fetched_at, snapshot_tag, fixture_id, market_id, value) "
                 "VALUES (?, 'opening', 1002, 1, 2.0)", (opened.isoformat(),))
    assert [d.reason for d in due_snapshots(conn, opened + timedelta(hours=12))] == []
    assert [d.reason for d in due_snapshots(conn, opened + timedelta(hours=25))] == ["interim"]


def test_scheduler_ignores_other_leagues(conn):
    upsert(conn, "fixtures", [_fixture_row(5000, KICKOFF, league_id=9)])
    assert due_snapshots(conn, NOW) == []


@responses.activate
def test_client_drops_fixtures_from_other_leagues(client):
    page = load_fixture("fixtures_page2.json")
    page["data"][1]["league_id"] = 9
    page["data"][0].pop("league_id")
    responses.get(f"{BASE}/fixtures/between/2030-10-25/2030-11-05", json=page)
    assert client.fixtures_between(date(2030, 10, 25), date(2030, 11, 5)) == []


def test_audit_flags_leaks_and_unmapped_names(conn):
    upsert(conn, "seasons", [{"id": 26001, "league_id": 8, "name": "2026/2027", "starting_at": "2026-08-21",
                              "ending_at": "2027-05-23", "is_current": 1, "finished": 0, "synced_at": "x"}])
    upsert(conn, "fixtures", [
        _fixture_row(1, KICKOFF, home="Hull City", away="Coventry City", home_id=1, away_id=2),
        _fixture_row(2, KICKOFF, home="Ipswich Town", away="Arsenal", home_id=3, away_id=4),
        _fixture_row(3, KICKOFF, league_id=9, home="Stoke City", away="Hull City", home_id=5, away_id=1),
        _fixture_row(4, KICKOFF, home="Glasgow Rangers", away="Arsenal", home_id=6, away_id=4),
    ])
    report = audit_db(conn)
    assert not report["ok"]
    assert any("not in league 8" in e and "3(9)" in e for e in report["errors"])
    assert any("Glasgow Rangers" in e for e in report["errors"])
    season = report["seasons"][0]
    # promoted 2026/27 clubs are legitimate EPL teams per football-data E0 2627
    assert {"Hull", "Coventry", "Ipswich"} <= set(season["teams"])
    assert season["not_in_football_data"] == ["Glasgow Rangers"]
    assert report["team_name_mapping"]["Hull City"] == "Hull"


def test_audit_clean_db_ok(conn):
    upsert(conn, "fixtures", [_fixture_row(1, KICKOFF)])
    report = audit_db(conn)
    assert report["ok"] and report["errors"] == []
