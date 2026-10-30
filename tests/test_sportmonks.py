from datetime import date, datetime, timezone

import pytest
import responses
from responses import matchers

from epl_prediction.db import connect
from epl_prediction.predict import (fit_football, fixture_inputs_from_db, load_history, next_matchday_fixture_ids,
                                    predict_fixture, store_predictions)
from epl_prediction.review import weekly_review
from epl_prediction.sportmonks import sync
from epl_prediction.sportmonks.client import MissingTokenError, SportmonksClient, SportmonksError
from epl_prediction.sportmonks.parse import parse_fixture

from .conftest import load_fixture

BASE = "https://api.sportmonks.com/v3/football"
TOKEN = "test-token-not-real"


@pytest.fixture
def client():
    return SportmonksClient(token=TOKEN, sleep=lambda s: None)


@pytest.fixture
def conn(tmp_path):
    c = connect(tmp_path / "sm.db")
    yield c
    c.close()


def test_missing_token_raises():
    with pytest.raises(MissingTokenError):
        SportmonksClient()


@responses.activate
def test_token_sent_in_header_not_url(client):
    responses.get(f"{BASE}/leagues/8", json=load_fixture("league_8.json"),
                  match=[matchers.header_matcher({"Authorization": TOKEN})])
    league = client.league(8)
    assert league["name"] == "Premier League"
    assert TOKEN not in responses.calls[0].request.url
    assert "include=currentSeason%3Bseasons" in responses.calls[0].request.url
    assert client.last_rate_limit["remaining"] == 2999


@responses.activate
def test_cursor_pagination_and_league_filter(client):
    url = f"{BASE}/fixtures/between/2026-10-20/2026-11-05"
    responses.get(url, json=load_fixture("fixtures_page1.json"),
                  match=[matchers.query_param_matcher({"include": "participants;scores;round",
                                                       "filters": "fixtureLeagues:8", "per_page": "50"})])
    responses.get(url, json=load_fixture("fixtures_page2.json"),
                  match=[matchers.query_param_matcher({"include": "participants;scores;round",
                                                       "filters": "fixtureLeagues:8", "cursor": "CURSOR-2"})])
    fixtures = client.fixtures_between(date(2026, 10, 20), date(2026, 11, 5))
    assert [f["id"] for f in fixtures] == [1001, 1002, 1003]
    assert "per_page" not in responses.calls[1].request.url  # docs: per_page + cursor -> HTTP 400


def _two_page_mock(url, first_pagination, second_params):
    page1, page2 = load_fixture("fixtures_page1.json"), load_fixture("fixtures_page2.json")
    page1["pagination"] = first_pagination
    base = {"include": "participants;scores;round", "filters": "fixtureLeagues:8"}
    responses.get(url, json=page1, match=[matchers.query_param_matcher({**base, "per_page": "50"})])
    responses.get(url, json=page2, match=[matchers.query_param_matcher({**base, **second_params})])


@responses.activate
def test_pagination_follows_next_page_url(client):
    """Live shape: next_page and next_cursor are both full URLs."""
    url = f"{BASE}/fixtures/between/2026-08-21/2026-11-28"
    _two_page_mock(url, {
        "count": 50, "per_page": 50, "current_page": 1, "has_more": True,
        "next_page": f"{url}?page=2&per_page=50&filters=fixtureLeagues%3A8",
        "next_cursor": f"{url}?cursor=WzEsNTAsWzE5MTM0NDU0XV0",
    }, {"page": "2", "per_page": "50"})
    fixtures = client.fixtures_between(date(2026, 8, 21), date(2026, 11, 28))
    assert [f["id"] for f in fixtures] == [1001, 1002, 1003]
    assert all("cursor" not in c.request.url for c in responses.calls)


@responses.activate
def test_pagination_extracts_token_from_cursor_url(client):
    url = f"{BASE}/fixtures/between/2026-08-21/2026-11-28"
    _two_page_mock(url, {
        "count": 50, "per_page": 50, "current_page": 1, "has_more": True,
        "next_cursor": f"{url}?cursor=WzEsNTAsWzE5MTM0NDU0XV0",
    }, {"cursor": "WzEsNTAsWzE5MTM0NDU0XV0"})
    fixtures = client.fixtures_between(date(2026, 8, 21), date(2026, 11, 28))
    assert [f["id"] for f in fixtures] == [1001, 1002, 1003]
    assert "per_page" not in responses.calls[1].request.url


@responses.activate
def test_date_range_split_into_100_day_windows(client):
    empty = {"data": [], "pagination": {"has_more": False, "next_cursor": None}}
    responses.get(f"{BASE}/fixtures/between/2026-08-01/2026-11-08", json=empty)
    responses.get(f"{BASE}/fixtures/between/2026-11-09/2027-02-16", json=empty)
    responses.get(f"{BASE}/fixtures/between/2027-02-17/2027-05-27", json=empty)
    responses.get(f"{BASE}/fixtures/between/2027-05-28/2027-05-30", json=empty)
    assert client.fixtures_between(date(2026, 8, 1), date(2027, 5, 30)) == []
    assert len(responses.calls) == 4


@responses.activate
def test_retries_on_429_then_succeeds(client):
    url = f"{BASE}/fixtures/1001"
    responses.get(url, status=429, json={"message": "Too many", "rate_limit": {"resets_in_seconds": 5, "remaining": 0}})
    responses.get(url, json={"data": load_fixture("fixtures_page1.json")["data"][0]})
    assert client.fixture(1001)["id"] == 1001
    assert len(responses.calls) == 2


@responses.activate
def test_403_raises_with_hint(client):
    responses.get(f"{BASE}/odds/pre-match/fixtures/1/markets/29", status=403, json={"message": "No access"})
    with pytest.raises(SportmonksError) as exc:
        client.prematch_odds(1, 29)
    assert exc.value.status == 403 and "plan" in str(exc.value)


def test_parse_fixture_scores():
    fx = parse_fixture(load_fixture("fixtures_page1.json")["data"][0])
    assert (fx["home_team"], fx["away_team"]) == ("Arsenal", "Chelsea")
    assert (fx["ft_home"], fx["ft_away"], fx["ht_home"], fx["ht_away"]) == (2, 1, 1, 0)
    assert fx["round_name"] == "9"


def test_snapshot_tags():
    assert sync.snapshot_tag(48, True) == "opening"
    assert sync.snapshot_tag(48, False) == "interim"
    assert sync.snapshot_tag(5.9, False) == "T-6h"
    assert sync.snapshot_tag(0.8, False) == "T-1h"
    assert sync.snapshot_tag(-0.1, True) == "post_kickoff"


def _mock_sync_responses():
    responses.get(f"{BASE}/leagues/8", json=load_fixture("league_8.json"))
    page1, page2 = load_fixture("fixtures_page1.json"), load_fixture("fixtures_page2.json")
    page1["pagination"]["has_more"] = False
    responses.get(f"{BASE}/fixtures/between/2026-10-20/2026-11-05", json=page1)
    responses.get(f"{BASE}/fixtures/between/2030-10-25/2030-11-05", json=page2)
    responses.get(f"{BASE}/odds/pre-match/fixtures/1002/markets/1", json=load_fixture("odds_1002_market1.json"))
    responses.get(f"{BASE}/odds/pre-match/fixtures/1002/markets/29", json=load_fixture("odds_1002_market29.json"))
    responses.get(f"{BASE}/odds/pre-match/fixtures/1003/markets/1", json={"data": []})
    responses.get(f"{BASE}/odds/pre-match/fixtures/1003/markets/29", json={"data": []})


@responses.activate
def test_sync_to_sqlite_and_predict_end_to_end(client, conn, tmp_path):
    _mock_sync_responses()
    seasons = sync.sync_seasons(client, conn)
    assert {s["id"] for s in seasons} == {25583, 26001}
    assert sync.current_season(conn)["id"] == 26001

    assert sync.sync_fixtures(client, conn, date(2026, 10, 20), date(2026, 11, 5)) == 1
    assert sync.sync_fixtures(client, conn, date(2030, 10, 25), date(2030, 11, 5)) == 2

    now = datetime(2030, 10, 31, 12, 0, tzinfo=timezone.utc)
    ids = sync.upcoming_fixture_ids(conn, days=7, now=now)
    assert ids == [1002, 1003]
    stats = sync.sync_odds(client, conn, ids, now=now)
    assert stats["by_market"]["1"] == {"fixtures_with_odds": 1, "fixtures_empty": 1, "rows": 6}
    assert stats["by_market"]["29"]["fixtures_empty"] == 1  # sparse HT/FT degrades gracefully
    tags = {r[0] for r in conn.execute("SELECT DISTINCT snapshot_tag FROM odds_snapshots")}
    assert tags == {"opening"}

    assert next_matchday_fixture_ids(conn, now=now) == [1002, 1003]
    inputs = fixture_inputs_from_db(conn, [1002, 1003])
    liv = inputs[0]
    assert set(liv.odds_1x2) == {"bet365", "Betfair"}
    assert len(liv.odds_htft["bet365"]) == 9 and "Betfair" in liv.odds_htft  # Betfair incomplete -> skipped later
    assert liv.odds_meta["29"]["unparsed_selections"] == 1

    model = fit_football(load_history(conn))
    p_liv = predict_fixture(model, liv, pool_weight=0.3)
    t = p_liv["tracks"]
    assert p_liv["home"] == "Liverpool" and p_liv["away"] == "Man City"
    for name in ("football", "market", "ensemble"):
        assert len(t[name]["p_1x2"]) == 3 and sum(t[name]["p_1x2"]) == pytest.approx(1, abs=1e-3)
        assert len(t[name]["top2_scores"]) == 2
        assert t[name]["model_version"] and t[name]["generated_at"]
    assert t["football"]["p_htft"] is None
    assert len(t["market"]["p_htft"]) == 9 and t["market"]["n_bookmakers_htft"] == 1
    assert t["ensemble"]["p_htft"] == t["market"]["p_htft"]
    assert t["ensemble"]["weights"] == {"football": 0.3, "market": 0.7}

    p_wol = predict_fixture(model, inputs[1])  # no odds at all
    assert p_wol["home"] == "Wolves" and p_wol["away"] == "Brighton"
    assert p_wol["tracks"]["market"]["available"] is False
    assert p_wol["tracks"]["ensemble"]["p_1x2"] == p_wol["tracks"]["football"]["p_1x2"]

    store_predictions(conn, [p_liv, p_wol])
    assert conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0] == 5  # wolves market track skipped

    # post-match review once 1002 is finished
    conn.execute("UPDATE fixtures SET state_id=5, ft_home=1, ft_away=1 WHERE id=1002")
    review = weekly_review(conn, days=7, now=datetime(2030, 11, 3, tzinfo=timezone.utc), out_dir=tmp_path / "reviews")
    assert review["tracks"]["market"]["n"] == 1

