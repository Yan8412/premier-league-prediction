import sqlite3

import numpy as np
import pandas as pd
import pytest
import responses

from epl_prediction.cli import main
from epl_prediction.data.xg import CsvXG, UnderstatXG, attach_xg, load_store, save_to_store
from epl_prediction.db import connect
from epl_prediction.models.dixon_coles import DixonColesModel
from epl_prediction.sportmonks import sync
from epl_prediction.sportmonks.client import SportmonksClient
from epl_prediction.sportmonks.parse import parse_fixture

from .conftest import FIXTURES, load_fixture
from .test_dixon_coles import synthetic_league

BASE = "https://api.sportmonks.com/v3/football"


def test_understat_file_import_parses_finished_matches_only():
    df = UnderstatXG(FIXTURES / "understat_dates_sample.json").match_xg()
    assert len(df) == 2
    first = df.iloc[0]
    assert (first["home"], first["away"], first["source"]) == ("Arsenal", "Chelsea", "understat")
    assert first["hxg"] == pytest.approx(1.84213)
    assert df.iloc[1]["home"] == "Man City" and df.iloc[1]["away"] == "Wolves"  # canonical names


def test_csv_import_and_store_roundtrip(tmp_path):
    src = tmp_path / "x.csv"
    src.write_text("Date,Home,Away,HxG,AxG\n2025-11-01,Manchester United,Fulham,1.2,0.8\n")
    df = CsvXG(src, "fbref_manual").match_xg()
    path = save_to_store(df, "fbref_manual", xg_dir=tmp_path / "xg")
    back = load_store(tmp_path / "xg")
    assert path.name == "fbref_manual_2025-11-01_2025-11-01.csv"
    assert back.iloc[0][["home", "away", "source"]].tolist() == ["Man United", "Fulham", "fbref_manual"]


def test_attach_xg_matches_within_one_day_and_keeps_existing():
    hist = pd.DataFrame({"date": pd.to_datetime(["2025-11-01", "2025-11-02", "2025-11-03"]),
                         "home": ["Arsenal", "Man City", "Everton"], "away": ["Chelsea", "Wolves", "Fulham"],
                         "hg": [2, 3, 0], "ag": [1, 0, 0], "hxg": [np.nan, np.nan, 0.5], "axg": [np.nan, np.nan, 0.4]})
    us = UnderstatXG(FIXTURES / "understat_dates_sample.json").match_xg()
    out = attach_xg(hist, [us])
    assert out.loc[0, "hxg"] == pytest.approx(1.84213) and out.loc[0, "xg_source"] == "understat"
    assert out.loc[1, "axg"] == pytest.approx(0.33)  # date differs by one day (timezone)
    assert out.loc[2, "hxg"] == 0.5 and out.loc[2, "xg_source"] == "inline"


def test_xg_mix_changes_fit_and_zero_weight_ignores_xg():
    df, _ = synthetic_league()
    base = DixonColesModel(xg_weight=0.0).fit(df.assign(hxg=np.nan, axg=np.nan))
    no_xg_cols = DixonColesModel().fit(df)
    np.testing.assert_allclose(base.attack, no_xg_cols.attack, atol=1e-6)

    # xG says T7 dominates every match it plays; with xG mixed in T7 gets stronger
    dfx = df.assign(hxg=df["hg"].astype(float), axg=df["ag"].astype(float))
    dfx.loc[dfx["home"] == "T7", "hxg"] += 2.0
    dfx.loc[dfx["away"] == "T7", "axg"] += 2.0
    goals_only = DixonColesModel(xg_weight=0.0).fit(dfx)
    mixed = DixonColesModel(xg_weight=0.5).fit(dfx)
    k = mixed.teams.index("T7")
    assert mixed.n_xg_matches == len(dfx) and goals_only.n_xg_matches == 0
    assert mixed.attack[k] > goals_only.attack[k] + 0.2


def test_parse_fixture_reads_xgfixture_include():
    fx = load_fixture("fixtures_page1.json")["data"][0]
    fx["xgfixture"] = [
        {"id": 1, "fixture_id": 1001, "type_id": 5304, "participant_id": 19, "data": {"value": 1.9}, "location": "home"},
        {"id": 2, "fixture_id": 1001, "type_id": 5304, "participant_id": 18, "data": {"value": 0.7}, "location": "away"},
        {"id": 3, "fixture_id": 1001, "type_id": 9999, "participant_id": 18, "data": {"value": 9.9}, "location": "away"},
    ]
    row = parse_fixture(fx)
    assert (row["xg_home"], row["xg_away"]) == (1.9, 0.7)
    assert parse_fixture(load_fixture("fixtures_page1.json")["data"][0])["xg_home"] is None


@responses.activate
def test_sync_with_xg_falls_back_on_403_and_keeps_stored_xg(tmp_path):
    conn = connect(tmp_path / "x.db")
    client = SportmonksClient(token="t", sleep=lambda s: None)
    url = f"{BASE}/fixtures/between/2026-10-20/2026-10-30"
    page = load_fixture("fixtures_page1.json")
    page["pagination"]["has_more"] = False
    with_xg = load_fixture("fixtures_page1.json")
    with_xg["pagination"]["has_more"] = False
    with_xg["data"][0]["xgfixture"] = [
        {"type_id": 5304, "data": {"value": 1.9}, "location": "home"},
        {"type_id": 5304, "data": {"value": 0.7}, "location": "away"}]

    responses.get(url, json=with_xg, match=[responses.matchers.query_param_matcher(
        {"include": "participants;scores;round;xGFixture", "filters": "fixtureLeagues:8", "per_page": "50"})])
    sync.sync_fixtures(client, conn, pd.Timestamp("2026-10-20").date(), pd.Timestamp("2026-10-30").date(), with_xg=True)
    assert tuple(conn.execute("SELECT xg_home, xg_away FROM fixtures WHERE id=1001").fetchone()) == (1.9, 0.7)

    responses.reset()
    responses.get(url, status=403, json={"message": "xG add-on required"},
                  match=[responses.matchers.query_param_matcher(
                      {"include": "participants;scores;round;xGFixture", "filters": "fixtureLeagues:8", "per_page": "50"})])
    responses.get(url, json=page)
    sync.sync_fixtures(client, conn, pd.Timestamp("2026-10-20").date(), pd.Timestamp("2026-10-30").date(), with_xg=True)
    # plain sync must not wipe previously stored xG
    assert tuple(conn.execute("SELECT xg_home, xg_away FROM fixtures WHERE id=1001").fetchone()) == (1.9, 0.7)


def test_migration_adds_xg_columns_to_old_db(tmp_path):
    from importlib import resources

    schema = resources.files("epl_prediction").joinpath("schema.sql").read_text(encoding="utf-8")
    pre_xg = "\n".join(line for line in schema.splitlines() if not line.strip().startswith("xg_"))
    path = tmp_path / "old.db"
    old = sqlite3.connect(path)
    old.executescript(pre_xg)
    assert "xg_home" not in {r[1] for r in old.execute("PRAGMA table_info(fixtures)")}
    old.close()
    cols = {r[1] for r in connect(path).execute("PRAGMA table_info(fixtures)")}
    assert {"xg_home", "xg_away"} <= cols


def test_cli_xg_import(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr("epl_prediction.data.xg.XG_DIR", tmp_path / "xg")
    import epl_prediction.data.xg as xgmod

    orig_save = xgmod.save_to_store
    monkeypatch.setattr(xgmod, "save_to_store", lambda df, source: orig_save(df, source, xg_dir=tmp_path / "xg"))
    assert main(["xg", "import", "--understat", str(FIXTURES / "understat_dates_sample.json")]) == 0
    assert "imported 2 matches" in capsys.readouterr().out
    assert len(list((tmp_path / "xg").glob("understat_*.csv"))) == 1
