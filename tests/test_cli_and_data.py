import json

import pytest

from epl_prediction.cli import main
from epl_prediction.config import FOOTBALL_DATA_DIR
from epl_prediction.data.football_data import _book_prefixes, load_season_csv
from epl_prediction.data.teams import canonical_team


def test_smoke_skips_without_token(capsys):
    assert main(["smoke"]) == 0
    assert "SKIP live smoke" in capsys.readouterr().out


def test_sync_requires_token(capsys):
    assert main(["sync", "all"]) == 2
    assert "SPORTMONKS_API_TOKEN is not set" in capsys.readouterr().err


def test_predict_offline_json(capsys):
    rc = main(["predict", "--home", "Arsenal", "--away", "Chelsea", "--date", "2025-03-01",
               "--odds", "1.80,3.80,4.50", "--no-db", "--football-weight", "0.4"])
    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    assert set(out["tracks"]) == {"football", "market", "ensemble"}
    assert out["tracks"]["ensemble"]["weights"]["football"] == 0.4
    assert "no auto-betting" in out["usage"]


def test_backtest_cli_runs_on_sample(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr("epl_prediction.backtest.walkforward.REPORTS_DIR", tmp_path)
    import epl_prediction.backtest.walkforward as wf

    orig = wf.write_report
    monkeypatch.setattr(wf, "write_report", lambda r, name: orig(r, out_dir=tmp_path, name=name))
    assert main(["backtest", "--test-seasons", "2425", "--name", "bt_test"]) == 0
    report = json.loads((tmp_path / "bt_test.json").read_text())
    assert report["n_matches"] == 380
    football = report["overall"]["football"]
    assert 0.85 < football["log_loss"] < 1.10
    assert 0.0 < football["top2_hit"] < 0.5


def test_football_data_book_detection():
    df = load_season_csv(FOOTBALL_DATA_DIR / "E0_2122.csv")
    assert len(df) == 380
    books = set().union(*df["odds_close"])
    assert "V" not in books and {"B365", "PS"} <= books
    assert df[["hg", "ag", "hthg", "htag"]].notna().all().all()


def test_book_prefix_rules():
    cols = ["B365H", "B365D", "B365A", "B365CH", "B365CD", "B365CA", "VCH", "VCD", "VCA",
            "VCCH", "VCCD", "VCCA", "AvgH", "AvgD", "AvgA", "BFEH", "BFED", "BFEA"]
    assert _book_prefixes(cols, closing=False) == ["B365", "VC"]
    assert _book_prefixes(cols, closing=True) == ["B365", "VC"]


@pytest.mark.parametrize("raw,canon", [
    ("Manchester United", "Man United"), ("Nottingham Forest", "Nott'm Forest"),
    ("Brighton & Hove Albion", "Brighton"), ("Arsenal", "Arsenal"), ("Arsenal FC", "Arsenal"),
    ("AFC Bournemouth", "Bournemouth"),
])
def test_canonical_team(raw, canon):
    assert canonical_team(raw) == canon
