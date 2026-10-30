import json
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from epl_prediction.db import connect, upsert
from epl_prediction.models.halftime import htft_index
from epl_prediction.predict import load_pool_weight
from epl_prediction.review import fit_live_weights, monthly_review, scored_predictions, weekly_review

NOW = datetime(2030, 12, 1, 12, 0, tzinfo=timezone.utc)
HT = {0: (1, 0), 1: (0, 0), 2: (0, 1)}


def _scores(cls: int) -> tuple[tuple[int, int], tuple[int, int]]:
    """An (HT score, FT score) pair of HT/FT class ``cls`` with FT >= HT for both sides."""
    ht_res, ft_res = divmod(cls, 3)
    h, a = HT[ht_res]
    m = max(h, a)
    return (h, a), {0: (m + 2, m), 1: (m + 1, m + 1), 2: (m, m + 2)}[ft_res]


@pytest.mark.parametrize("cls", range(9))
def test_score_helper_roundtrip(cls):
    (hh, ha), (fh, fa) = _scores(cls)
    assert fh >= hh and fa >= ha and htft_index(hh, ha, fh, fa) == cls


def _populate(conn, n=80, seed=0):
    rng = np.random.default_rng(seed)
    for k in range(n):
        fid = 10_000 + k
        ko = NOW - timedelta(days=1 + k % 25, hours=k)
        truth = rng.dirichlet(np.full(9, 2.0))
        cls = rng.choice(9, p=truth)
        (hh, ha), (fh, fa) = _scores(cls)
        upsert(conn, "fixtures", [{
            "id": fid, "league_id": 8, "season_id": 1, "round_id": 1, "round_name": "1", "name": "A v B",
            "starting_at": ko.strftime("%Y-%m-%d %H:%M:%S"), "starting_at_timestamp": int(ko.timestamp()),
            "state_id": 5, "home_team_id": 1, "home_team": "A", "away_team_id": 2, "away_team": "B",
            "ft_home": fh, "ft_away": fa, "ht_home": hh, "ht_away": ha, "result_info": None, "synced_at": "x"}])
        truth_1x2 = truth.reshape(3, 3).sum(axis=0)
        tracks = {
            # market sharp on 1X2, flat on HT/FT; football the other way round
            "football": (np.full(3, 1 / 3), truth),
            "market": (truth_1x2, np.full(9, 1 / 9)),
        }
        gen = (ko - timedelta(hours=2)).isoformat(timespec="seconds")
        for track, (p1, ph) in tracks.items():
            payload = {"p_1x2": p1.tolist(), "p_htft": ph.tolist(),
                       "top2_scores": [{"score": f"{fh}-{fa}", "p": 0.2}, {"score": "9-9", "p": 0.1}],
                       "model_version": "t", "generated_at": gen}
            conn.execute("INSERT INTO predictions (fixture_id, home_team, away_team, kickoff, track, model_version, "
                         "generated_at, p_home, p_draw, p_away, payload_json) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                         (fid, "A", "B", None, track, "t", gen, *p1.tolist(), json.dumps(payload)))
        # a post-kickoff prediction must never be scored
        late = (ko + timedelta(hours=3)).isoformat(timespec="seconds")
        conn.execute("INSERT INTO predictions (fixture_id, home_team, away_team, kickoff, track, model_version, "
                     "generated_at, p_home, p_draw, p_away, payload_json) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                     (fid, "A", "B", None, "football", "late", late, 1.0, 0.0, 0.0,
                      json.dumps({"p_1x2": [1, 0, 0], "p_htft": None, "top2_scores": []})))
    conn.commit()


@pytest.fixture
def conn(tmp_path):
    c = connect(tmp_path / "r.db")
    _populate(c)
    yield c
    c.close()


def test_scored_predictions_ignore_post_kickoff(conn):
    items = scored_predictions(conn)
    assert len(items) == 160
    assert all(i["p_1x2"] != [1.0, 0.0, 0.0] for i in items)


def test_weekly_review_scores_1x2_htft_top2(conn, tmp_path):
    res = weekly_review(conn, days=7, now=NOW, out_dir=tmp_path / "reviews")
    assert set(res["tracks"]) == {"football", "market"}
    f = res["tracks"]["football"]
    assert f["top2_hit"] == 1.0 and f["htft"]["n"] == f["n"] > 0
    assert f["log_loss"] == pytest.approx(np.log(3))
    assert (tmp_path / "reviews" / "weekly_2030-12-01.md").exists()


def test_live_weights_follow_each_markets_skill(conn):
    w = fit_live_weights(conn, min_matches=50, now=NOW)
    assert w["1x2"]["n"] == 80 and w["htft"]["n"] == 80
    assert w["1x2"]["w_football"] < 0.2     # market is the sharp 1X2 source
    assert w["htft"]["w_football"] > 0.8    # football is the sharp HT/FT source
    assert fit_live_weights(conn, min_matches=500)["1x2"]["w_football"] is None


def test_monthly_review_writes_weights_used_by_predict(conn, tmp_path):
    weights = tmp_path / "ensemble_weights.json"
    res = monthly_review(conn, now=NOW, out_dir=tmp_path / "reviews", weights_path=weights, run_backtest_=False)
    assert weights.exists() and res["live_weights"]["htft"]["w_football"] > 0.8
    assert (tmp_path / "reviews" / "monthly_2030-12-01.md").read_text(encoding="utf-8").count("htft") >= 1
    missing = tmp_path / "nope.json"
    assert load_pool_weight(report=missing, kind="htft", live=weights) == res["live_weights"]["htft"]["w_football"]
    assert load_pool_weight(report=missing, kind="close", live=weights) == res["live_weights"]["1x2"]["w_football"]
    assert load_pool_weight(report=missing, kind="htft", live=missing) == 0.5
