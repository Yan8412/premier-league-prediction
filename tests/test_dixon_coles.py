import numpy as np
import pandas as pd
import pytest

from epl_prediction.models.dixon_coles import DixonColesModel


def synthetic_league(seed=0, rounds=6):
    rng = np.random.default_rng(seed)
    teams = [f"T{i}" for i in range(8)]
    attack = np.linspace(0.4, -0.4, 8)
    defence = np.linspace(-0.3, 0.3, 8)
    home_adv = 0.3
    rows, day = [], pd.Timestamp("2024-01-01")
    for _ in range(rounds):
        for i, h in enumerate(teams):
            for j, a in enumerate(teams):
                if i == j:
                    continue
                lam = np.exp(home_adv + attack[i] + defence[j])
                mu = np.exp(attack[j] + defence[i])
                rows.append({"date": day, "home": h, "away": a, "hg": rng.poisson(lam), "ag": rng.poisson(mu)})
                day += pd.Timedelta(days=1)
    return pd.DataFrame(rows), teams


def test_fit_recovers_ordering_and_home_advantage():
    df, teams = synthetic_league()
    m = DixonColesModel(xi=0.0, l2=1e-4).fit(df, max_days=None)
    assert m.converged
    assert m.home_adv == pytest.approx(0.3, abs=0.1)
    strength = {t: a - d for t, a, d in zip(m.teams, m.attack, m.defence)}
    assert strength["T0"] > strength["T3"] > strength["T7"]
    p = m.predict_1x2("T0", "T7")
    assert p.sum() == pytest.approx(1.0) and p[0] > 0.6


def test_fit_uses_only_matches_before_as_of():
    df, _ = synthetic_league()
    cutoff = df["date"].iloc[100]
    m = DixonColesModel().fit(df, as_of=cutoff)
    assert m.n_matches == 100


def test_unseen_team_gets_weak_prior():
    df, _ = synthetic_league()
    m = DixonColesModel().fit(df)
    lam_known, _ = m.expected_goals("T0", "T1")
    lam_new, _ = m.expected_goals("Promoted FC", "T1")
    assert lam_new < lam_known


def test_too_few_matches():
    df, _ = synthetic_league(rounds=1)
    with pytest.raises(ValueError):
        DixonColesModel().fit(df.head(5))
