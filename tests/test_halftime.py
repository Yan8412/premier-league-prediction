import numpy as np
import pytest
from scipy.stats import poisson

from epl_prediction.contract import validate_prediction
from epl_prediction.models.dixon_coles import DixonColesModel, dc_score_matrix
from epl_prediction.models.halftime import align_htft_to_1x2, htft_index, htft_label, htft_probs
from epl_prediction.models.score_matrix import HTFT_ORDER, matrix_to_1x2
from epl_prediction.predict import FixtureInput, predict_fixture

from .test_dixon_coles import synthetic_league


@pytest.mark.parametrize("ht,ft,label", [
    ((1, 0), (2, 0), "H/H"), ((1, 0), (1, 1), "H/D"), ((2, 1), (2, 3), "H/A"),
    ((0, 0), (1, 0), "D/H"), ((0, 0), (0, 0), "D/D"), ((1, 1), (1, 2), "D/A"),
    ((0, 1), (2, 1), "A/H"), ((0, 2), (2, 2), "A/D"), ((0, 1), (0, 3), "A/A"),
])
def test_htft_mapping(ht, ft, label):
    assert htft_label(*ht, *ft) == label
    assert HTFT_ORDER[htft_index(*ht, *ft)] == label


def brute_force_htft(lam, mu, s_h, s_a, g=10):
    """Independent-halves enumeration (rho = 0) for cross-checking."""
    k = np.arange(g + 1)
    p1h, p1a = poisson.pmf(k, lam * s_h), poisson.pmf(k, mu * s_a)
    p2h, p2a = poisson.pmf(k, lam * (1 - s_h)), poisson.pmf(k, mu * (1 - s_a))
    out = np.zeros(9)
    for h1 in k:
        for a1 in k:
            for h2 in k:
                for a2 in k:
                    out[htft_index(h1, a1, h1 + h2, a1 + a2)] += p1h[h1] * p1a[a1] * p2h[h2] * p2a[a2]
    return out / out.sum()


def test_htft_matches_brute_force_without_rho():
    lam, mu, s_h, s_a = 1.6, 1.1, 0.45, 0.42
    g = 10
    k = np.arange(g + 1)
    ft = np.outer(poisson.pmf(k, lam), poisson.pmf(k, mu))  # sum of independent halves
    np.testing.assert_allclose(htft_probs(lam, mu, s_h, s_a, 0.0, ft), brute_force_htft(lam, mu, s_h, s_a, g), atol=2e-4)


def test_htft_ft_margins_equal_1x2():
    ft = dc_score_matrix(1.4, 1.2, -0.08)
    p = htft_probs(1.4, 1.2, 0.45, 0.42, 0.05, ft)
    assert p.sum() == pytest.approx(1.0)
    np.testing.assert_allclose(p.reshape(3, 3).sum(axis=0), matrix_to_1x2(ft), atol=1e-12)


def test_align_htft_to_1x2():
    p = np.full(9, 1 / 9)
    out = align_htft_to_1x2(p, [0.5, 0.3, 0.2])
    np.testing.assert_allclose(out.reshape(3, 3).sum(axis=0), [0.5, 0.3, 0.2])


def test_football_htft_non_null_after_fit():
    df, _ = synthetic_league()
    rng = np.random.default_rng(3)
    df["hthg"] = rng.binomial(df["hg"], 0.45)
    df["htag"] = rng.binomial(df["ag"], 0.40)
    m = DixonColesModel().fit(df)
    assert m.ht_share_home == pytest.approx(0.45, abs=0.05)
    assert m.ht_share_away == pytest.approx(0.40, abs=0.06)

    pred = predict_fixture(m, FixtureInput(home="T0", away="T7"))
    f = pred["tracks"]["football"]
    assert f["p_htft"] is not None and len(f["p_htft"]) == 9
    assert sum(f["p_htft"]) == pytest.approx(1.0, abs=1e-3)
    assert pred["tracks"]["ensemble"]["p_htft"] == f["p_htft"]  # no market odds -> football only
    assert validate_prediction(pred) == []


def test_validate_prediction_flags_problems():
    df, _ = synthetic_league()
    pred = predict_fixture(DixonColesModel().fit(df), FixtureInput(home="T0", away="T1"))
    pred["tracks"]["football"]["p_htft"] = None
    pred["tracks"]["ensemble"]["p_1x2"] = [0.5, 0.5, 0.5]
    errors = validate_prediction(pred)
    assert any("football: p_htft is null" in e for e in errors)
    assert any("ensemble: p_1x2" in e for e in errors)
