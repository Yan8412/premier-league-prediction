import numpy as np
import pytest

from epl_prediction.backtest.metrics import accuracy, brier, log_loss, rps
from epl_prediction.models.ensemble import fit_pool_weight, log_pool, log_pool_rows
from epl_prediction.models.market import (consensus_1x2, consensus_htft, htft_to_ft_1x2, parse_1x2_label,
                                          parse_htft_label)
from epl_prediction.models.score_matrix import HTFT_ORDER


@pytest.mark.parametrize("label,name,expected", [
    ("1/1", None, "H/H"), ("1/X", None, "H/D"), ("X/2", None, "D/A"), ("Home/Draw", None, "H/D"),
    ("Draw - Away", None, "D/A"), ("HD", None, "H/D"), (None, "Away/Home", "A/H"), ("Mystery", "Unknown", None),
])
def test_parse_htft_label(label, name, expected):
    assert parse_htft_label(label, name) == expected


@pytest.mark.parametrize("label,expected", [("Home", "H"), ("1", "H"), ("X", "D"), ("Draw", "D"), ("2", "A"), ("Over", None)])
def test_parse_1x2_label(label, expected):
    assert parse_1x2_label(label) == expected


def test_consensus_1x2_averages_and_skips_invalid_books():
    est = consensus_1x2({"a": (2.0, 3.4, 4.0), "b": (2.1, 3.3, 3.9), "bad": (0.9, 3.0, 3.0)})
    assert est.n_books == 2 and set(est.books) == {"a", "b"}
    assert est.probs.sum() == pytest.approx(1.0)
    assert est.mean_overround > 0


def test_consensus_htft_requires_all_nine():
    full = {k: 9.0 for k in HTFT_ORDER}
    est = consensus_htft({"full": full, "partial": {"H/H": 3.0}})
    assert est.n_books == 1
    np.testing.assert_allclose(est.probs, np.full(9, 1 / 9))
    assert consensus_htft({"partial": {"H/H": 3.0}}) is None


def test_htft_to_ft():
    p = np.arange(1, 10, dtype=float)
    p /= p.sum()
    ft = htft_to_ft_1x2(p)  # columns of the 3x3 HT x FT grid
    np.testing.assert_allclose(ft, [p[0] + p[3] + p[6], p[1] + p[4] + p[7], p[2] + p[5] + p[8]])


def test_log_pool_properties():
    a, b = np.array([0.5, 0.3, 0.2]), np.array([0.2, 0.3, 0.5])
    np.testing.assert_allclose(log_pool([a, b], [1, 0]), a)
    pooled = log_pool([a, b], [0.5, 0.5])
    g = np.sqrt(a * b)
    np.testing.assert_allclose(pooled, g / g.sum())
    np.testing.assert_allclose(log_pool_rows(a[None], b[None], 0.5)[0], pooled)
    with pytest.raises(ValueError):
        log_pool([a, b], [0.5])


def test_fit_pool_weight_prefers_informative_model():
    rng = np.random.default_rng(1)
    n = 2000
    truth = rng.dirichlet([4, 2.5, 3], size=n)
    y = np.array([rng.choice(3, p=p) for p in truth])
    good, noise = truth, np.full((n, 3), 1 / 3)
    w, ll = fit_pool_weight(good, noise, y)
    assert w > 0.8
    assert ll <= log_loss(noise, y)


def test_metrics_known_values():
    p = np.array([[1.0, 0.0, 0.0], [0.5, 0.25, 0.25]])
    y = np.array([0, 2])
    assert log_loss(p, y) == pytest.approx(-np.log(0.25) / 2, rel=1e-9)
    assert brier(p, y) == pytest.approx((0 + (0.25 + 0.0625 + 0.5625)) / 2)
    assert rps(p, y) == pytest.approx((0 + ((0.5 - 0) ** 2 + (0.75 - 0) ** 2) / 2) / 2)
    assert accuracy(p, y) == 0.5
