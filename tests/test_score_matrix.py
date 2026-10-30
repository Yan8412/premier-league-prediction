import numpy as np
import pytest
from scipy.stats import poisson

from epl_prediction.models.dixon_coles import dc_score_matrix, invert_1x2
from epl_prediction.models.score_matrix import matrix_to_1x2, reweight_to_1x2, top_k_scores


def test_matrix_to_1x2_on_hand_built_matrix():
    m = np.array([
        [0.10, 0.05, 0.05],   # 0-0 draw, 0-1 away, 0-2 away
        [0.20, 0.10, 0.05],   # 1-0 home, 1-1 draw, 1-2 away
        [0.15, 0.20, 0.10],   # 2-0 home, 2-1 home, 2-2 draw
    ])
    np.testing.assert_allclose(matrix_to_1x2(m), [0.55, 0.30, 0.15])


def test_matrix_to_1x2_normalises_unnormalised_input():
    m = np.ones((3, 3))
    np.testing.assert_allclose(matrix_to_1x2(m), [3 / 9, 3 / 9, 3 / 9])


def test_top2_selection_and_tiebreak():
    m = np.zeros((4, 4))
    m[1, 0], m[1, 1], m[0, 0], m[2, 1] = 0.3, 0.3, 0.2, 0.2
    top = top_k_scores(m, 2)
    # 1-0 and 1-1 tie on p; lower total goals first
    assert [t["score"] for t in top] == ["1-0", "1-1"]
    assert top[0]["p"] == pytest.approx(0.3)


def test_top2_returns_highest_cells():
    m = dc_score_matrix(1.6, 0.9, -0.05)
    flat = sorted(((m[i, j], f"{i}-{j}") for i in range(m.shape[0]) for j in range(m.shape[1])), reverse=True)
    assert [t["score"] for t in top_k_scores(m, 2)] == [flat[0][1], flat[1][1]]


def test_dc_matrix_with_zero_rho_is_independent_poisson():
    m = dc_score_matrix(1.4, 1.1, 0.0, max_goals=12)
    g = np.arange(13)
    ref = np.outer(poisson.pmf(g, 1.4), poisson.pmf(g, 1.1))
    np.testing.assert_allclose(m, ref / ref.sum(), atol=1e-12)
    assert m.sum() == pytest.approx(1.0)


def test_negative_rho_inflates_draws():
    assert matrix_to_1x2(dc_score_matrix(1.3, 1.1, -0.1))[1] > matrix_to_1x2(dc_score_matrix(1.3, 1.1, 0.0))[1]


def test_reweight_to_1x2_hits_target_margins():
    m = dc_score_matrix(1.5, 1.0, -0.05)
    target = np.array([0.40, 0.30, 0.30])
    np.testing.assert_allclose(matrix_to_1x2(reweight_to_1x2(m, target)), target, atol=1e-12)


def test_invert_1x2_roundtrip():
    lam, mu = 1.75, 0.85
    p = matrix_to_1x2(dc_score_matrix(lam, mu, -0.05))
    lam2, mu2 = invert_1x2(p, rho=-0.05)
    assert lam2 == pytest.approx(lam, rel=1e-4)
    assert mu2 == pytest.approx(mu, rel=1e-4)
