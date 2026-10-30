import numpy as np
import pytest

from epl_prediction.models.devig import devig, multiplicative, overround, shin, shin_z


def shin_prices(p, z):
    """Forward Shin model: bookmaker implied probabilities for true probs p and insider share z."""
    p = np.asarray(p, dtype=float)
    s = np.sqrt(z * p + (1 - z) * p**2)
    return s * s.sum()


def test_multiplicative_normalises():
    p = multiplicative([2.0, 3.5, 4.0])
    q = np.array([1 / 2.0, 1 / 3.5, 1 / 4.0])
    assert p.sum() == pytest.approx(1.0)
    np.testing.assert_allclose(p, q / q.sum())


def test_overround():
    assert overround([1.9, 1.9]) == pytest.approx(2 / 1.9 - 1)


@pytest.mark.parametrize("p,z", [([0.5, 0.3, 0.2], 0.02), ([0.7, 0.2, 0.1], 0.05), ([0.25, 0.25, 0.5], 0.01)])
def test_shin_recovers_true_probabilities(p, z):
    odds = 1.0 / shin_prices(p, z)
    assert shin_z(odds) == pytest.approx(z, abs=1e-8)
    np.testing.assert_allclose(shin(odds), p, atol=1e-8)


def test_shin_shades_longshot_more_than_multiplicative():
    odds = [1.30, 5.50, 11.0]
    s, m = shin(odds), multiplicative(odds)
    assert s.sum() == pytest.approx(1.0)
    assert s[0] > m[0] and s[2] < m[2]  # favourite-longshot bias correction


def test_shin_without_overround_is_normalised_implied():
    odds = [2.0, 4.0, 4.0]  # booksum == 1
    np.testing.assert_allclose(shin(odds), [0.5, 0.25, 0.25])
    assert shin_z(odds) == 0.0


@pytest.mark.parametrize("bad", [[1.0, 2.0, 3.0], [2.0], [2.0, float("nan"), 3.0], [-2.0, 3.0, 3.0]])
def test_invalid_odds_raise(bad):
    with pytest.raises(ValueError):
        devig(bad, "shin")


def test_unknown_method():
    with pytest.raises(ValueError):
        devig([2.0, 3.0, 4.0], "power")
