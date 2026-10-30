"""Overround removal ("de-vig") for decimal odds.

* ``multiplicative``: normalise implied probabilities ``1/o`` to sum to one.
  Used for high-dimensional markets (e.g. 9-way HT/FT) where Shin is fragile.
* ``shin``: Shin (1992/1993) insider-trading model. Given implied
  probabilities ``q_i`` with booksum ``B = sum q_i`` the fair probabilities are

      p_i(z) = (sqrt(z^2 + 4 (1 - z) q_i^2 / B) - z) / (2 (1 - z))

  and ``z`` (share of insider money) is chosen so that ``sum p_i(z) = 1``.
  Self-contained implementation; the closed form follows Jullien & Salanie
  (1994) and the approach popularised by mberk/shin (MIT).
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
from scipy.optimize import brentq


def implied_probabilities(odds: Sequence[float]) -> np.ndarray:
    o = np.asarray(odds, dtype=float)
    if o.ndim != 1 or o.size < 2:
        raise ValueError("need a 1-D vector of at least two decimal odds")
    if not np.all(np.isfinite(o)) or np.any(o <= 1.0):
        raise ValueError(f"decimal odds must be finite and > 1, got {o.tolist()}")
    return 1.0 / o


def overround(odds: Sequence[float]) -> float:
    return float(implied_probabilities(odds).sum() - 1.0)


def multiplicative(odds: Sequence[float]) -> np.ndarray:
    q = implied_probabilities(odds)
    return q / q.sum()


def _shin_probs(q: np.ndarray, booksum: float, z: float) -> np.ndarray:
    return (np.sqrt(z * z + 4.0 * (1.0 - z) * q * q / booksum) - z) / (2.0 * (1.0 - z))


def shin_z(odds: Sequence[float]) -> float:
    q = implied_probabilities(odds)
    booksum = q.sum()
    if booksum <= 1.0:
        return 0.0
    f = lambda z: _shin_probs(q, booksum, z).sum() - 1.0  # noqa: E731
    # f(0) = sqrt(B) - 1 > 0 and f decreases in z; widen the bracket if needed.
    hi = 0.5
    while f(hi) > 0 and hi < 0.999:
        hi = min(0.999, hi + (1.0 - hi) / 2)
    return float(brentq(f, 0.0, hi, xtol=1e-12))


def shin(odds: Sequence[float]) -> np.ndarray:
    """Shin fair probabilities. Falls back to normalisation when booksum <= 1."""
    q = implied_probabilities(odds)
    booksum = q.sum()
    if booksum <= 1.0:
        return q / booksum
    p = _shin_probs(q, booksum, shin_z(odds))
    return p / p.sum()


METHODS = {"shin": shin, "multiplicative": multiplicative}


def devig(odds: Sequence[float], method: str = "shin") -> np.ndarray:
    try:
        fn = METHODS[method]
    except KeyError as exc:
        raise ValueError(f"unknown de-vig method {method!r}; choose from {sorted(METHODS)}") from exc
    return fn(odds)
