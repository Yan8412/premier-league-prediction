"""Score-matrix helpers. ``m[i, j]`` = P(home scores i, away scores j)."""

from __future__ import annotations

import numpy as np
from scipy.stats import poisson

OUTCOMES_1X2 = ("H", "D", "A")
# HT/FT order used everywhere: first symbol = half-time result, second = full-time.
HTFT_ORDER = ("H/H", "H/D", "H/A", "D/H", "D/D", "D/A", "A/H", "A/D", "A/A")


def normalise(m: np.ndarray) -> np.ndarray:
    m = np.clip(np.asarray(m, dtype=float), 0.0, None)
    total = m.sum()
    if total <= 0:
        raise ValueError("score matrix has no mass")
    return m / total


def outcome_masks(shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    i, j = np.indices(shape)
    return i > j, i == j, i < j


def matrix_to_1x2(m: np.ndarray) -> np.ndarray:
    m = normalise(m)
    home, draw, away = outcome_masks(m.shape)
    return np.array([m[home].sum(), m[draw].sum(), m[away].sum()])


def top_k_scores(m: np.ndarray, k: int = 2) -> list[dict]:
    """Most likely exact scores, ties broken by lower total goals then home goals."""
    m = normalise(m)
    cells = [(-m[i, j], i + j, i, i, j) for i in range(m.shape[0]) for j in range(m.shape[1])]
    cells.sort()
    return [{"score": f"{i}-{j}", "p": float(-neg)} for neg, _, _, i, j in cells[:k]]


def reweight_to_1x2(m: np.ndarray, target_1x2: np.ndarray) -> np.ndarray:
    """Rescale each outcome region (home win / draw / away win) of ``m`` so its
    1X2 margins equal ``target_1x2`` while keeping the within-region shape."""
    m = normalise(m)
    target = np.asarray(target_1x2, dtype=float)
    target = target / target.sum()
    current = matrix_to_1x2(m)
    out = m.copy()
    for mask, cur, tgt in zip(outcome_masks(m.shape), current, target):
        if cur > 0:
            out[mask] *= tgt / cur
    return normalise(out)


RHO_BOUNDS = (-0.25, 0.25)


def tau_matrix(lam: float, mu: float, rho: float, max_goals: int) -> np.ndarray:
    t = np.ones((max_goals + 1, max_goals + 1))
    t[0, 0] = 1.0 - lam * mu * rho
    t[0, 1] = 1.0 + lam * rho
    t[1, 0] = 1.0 + mu * rho
    t[1, 1] = 1.0 - rho
    return np.clip(t, 1e-10, None)


def dc_score_matrix(lam: float, mu: float, rho: float, max_goals: int = 10) -> np.ndarray:
    goals = np.arange(max_goals + 1)
    m = np.outer(poisson.pmf(goals, lam), poisson.pmf(goals, mu))
    return normalise(m * tau_matrix(lam, mu, rho, max_goals))
