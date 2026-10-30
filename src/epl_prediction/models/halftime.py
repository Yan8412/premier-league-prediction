"""Football-track HT/FT (path A): split expected goals into halves.

First half:  (HT_h, HT_a) ~ DC(lambda * s_h, mu * s_a, rho_ht)
Second half: independent Poisson(lambda * (1 - s_h)), Poisson(mu * (1 - s_a))
FT = HT + second half.

``s_h`` / ``s_a`` (first-half share of goals, the "half tempo") and
``rho_ht`` are fitted alongside the full-time Dixon-Coles parameters, sharing
its attack/defence. The half process supplies P(HT result | FT score); the
FT score marginal is taken from the full-time Dixon-Coles matrix, so the
returned HT/FT vector sums over HT exactly to the model's 1X2 and is
consistent with its TOP2 scores.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.stats import poisson

from .score_matrix import HTFT_ORDER, RHO_BOUNDS, dc_score_matrix, normalise, outcome_masks

DEFAULT_FIRST_HALF_SHARE = 0.45


def htft_label(ht_home: int, ht_away: int, ft_home: int, ft_away: int) -> str:
    def sym(h: int, a: int) -> str:
        return "H" if h > a else "D" if h == a else "A"

    return f"{sym(ht_home, ht_away)}/{sym(ft_home, ft_away)}"


def htft_index(ht_home: int, ht_away: int, ft_home: int, ft_away: int) -> int:
    return HTFT_ORDER.index(htft_label(ht_home, ht_away, ft_home, ft_away))


def htft_probs(lam: float, mu: float, share_home: float, share_away: float, rho_ht: float,
               ft_matrix: np.ndarray) -> np.ndarray:
    """9-vector in ``HTFT_ORDER`` (H/H, H/D, ..., A/A)."""
    ft = normalise(ft_matrix)
    g_max = ft.shape[0] - 1
    first = dc_score_matrix(lam * share_home, mu * share_away, rho_ht, g_max)
    goals = np.arange(g_max + 1)
    second = np.outer(poisson.pmf(goals, lam * (1 - share_home)), poisson.pmf(goals, mu * (1 - share_away)))

    # by_ht[c, H, A] = P(HT result c, FT score H-A) under the half process
    by_ht = np.zeros((3, g_max + 1, g_max + 1))
    for h1 in range(g_max + 1):
        for a1 in range(g_max + 1):
            c = 0 if h1 > a1 else 1 if h1 == a1 else 2
            by_ht[c, h1:, a1:] += first[h1, a1] * second[: g_max + 1 - h1, : g_max + 1 - a1]
    cond = by_ht / np.clip(by_ht.sum(axis=0, keepdims=True), 1e-300, None)
    joint = cond * ft[None]
    masks = outcome_masks(ft.shape)
    out = np.array([[joint[c][m].sum() for m in masks] for c in range(3)]).reshape(9)
    return out / out.sum()


def fit_half_params(lam: np.ndarray, mu: np.ndarray, hg: np.ndarray, ag: np.ndarray,
                    hthg: np.ndarray, htag: np.ndarray, w: np.ndarray) -> tuple[float, float, float]:
    """Weighted first-half goal shares and the HT low-score correlation rho_ht.

    The shares are the MLE of a binomial split of each side's goals; rho_ht
    maximises the weighted DC tau term on half-time scores."""
    ok = ~(np.isnan(hthg) | np.isnan(htag) | np.isnan(hg) | np.isnan(ag))
    if ok.sum() < 50:
        return DEFAULT_FIRST_HALF_SHARE, DEFAULT_FIRST_HALF_SHARE, 0.0
    lam, mu, w = lam[ok], mu[ok], w[ok]
    hg, ag, x, y = hg[ok], ag[ok], hthg[ok], htag[ok]
    share_h = float(np.clip((w @ x) / max(w @ hg, 1e-9), 0.2, 0.8))
    share_a = float(np.clip((w @ y) / max(w @ ag, 1e-9), 0.2, 0.8))
    l1, m1 = lam * share_h, mu * share_a
    is00, is01, is10, is11 = (x == 0) & (y == 0), (x == 0) & (y == 1), (x == 1) & (y == 0), (x == 1) & (y == 1)

    def nll(rho: float) -> float:
        tau = np.ones_like(l1)
        tau[is00] = 1 - l1[is00] * m1[is00] * rho
        tau[is01] = 1 + l1[is01] * rho
        tau[is10] = 1 + m1[is10] * rho
        tau[is11] = 1 - rho
        return float(-(w @ np.log(np.clip(tau, 1e-10, None))))

    rho_ht = float(minimize_scalar(nll, bounds=RHO_BOUNDS, method="bounded").x)
    return share_h, share_a, rho_ht


def align_htft_to_1x2(p_htft, p_1x2) -> np.ndarray:
    """Rescale each FT column of the 3x3 HT/FT grid so its FT margins equal
    ``p_1x2`` (keeps P(HT result | FT result) unchanged)."""
    grid = np.asarray(p_htft, dtype=float).reshape(3, 3)
    target = np.asarray(p_1x2, dtype=float)
    col = grid.sum(axis=0)
    grid = grid * np.where(col > 0, target / np.where(col > 0, col, 1), 0)
    return (grid / grid.sum()).reshape(9)
