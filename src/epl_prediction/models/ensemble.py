"""Logarithmic opinion pool: p_k ∝ prod_m p_{m,k} ** w_m, with sum(w) = 1."""

from __future__ import annotations

from typing import Sequence

import numpy as np

from ..backtest.metrics import log_loss

EPS = 1e-12


def log_pool(prob_vectors: Sequence[Sequence[float]], weights: Sequence[float]) -> np.ndarray:
    P = np.clip(np.asarray(prob_vectors, dtype=float), EPS, 1.0)
    w = np.asarray(weights, dtype=float)
    if P.shape[0] != w.shape[0]:
        raise ValueError("one weight per probability vector")
    if np.any(w < 0) or w.sum() <= 0:
        raise ValueError("weights must be non-negative and not all zero")
    w = w / w.sum()
    logp = (w[:, None] * np.log(P)).sum(axis=0)
    logp -= logp.max()
    p = np.exp(logp)
    return p / p.sum()


def log_pool_rows(p_a: np.ndarray, p_b: np.ndarray, w_a: float) -> np.ndarray:
    """Vectorised two-model pool over rows (n, k)."""
    la = np.log(np.clip(p_a, EPS, 1.0))
    lb = np.log(np.clip(p_b, EPS, 1.0))
    logp = w_a * la + (1.0 - w_a) * lb
    logp -= logp.max(axis=1, keepdims=True)
    p = np.exp(logp)
    return p / p.sum(axis=1, keepdims=True)


def fit_pool_weight(p_football: np.ndarray, p_market: np.ndarray, outcomes: np.ndarray,
                    grid: np.ndarray | None = None) -> tuple[float, float]:
    """Weight on the football model minimising historical log-loss.

    Returns ``(w_football, log_loss)``; ``w_market = 1 - w_football``."""
    grid = np.linspace(0.0, 1.0, 101) if grid is None else grid
    best_w, best_ll = 0.5, np.inf
    for w in grid:
        ll = log_loss(log_pool_rows(p_football, p_market, float(w)), outcomes)
        if ll < best_ll:
            best_w, best_ll = float(w), ll
    return best_w, float(best_ll)
