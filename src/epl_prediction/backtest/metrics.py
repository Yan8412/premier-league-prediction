"""Probabilistic scoring rules. ``outcomes`` are integer class indices (0=H, 1=D, 2=A)."""

from __future__ import annotations

import numpy as np

EPS = 1e-15


def log_loss(probs: np.ndarray, outcomes: np.ndarray) -> float:
    p = np.clip(np.asarray(probs, dtype=float), EPS, 1.0)
    y = np.asarray(outcomes, dtype=int)
    return float(-np.mean(np.log(p[np.arange(len(y)), y])))


def brier(probs: np.ndarray, outcomes: np.ndarray) -> float:
    """Multi-class Brier score (sum over classes, mean over matches; range 0-2)."""
    p = np.asarray(probs, dtype=float)
    y = np.asarray(outcomes, dtype=int)
    onehot = np.zeros_like(p)
    onehot[np.arange(len(y)), y] = 1.0
    return float(np.mean(((p - onehot) ** 2).sum(axis=1)))


def rps(probs: np.ndarray, outcomes: np.ndarray) -> float:
    """Ranked probability score for ordered outcomes H < D < A."""
    p = np.asarray(probs, dtype=float)
    y = np.asarray(outcomes, dtype=int)
    onehot = np.zeros_like(p)
    onehot[np.arange(len(y)), y] = 1.0
    cp, co = np.cumsum(p, axis=1)[:, :-1], np.cumsum(onehot, axis=1)[:, :-1]
    return float(np.mean(((cp - co) ** 2).sum(axis=1) / (p.shape[1] - 1)))


def accuracy(probs: np.ndarray, outcomes: np.ndarray) -> float:
    return float(np.mean(np.argmax(probs, axis=1) == np.asarray(outcomes, dtype=int)))


def summary(probs: np.ndarray, outcomes: np.ndarray) -> dict[str, float]:
    return {
        "n": int(len(outcomes)),
        "log_loss": log_loss(probs, outcomes),
        "brier": brier(probs, outcomes),
        "rps": rps(probs, outcomes),
        "accuracy": accuracy(probs, outcomes),
    }
