"""Pure-market track: multi-bookmaker odds -> de-vigged consensus probabilities."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from .devig import devig, overround
from .score_matrix import HTFT_ORDER

_SYMBOL = {
    "1": "H", "h": "H", "home": "H",
    "x": "D", "d": "D", "draw": "D", "tie": "D",
    "2": "A", "a": "A", "away": "A",
}


@dataclass
class MarketEstimate:
    probs: np.ndarray
    n_books: int
    mean_overround: float
    books: list[str]


def _symbol(token: str) -> str | None:
    return _SYMBOL.get(token.strip().lower())


def parse_1x2_label(label: str | None, name: str | None = None) -> str | None:
    """Map a Fulltime Result selection to H/D/A. Sportmonks pre-match examples
    use ``Home``/``Draw``/``Away``; ``1``/``X``/``2`` is also accepted."""
    for text in (label, name):
        if text:
            sym = _symbol(text)
            if sym:
                return sym
    return None


def parse_htft_label(label: str | None, name: str | None = None) -> str | None:
    """Map an HT/FT selection to e.g. ``"H/D"``.

    The Sportmonks docs do not list market 29's selection labels, so accept
    the common spellings: ``1/X``, ``Home/Draw``, ``Home - Draw``, ``1-X``,
    ``HD``. Returns ``None`` when the label cannot be understood (the caller
    then ignores that selection rather than guessing)."""
    for text in (label, name):
        if not text:
            continue
        parts = [p for p in re.split(r"\s*[/\-]\s*|\s+", text.strip()) if p]
        if len(parts) == 1 and len(parts[0]) == 2:
            parts = list(parts[0])
        if len(parts) == 2:
            a, b = _symbol(parts[0]), _symbol(parts[1])
            if a and b:
                return f"{a}/{b}"
    return None


def consensus_1x2(book_odds: Mapping[str, Sequence[float]], method: str = "shin") -> MarketEstimate | None:
    """``book_odds``: bookmaker -> (home, draw, away) decimal odds.

    Each bookmaker is de-vigged separately (Shin by default), then the fair
    probabilities are averaged. Books with invalid prices are skipped."""
    probs, ors, used = [], [], []
    for book, odds in book_odds.items():
        try:
            probs.append(devig(odds, method))
            ors.append(overround(odds))
            used.append(str(book))
        except ValueError:
            continue
    if not probs:
        return None
    p = np.mean(probs, axis=0)
    return MarketEstimate(p / p.sum(), len(used), float(np.mean(ors)), used)


def consensus_htft(book_odds: Mapping[str, Mapping[str, float]], method: str = "multiplicative") -> MarketEstimate | None:
    """``book_odds``: bookmaker -> {"H/H": odds, ...}. Only bookmakers quoting
    all nine outcomes are used; multiplicative de-vig by default."""
    probs, ors, used = [], [], []
    for book, sel in book_odds.items():
        if any(k not in sel for k in HTFT_ORDER):
            continue
        odds = [sel[k] for k in HTFT_ORDER]
        try:
            probs.append(devig(odds, method))
            ors.append(overround(odds))
            used.append(str(book))
        except ValueError:
            continue
    if not probs:
        return None
    p = np.mean(probs, axis=0)
    return MarketEstimate(p / p.sum(), len(used), float(np.mean(ors)), used)


def htft_to_ft_1x2(p_htft: Sequence[float]) -> np.ndarray:
    p = np.asarray(p_htft, dtype=float).reshape(3, 3)
    return p.sum(axis=0)
