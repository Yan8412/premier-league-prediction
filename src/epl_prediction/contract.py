"""Output contract for ``epl predict``.

Every track carries: p_1x2[3] (H, D, A), p_htft[9] (H/H .. A/A), top2_scores
[{score, p} x 2], model_version, generated_at. Market fields may be null when
odds are missing (``available: false``); football and ensemble are never null.
"""

from __future__ import annotations

import re
from typing import Any

import numpy as np

REQUIRED_KEYS = ("p_1x2", "p_htft", "top2_scores", "model_version", "generated_at")
NULLABLE_TRACKS = {"market"}
SUM_TOL = 2e-3  # outputs are rounded to 4 decimals
SCORE_RE = re.compile(r"^\d+-\d+$")


def consistency_metrics(track: dict[str, Any]) -> dict[str, Any]:
    """How well a track's HT/FT agrees with its own 1X2 (FT margins of HT/FT vs p_1x2)."""
    p1, ph = track.get("p_1x2"), track.get("p_htft")
    out: dict[str, Any] = {"htft_ft_margin_max_abs_diff": None}
    if p1 is not None and ph is not None:
        ft = np.asarray(ph, dtype=float).reshape(3, 3).sum(axis=0)
        out["htft_ft_margin_max_abs_diff"] = round(float(np.abs(ft - np.asarray(p1)).max()), 4)
    return out


def validate_prediction(pred: dict[str, Any]) -> list[str]:
    """Return a list of contract violations (empty = OK)."""
    errors: list[str] = []
    for name, t in (pred.get("tracks") or {}).items():
        for key in REQUIRED_KEYS:
            if key not in t:
                errors.append(f"{name}: missing {key}")
        if name in NULLABLE_TRACKS and not t.get("available", True):
            continue
        for key, n in (("p_1x2", 3), ("p_htft", 9)):
            v = t.get(key)
            if v is None:
                if name not in NULLABLE_TRACKS:
                    errors.append(f"{name}: {key} is null")
                continue
            arr = np.asarray(v, dtype=float)
            if arr.shape != (n,):
                errors.append(f"{name}: {key} has length {arr.size}, expected {n}")
            elif np.any(arr < 0) or abs(arr.sum() - 1) > SUM_TOL:
                errors.append(f"{name}: {key} is not a probability vector (sum={arr.sum():.4f})")
        top2 = t.get("top2_scores")
        if top2 is not None and (len(top2) != 2 or any(not SCORE_RE.match(s.get("score", "")) for s in top2)
                                 or top2[0]["p"] < top2[1]["p"]):
            errors.append(f"{name}: top2_scores malformed")
        if name != "market":
            diff = consistency_metrics(t)["htft_ft_margin_max_abs_diff"]
            if diff is not None and diff > SUM_TOL:
                errors.append(f"{name}: HT/FT FT-margins differ from p_1x2 by {diff}")
    if set((pred.get("tracks") or {}).keys()) != {"football", "market", "ensemble"}:
        errors.append("tracks must be exactly football, market, ensemble")
    return errors
