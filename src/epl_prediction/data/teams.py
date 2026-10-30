"""Canonical team names = football-data.co.uk spelling (e.g. "Man United")."""

from __future__ import annotations

import csv
import logging
import re
from functools import lru_cache
from importlib import resources

log = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _aliases() -> dict[str, str]:
    text = resources.files("epl_prediction").joinpath("team_names.csv").read_text(encoding="utf-8")
    return {row["alias"].strip().lower(): row["canonical"].strip() for row in csv.DictReader(text.splitlines())}


def canonical_team(name: str | None) -> str | None:
    if not name:
        return None
    key = name.strip().lower()
    aliases = _aliases()
    if key in aliases:
        return aliases[key]
    stripped = re.sub(r"\s+(f\.?c\.?|a\.?f\.?c\.?)$|^(a\.?f\.?c\.?)\s+", "", key).strip()
    if stripped in aliases:
        return aliases[stripped]
    return name.strip() if stripped == key else stripped.title()
