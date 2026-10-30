"""Expected-goals (xG) providers. Not used by the MVP models yet.

Interface first, so a v1 xG-augmented Dixon-Coles (fit on xG or a goals/xG
blend) can plug in without touching callers.

* ``FootballDataXG``: football-data.co.uk added ``HxG``/``AxG`` columns from
  2026/27; available offline from the committed CSVs.
* ``UnderstatXG``: stub. Understat has no official API; its pages embed JSON
  in ``<script>`` tags which changes without notice.
"""

from __future__ import annotations

from typing import Protocol

import pandas as pd

from .football_data import load_all


class XGProvider(Protocol):
    def match_xg(self, season: str) -> pd.DataFrame:
        """Columns: date, home, away, hxg, axg (canonical team names)."""
        ...


class FootballDataXG:
    def match_xg(self, season: str) -> pd.DataFrame:
        df = load_all(seasons=[season])
        return df.loc[df["hxg"].notna(), ["date", "home", "away", "hxg", "axg"]].reset_index(drop=True)


class UnderstatXG:
    """TODO(v1): implement Understat scraping.

    Plan: GET https://understat.com/league/EPL/<start_year>, extract the
    ``datesData`` JSON from the page (JSON.parse('...') with \\x escapes),
    map team titles through ``data.teams.canonical_team``, cache raw pages
    under ``data/cache/understat/`` and rate-limit politely. Respect
    Understat's terms of use.
    """

    def match_xg(self, season: str) -> pd.DataFrame:
        raise NotImplementedError("Understat xG scraping is not implemented in the MVP; see UnderstatXG docstring.")
