"""Expected-goals (xG) sources, merged into the football-track history.

Dixon-Coles uses xG where present (``DixonColesModel.xg_weight``): Poisson
targets become ``a * xG + (1 - a) * goals``.

Sources (all optional, merged by date + canonical home/away):

* ``football_data``: ``HxG``/``AxG`` columns in football-data.co.uk CSVs (from 2026/27).
* ``sportmonks``: ``include=xGFixture`` (type 5304) on fixtures; needs the
  Sportmonks *xG Basic* add-on. ``epl sync fixtures --with-xg``.
* ``understat``: **file import only**. understat.com's robots.txt is
  ``User-agent: * / Disallow: /`` (checked Oct 2026), so this project ships
  no crawler. If you have Understat match data obtained in a way its terms
  allow (e.g. saved manually from your browser, or a licensed copy), import
  it with ``epl xg import --understat FILE``. Accepted shape: Understat's
  match list JSON (``datesData``): ``[{"datetime", "isResult", "h": {"title"},
  "a": {"title"}, "goals": {"h", "a"}, "xG": {"h", "a"}}, ...]``.
* ``csv``: any CSV with columns date, home, away, hxg, axg (``epl xg import --csv``).

Imported files are normalised into ``data/xg/<source>_<from>_<to>.csv``
(columns date, home, away, hxg, axg, source).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Protocol

import pandas as pd

from ..config import DATA_DIR
from .football_data import load_all
from .teams import canonical_team

XG_DIR = DATA_DIR / "xg"
XG_COLUMNS = ["date", "home", "away", "hxg", "axg", "source"]


class XGProvider(Protocol):
    def match_xg(self) -> pd.DataFrame:
        """Columns: date, home, away, hxg, axg, source (canonical team names)."""
        ...


def _normalise(df: pd.DataFrame, source: str) -> pd.DataFrame:
    out = pd.DataFrame({
        "date": pd.to_datetime(df["date"]).dt.normalize(),
        "home": df["home"].map(canonical_team),
        "away": df["away"].map(canonical_team),
        "hxg": pd.to_numeric(df["hxg"], errors="coerce"),
        "axg": pd.to_numeric(df["axg"], errors="coerce"),
        "source": df["source"] if "source" in df else source,
    })
    return out.dropna(subset=["date", "home", "away", "hxg", "axg"]).reset_index(drop=True)


class FootballDataXG:
    def match_xg(self) -> pd.DataFrame:
        df = load_all()
        return _normalise(df.loc[df["hxg"].notna(), ["date", "home", "away", "hxg", "axg"]], "football_data")


class UnderstatXG:
    """Parse an Understat match-list JSON file. Never fetches from understat.com."""

    def __init__(self, path: Path):
        self.path = Path(path)

    def match_xg(self) -> pd.DataFrame:
        raw = json.loads(self.path.read_text(encoding="utf-8"))
        if isinstance(raw, dict):
            raw = raw.get("datesData") or raw.get("dates") or list(raw.values())
        rows = []
        for m in raw:
            if str(m.get("isResult", True)).lower() in ("false", "0"):
                continue
            xg = m.get("xG") or {}
            rows.append({"date": str(m["datetime"])[:10], "home": m["h"]["title"], "away": m["a"]["title"],
                         "hxg": xg.get("h"), "axg": xg.get("a")})
        return _normalise(pd.DataFrame(rows, columns=["date", "home", "away", "hxg", "axg"]), "understat")


class CsvXG:
    def __init__(self, path: Path, source: str = "csv"):
        self.path, self.source = Path(path), source

    def match_xg(self) -> pd.DataFrame:
        df = pd.read_csv(self.path)
        df.columns = [c.strip().lower() for c in df.columns]
        return _normalise(df, self.source)


def save_to_store(df: pd.DataFrame, source: str, xg_dir: Path = XG_DIR) -> Path:
    xg_dir.mkdir(parents=True, exist_ok=True)
    lo, hi = df["date"].min().date(), df["date"].max().date()
    path = xg_dir / f"{source}_{lo}_{hi}.csv"
    df[XG_COLUMNS].assign(date=df["date"].dt.date).to_csv(path, index=False)
    return path


def load_store(xg_dir: Path = XG_DIR) -> pd.DataFrame:
    files = sorted(Path(xg_dir).glob("*.csv")) if Path(xg_dir).exists() else []
    if not files:
        return pd.DataFrame(columns=XG_COLUMNS)
    return _normalise(pd.concat([pd.read_csv(f) for f in files], ignore_index=True), "store")


def attach_xg(history: pd.DataFrame, sources: Iterable[pd.DataFrame]) -> pd.DataFrame:
    """Fill ``hxg``/``axg`` in ``history`` from ``sources`` (first source wins), matched on
    date (±1 day, kickoff timezones differ between providers) and canonical teams."""
    hist = history.copy()
    for col in ("hxg", "axg"):
        if col not in hist:
            hist[col] = float("nan")
    hist["xg_source"] = hist.get("xg_source", pd.Series(index=hist.index, dtype=object))
    hist.loc[hist["hxg"].notna() & hist["xg_source"].isna(), "xg_source"] = "inline"
    for src in sources:
        if src is None or src.empty:
            continue
        lookup = {}
        for r in src.itertuples(index=False):
            for shift in (-1, 0, 1):
                lookup.setdefault((r.date + pd.Timedelta(days=shift), r.home, r.away), (r.hxg, r.axg, r.source))
        missing = hist["hxg"].isna() | hist["axg"].isna()
        for i in hist.index[missing]:
            hit = lookup.get((pd.Timestamp(hist.at[i, "date"]).normalize(), hist.at[i, "home"], hist.at[i, "away"]))
            if hit:
                hist.at[i, "hxg"], hist.at[i, "axg"], hist.at[i, "xg_source"] = hit
    return hist
