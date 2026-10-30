"""football-data.co.uk EPL CSV loader (https://www.football-data.co.uk/englandm.php).

Files live in ``data/football_data/E0_<yyYY>.csv`` (e.g. ``E0_2425.csv``).
Columns used: Date, HomeTeam, AwayTeam, FTHG, FTAG, HTHG, HTAG, optional
HxG/AxG (present from 2026/27), and 1X2 odds ``<book>H/D/A`` (opening, i.e.
collected around Friday for weekend games) and ``<book>CH/CD/CA`` (closing).
The set of bookmakers changes between seasons, so it is detected per file.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import requests

from ..config import FOOTBALL_DATA_DIR

BASE_URL = "https://www.football-data.co.uk/mmz4281/{season}/E0.csv"
AGGREGATE_PREFIXES = {"Max", "Avg"}
EXCHANGE_PREFIXES = {"BF", "BFE"}  # exchanges: near-zero overround, kept out of the bookmaker consensus
RESULT_INDEX = {"H": 0, "D": 1, "A": 2}


def season_code(start_year: int) -> str:
    return f"{start_year % 100:02d}{(start_year + 1) % 100:02d}"


def download_season(start_year: int, dest_dir: Path = FOOTBALL_DATA_DIR, timeout: float = 30.0) -> Path:
    code = season_code(start_year)
    resp = requests.get(BASE_URL.format(season=code), timeout=timeout)
    resp.raise_for_status()
    dest_dir.mkdir(parents=True, exist_ok=True)
    path = dest_dir / f"E0_{code}.csv"
    path.write_bytes(resp.content)
    return path


def _book_prefixes(columns: Iterable[str], closing: bool) -> list[str]:
    cols = set(columns)
    suffix = "C" if closing else ""
    out = []
    for c in cols:
        m = re.fullmatch(rf"(.+){suffix}H", c)
        if not m:
            continue
        p = m.group(1)
        if not {f"{p}{suffix}D", f"{p}{suffix}A"} <= cols:
            continue
        if not closing and p.endswith("C") and {f"{p[:-1]}H", f"{p[:-1]}D", f"{p[:-1]}A"} <= cols:
            continue  # "B365C" is the closing variant of "B365", not a bookmaker
        if closing and f"{p}CCH" in cols:
            continue  # "VCH" is VC Bet's opening price, not closing odds of a book "V"
        if p in AGGREGATE_PREFIXES or p in EXCHANGE_PREFIXES or p.startswith("HT") or p.startswith("FT"):
            continue
        out.append(p)
    return sorted(out)


def _parse_dates(s: pd.Series) -> pd.Series:
    out = pd.to_datetime(s, format="%d/%m/%Y", errors="coerce")
    short = out.isna()
    if short.any():
        out[short] = pd.to_datetime(s[short], format="%d/%m/%y", errors="coerce")
    return out


def load_season_csv(path: Path) -> pd.DataFrame:
    raw = pd.read_csv(path, encoding="latin-1")
    raw = raw.dropna(subset=["HomeTeam", "AwayTeam"])
    m = re.search(r"(\d{4})", Path(path).stem)
    season = m.group(1) if m else Path(path).stem

    df = pd.DataFrame({
        "season": season,
        "date": _parse_dates(raw["Date"].astype(str)),
        "home": raw["HomeTeam"].str.strip(),
        "away": raw["AwayTeam"].str.strip(),
        "hg": pd.to_numeric(raw.get("FTHG"), errors="coerce"),
        "ag": pd.to_numeric(raw.get("FTAG"), errors="coerce"),
        "hthg": pd.to_numeric(raw.get("HTHG"), errors="coerce"),
        "htag": pd.to_numeric(raw.get("HTAG"), errors="coerce"),
        "hxg": pd.to_numeric(raw["HxG"], errors="coerce") if "HxG" in raw else np.nan,
        "axg": pd.to_numeric(raw["AxG"], errors="coerce") if "AxG" in raw else np.nan,
    })
    if "Time" in raw:
        df["time"] = raw["Time"].astype(str)

    for kind, closing in (("open", False), ("close", True)):
        books = _book_prefixes(raw.columns, closing)
        suffix = "C" if closing else ""
        odds = {}
        for b in books:
            trip = raw[[f"{b}{suffix}H", f"{b}{suffix}D", f"{b}{suffix}A"]].apply(pd.to_numeric, errors="coerce")
            odds[b] = trip.to_numpy()
        df[f"odds_{kind}"] = [
            {b: tuple(odds[b][i]) for b in books if np.all(np.isfinite(odds[b][i])) and np.all(odds[b][i] > 1)}
            for i in range(len(raw))
        ]
        avg = [f"Avg{suffix}H", f"Avg{suffix}D", f"Avg{suffix}A"]
        if set(avg) <= set(raw.columns):
            df[f"avg_{kind}"] = list(raw[avg].apply(pd.to_numeric, errors="coerce").to_numpy())

    df = df.dropna(subset=["date"]).reset_index(drop=True)
    played = df["hg"].notna() & df["ag"].notna()
    df["result"] = np.where(~played, -1,
                            np.where(df["hg"] > df["ag"], 0, np.where(df["hg"] == df["ag"], 1, 2)))
    return df


def load_all(data_dir: Path = FOOTBALL_DATA_DIR, seasons: Iterable[str] | None = None) -> pd.DataFrame:
    paths = sorted(Path(data_dir).glob("E0_*.csv"))
    if seasons is not None:
        wanted = set(seasons)
        paths = [p for p in paths if p.stem.split("_")[1] in wanted]
    if not paths:
        raise FileNotFoundError(f"no football-data CSVs found in {data_dir}")
    frames = [load_season_csv(p) for p in paths]
    df = pd.concat(frames, ignore_index=True)
    return df.sort_values(["date", "home"]).reset_index(drop=True)
