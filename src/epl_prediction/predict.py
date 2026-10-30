"""Assemble per-fixture predictions for the three tracks.

Output per track: p_1x2[3] (H, D, A), p_htft[9] (order HTFT_ORDER) or null,
top2_scores[{score, p} x 2], model_version, generated_at.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from . import __version__
from .config import EPL_LEAGUE_ID, MARKET_FT_1X2, MARKET_HT_FT, REPORTS_DIR
from .data.football_data import load_all
from .data.teams import canonical_team
from .data.xg import attach_xg
from .data.xg import load_store as load_xg_store
from .models.dixon_coles import DixonColesModel, dc_score_matrix, invert_1x2
from .contract import consistency_metrics
from .models.ensemble import log_pool
from .models.halftime import align_htft_to_1x2, htft_probs
from .models.market import consensus_1x2, consensus_htft, htft_to_ft_1x2, parse_1x2_label, parse_htft_label
from .models.score_matrix import HTFT_ORDER, matrix_to_1x2, reweight_to_1x2, top_k_scores
from .sportmonks.parse import FINISHED_STATES, NOT_STARTED_STATE

log = logging.getLogger(__name__)

USAGE_NOTE = "仅供参考/研究 (tip / research reference only) — no auto-betting."
DEFAULT_POOL_WEIGHT = 0.5


@dataclass
class FixtureInput:
    home: str
    away: str
    kickoff: str | None = None
    fixture_id: int | None = None
    odds_1x2: dict[str, tuple[float, float, float]] = field(default_factory=dict)
    odds_htft: dict[str, dict[str, float]] = field(default_factory=dict)
    odds_meta: dict[str, Any] = field(default_factory=dict)


# ------------------------------------------------------------------ history
def load_history(conn: sqlite3.Connection | None = None) -> pd.DataFrame:
    """football-data CSVs plus finished Sportmonks fixtures (canonical names), with xG
    attached from football-data, Sportmonks xGFixture and the ``data/xg`` store."""
    fd = load_all()
    hist = fd.loc[fd["result"] >= 0, ["date", "home", "away", "hg", "ag", "hthg", "htag", "hxg", "axg"]].copy()
    xg_sources = []
    if conn is not None:
        rows = conn.execute(
            f"SELECT starting_at, home_team, away_team, ft_home, ft_away, ht_home, ht_away, xg_home, xg_away FROM fixtures "
            f"WHERE league_id={EPL_LEAGUE_ID} AND state_id IN ({','.join(map(str, FINISHED_STATES))}) "
            f"AND ft_home IS NOT NULL").fetchall()
        if rows:
            sm = pd.DataFrame([{
                "date": pd.Timestamp(r["starting_at"][:10]), "home": canonical_team(r["home_team"]),
                "away": canonical_team(r["away_team"]), "hg": r["ft_home"], "ag": r["ft_away"],
                "hthg": r["ht_home"], "htag": r["ht_away"], "hxg": r["xg_home"], "axg": r["xg_away"]} for r in rows])
            num = ["hg", "ag", "hthg", "htag", "hxg", "axg"]
            sm[num] = sm[num].astype(float)
            xg_sources.append(sm.loc[sm["hxg"].notna() & sm["axg"].notna(),
                                     ["date", "home", "away", "hxg", "axg"]].assign(source="sportmonks"))
            hist = pd.concat([hist, sm], ignore_index=True)
            hist = hist.drop_duplicates(subset=["date", "home", "away"], keep="first")
    xg_sources.append(load_xg_store())
    hist = attach_xg(hist, xg_sources)
    return hist.sort_values("date").reset_index(drop=True)


def fit_football(history: pd.DataFrame, as_of: pd.Timestamp | None = None,
                 xg_weight: float | None = None) -> DixonColesModel:
    as_of = as_of or pd.Timestamp(datetime.now(timezone.utc).date())
    model = DixonColesModel() if xg_weight is None else DixonColesModel(xg_weight=xg_weight)
    return model.fit(history, as_of=as_of)


def load_pool_weight(report: Path = REPORTS_DIR / "backtest_epl.json", kind: str = "close") -> float:
    """Football weight fitted on historical out-of-sample log-loss (latest season in the report).

    ``kind="htft"`` needs stored football + market HT/FT predictions (written by
    ``epl review monthly``); until then the prior weight is used."""
    try:
        weights = json.loads(report.read_text(encoding="utf-8"))["ensemble_weight_football"][kind]
        return float(weights[max(weights)])
    except (OSError, KeyError, ValueError):
        log.info("no %r pool weight in %s; using prior %.2f", kind, report, DEFAULT_POOL_WEIGHT)
        return DEFAULT_POOL_WEIGHT


# ----------------------------------------------------------------- odds IO
def odds_from_db(conn: sqlite3.Connection, fixture_id: int) -> tuple[dict, dict, dict]:
    """Latest pre-kickoff snapshot per market -> (odds_1x2, odds_htft, meta)."""
    meta: dict[str, Any] = {}
    odds_1x2: dict[str, list] = {}
    odds_htft: dict[str, dict[str, float]] = {}
    for market_id in (MARKET_FT_1X2, MARKET_HT_FT):
        snap = conn.execute(
            "SELECT fetched_at, snapshot_tag FROM odds_snapshots WHERE fixture_id=? AND market_id=? "
            "AND snapshot_tag != 'post_kickoff' ORDER BY fetched_at DESC LIMIT 1", (fixture_id, market_id)).fetchone()
        if snap is None:
            continue
        meta[str(market_id)] = {"fetched_at": snap["fetched_at"], "snapshot_tag": snap["snapshot_tag"]}
        rows = conn.execute(
            "SELECT o.bookmaker_id, b.name AS bookmaker, o.label, o.name, o.value FROM odds_snapshots o "
            "LEFT JOIN bookmakers b ON b.id = o.bookmaker_id WHERE o.fixture_id=? AND o.market_id=? "
            "AND o.fetched_at=? AND (o.stopped IS NULL OR o.stopped=0)",
            (fixture_id, market_id, snap["fetched_at"])).fetchall()
        unparsed = 0
        for r in rows:
            book = r["bookmaker"] or str(r["bookmaker_id"])
            if market_id == MARKET_FT_1X2:
                sym = parse_1x2_label(r["label"], r["name"])
                if sym is None:
                    unparsed += 1
                    continue
                odds_1x2.setdefault(book, [None, None, None])["HDA".index(sym)] = r["value"]
            else:
                key = parse_htft_label(r["label"], r["name"])
                if key is None:
                    unparsed += 1
                    continue
                odds_htft.setdefault(book, {})[key] = r["value"]
        meta[str(market_id)]["unparsed_selections"] = unparsed
    complete_1x2 = {b: tuple(v) for b, v in odds_1x2.items() if all(x is not None for x in v)}
    return complete_1x2, odds_htft, meta


# ---------------------------------------------------------------- predict
def _r(x: np.ndarray | None, nd: int = 4) -> list[float] | None:
    return None if x is None else [round(float(v), nd) for v in x]


def _top2(m: np.ndarray) -> list[dict]:
    return [{"score": s["score"], "p": round(s["p"], 4)} for s in top_k_scores(m, 2)]


def predict_fixture(model: DixonColesModel, fx: FixtureInput, pool_weight: float = DEFAULT_POOL_WEIGHT,
                    devig_method: str = "shin", generated_at: str | None = None,
                    htft_weight: float = DEFAULT_POOL_WEIGHT) -> dict:
    generated_at = generated_at or datetime.now(timezone.utc).isoformat(timespec="seconds")
    home, away = canonical_team(fx.home), canonical_team(fx.away)

    # football track
    lam, mu = model.expected_goals(home, away)
    m_f = dc_score_matrix(lam, mu, model.rho, model.max_goals)
    p_f = matrix_to_1x2(m_f)
    h_f = htft_probs(lam, mu, model.ht_share_home, model.ht_share_away, model.rho_ht, m_f)
    f_notes: list[str] = []
    for side, team in (("home", home), ("away", away)):
        if team not in model.teams:
            f_notes.append(f"{side} team '{team}' unseen in training window: weak-team prior used")
    football = {
        "p_1x2": _r(p_f), "p_htft": _r(h_f), "top2_scores": _top2(m_f),
        "expected_goals": {"home": round(lam, 3), "away": round(mu, 3)},
        "expected_goals_first_half": {"home": round(lam * model.ht_share_home, 3),
                                      "away": round(mu * model.ht_share_away, 3)},
        "model_version": f"football-dixon-coles/{__version__} (xi={model.xi}, l2={model.l2}, "
                         f"xg_weight={model.xg_weight}, xg_matches={model.n_xg_matches})",
        "generated_at": generated_at, "notes": f_notes,
    }

    # market track
    est_1x2 = consensus_1x2(fx.odds_1x2, devig_method) if fx.odds_1x2 else None
    est_htft = consensus_htft(fx.odds_htft) if fx.odds_htft else None
    m_notes: list[str] = []
    p_m = None
    if est_1x2 is not None:
        p_m = est_1x2.probs
    elif est_htft is not None:
        p_m = htft_to_ft_1x2(est_htft.probs)
        m_notes.append("FT 1X2 odds missing: p_1x2 derived from HT/FT market margins")
    if est_htft is None:
        m_notes.append("HT/FT (market 29) odds unavailable or incomplete: p_htft=null")
    top2_m = None
    if p_m is not None:
        lam_m, mu_m = invert_1x2(p_m, rho=model.rho)
        top2_m = _top2(dc_score_matrix(lam_m, mu_m, model.rho, model.max_goals))
    else:
        m_notes.append("no usable odds: market track unavailable")
    market = {
        "available": p_m is not None,
        "p_1x2": _r(p_m), "p_htft": _r(est_htft.probs if est_htft else None), "top2_scores": top2_m,
        "n_bookmakers_1x2": est_1x2.n_books if est_1x2 else 0,
        "n_bookmakers_htft": est_htft.n_books if est_htft else 0,
        "mean_overround_1x2": round(est_1x2.mean_overround, 4) if est_1x2 else None,
        "odds_snapshot": fx.odds_meta or None,
        "model_version": f"market-{devig_method}-1x2+multiplicative-htft/{__version__}",
        "generated_at": generated_at, "notes": m_notes,
    }

    # ensemble track
    e_notes: list[str] = []
    if p_m is not None:
        p_e = log_pool([p_f, p_m], [pool_weight, 1.0 - pool_weight])
        weights = {"football": round(pool_weight, 4), "market": round(1.0 - pool_weight, 4)}
    else:
        p_e = p_f
        weights = {"football": 1.0, "market": 0.0}
        e_notes.append("market unavailable: ensemble falls back to football track")
    m_e = reweight_to_1x2(m_f, p_e)
    if est_htft is not None:
        h_e = log_pool([h_f, est_htft.probs], [htft_weight, 1.0 - htft_weight])
        weights_htft = {"football": round(htft_weight, 4), "market": round(1.0 - htft_weight, 4)}
    else:
        h_e = h_f
        weights_htft = {"football": 1.0, "market": 0.0}
        e_notes.append("market HT/FT unavailable: p_htft from football track only")
    h_e = align_htft_to_1x2(h_e, p_e)
    ensemble = {
        "p_1x2": _r(p_e), "p_htft": _r(h_e), "top2_scores": _top2(m_e),
        "weights": weights, "weights_htft": weights_htft,
        "model_version": f"ensemble-log-opinion-pool/{__version__}",
        "generated_at": generated_at, "notes": e_notes,
    }

    tracks = {"football": football, "market": market, "ensemble": ensemble}
    for t in tracks.values():
        t["consistency"] = consistency_metrics(t)
    return {
        "fixture_id": fx.fixture_id, "home": home, "away": away, "kickoff": fx.kickoff,
        "outcome_order_1x2": ["H", "D", "A"], "outcome_order_htft": list(HTFT_ORDER),
        "usage": USAGE_NOTE,
        "tracks": tracks,
    }


# ---------------------------------------------------------- fixture lookup
def fixture_inputs_from_db(conn: sqlite3.Connection, fixture_ids: list[int]) -> list[FixtureInput]:
    out = []
    for fid in fixture_ids:
        row = conn.execute("SELECT * FROM fixtures WHERE id=?", (fid,)).fetchone()
        if row is None:
            raise LookupError(f"fixture {fid} not in DB; run `epl sync fixture --fixture-id {fid}` first")
        o1, oh, meta = odds_from_db(conn, fid)
        out.append(FixtureInput(home=row["home_team"], away=row["away_team"], kickoff=row["starting_at"],
                                fixture_id=fid, odds_1x2=o1, odds_htft=oh, odds_meta=meta))
    return out


def next_matchday_fixture_ids(conn: sqlite3.Connection, now: datetime | None = None) -> list[int]:
    """All not-started fixtures in the round of the earliest upcoming fixture."""
    now_ts = int((now or datetime.now(timezone.utc)).timestamp())
    first = conn.execute(
        "SELECT round_id, season_id FROM fixtures WHERE league_id=? AND starting_at_timestamp >= ? "
        "AND (state_id IS NULL OR state_id=?) ORDER BY starting_at_timestamp LIMIT 1",
        (EPL_LEAGUE_ID, now_ts, NOT_STARTED_STATE)).fetchone()
    if first is None:
        return []
    if first["round_id"] is None:
        rows = conn.execute(
            "SELECT id FROM fixtures WHERE league_id=? AND starting_at_timestamp BETWEEN ? AND ? "
            "ORDER BY starting_at_timestamp", (EPL_LEAGUE_ID, now_ts, now_ts + 7 * 86400)).fetchall()
    else:
        rows = conn.execute(
            "SELECT id FROM fixtures WHERE league_id=? AND round_id=? AND (state_id IS NULL OR state_id=?) "
            "ORDER BY starting_at_timestamp", (EPL_LEAGUE_ID, first["round_id"], NOT_STARTED_STATE)).fetchall()
    return [r["id"] for r in rows]


def store_predictions(conn: sqlite3.Connection, preds: list[dict]) -> None:
    for p in preds:
        for track, t in p["tracks"].items():
            if t.get("p_1x2") is None:
                continue
            conn.execute(
                "INSERT INTO predictions (fixture_id, home_team, away_team, kickoff, track, model_version, generated_at, "
                "p_home, p_draw, p_away, payload_json) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (p["fixture_id"], p["home"], p["away"], p["kickoff"], track, t["model_version"], t["generated_at"],
                 *t["p_1x2"], json.dumps(t, ensure_ascii=False)))
    conn.commit()
