"""Self-upgrade cadence (run on the operator box, where the DB lives).

weekly  (``epl review weekly``): score the latest *pre-kickoff* stored
        prediction of every finished fixture in the window, per track:
        1X2 log-loss / Brier / RPS / accuracy, HT/FT 9-class log-loss /
        accuracy, TOP2 exact-score hit. Writes reports/reviews/weekly_<date>.{json,md}.
monthly (``epl review monthly``): re-runs the walk-forward backtest (refreshes
        reports/backtest_epl.* and the backtest 1X2 pool weight), fits *live*
        pool weights (1X2 and HT/FT) on all stored pre-kickoff predictions with
        known results, writes reports/ensemble_weights.json (read by
        ``epl predict``) and reports/reviews/monthly_<date>.{json,md}.

Live weights are only used once at least ``MIN_LIVE_MATCHES`` fixtures have
both football and market predictions; until then predict falls back to the
backtest weight (1X2) or the 0.5 prior (HT/FT).

TODO(v1): re-tune xi / l2 / xg_weight monthly, calibration curves, drift alerts.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np

from .backtest.metrics import accuracy, brier, log_loss, summary
from .config import LIVE_WEIGHTS_PATH, REPORTS_DIR
from .models.ensemble import fit_pool_weight
from .models.halftime import htft_index
from .sportmonks.parse import FINISHED_STATES

MIN_LIVE_MATCHES = 50
REVIEWS_DIR = REPORTS_DIR / "reviews"


def _parse_dt(value: str) -> datetime:
    dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def scored_predictions(conn: sqlite3.Connection, since: datetime | None = None,
                       until: datetime | None = None) -> list[dict[str, Any]]:
    """Latest prediction made before kickoff per (fixture, track), joined with the result."""
    lo = int(since.timestamp()) if since else 0
    hi = int(until.timestamp()) if until else 2**62
    rows = conn.execute(
        f"""SELECT p.id, p.fixture_id, p.track, p.generated_at, p.p_home, p.p_draw, p.p_away, p.payload_json,
                   f.starting_at_timestamp AS ko, f.ft_home, f.ft_away, f.ht_home, f.ht_away
            FROM predictions p JOIN fixtures f ON f.id = p.fixture_id
            WHERE f.state_id IN ({','.join(map(str, FINISHED_STATES))}) AND f.ft_home IS NOT NULL
              AND f.starting_at_timestamp BETWEEN ? AND ?
            ORDER BY p.id""", (lo, hi)).fetchall()
    latest: dict[tuple[int, str], sqlite3.Row] = {}
    for r in rows:
        if _parse_dt(r["generated_at"]).timestamp() < r["ko"]:
            latest[(r["fixture_id"], r["track"])] = r
    out = []
    for (fid, track), r in latest.items():
        payload = json.loads(r["payload_json"])
        hg, ag = r["ft_home"], r["ft_away"]
        has_ht = r["ht_home"] is not None and r["ht_away"] is not None
        out.append({
            "fixture_id": fid, "track": track, "generated_at": r["generated_at"],
            "p_1x2": [r["p_home"], r["p_draw"], r["p_away"]], "p_htft": payload.get("p_htft"),
            "top2": [s["score"] for s in payload.get("top2_scores") or []],
            "y_1x2": 0 if hg > ag else 1 if hg == ag else 2,
            "y_htft": htft_index(r["ht_home"], r["ht_away"], hg, ag) if has_ht else None,
            "score": f"{hg}-{ag}",
        })
    return out


def _metrics(items: list[dict]) -> dict:
    m = summary(np.array([i["p_1x2"] for i in items]), np.array([i["y_1x2"] for i in items]))
    with_top2 = [i for i in items if i["top2"]]
    m["top2_hit"] = float(np.mean([i["score"] in i["top2"] for i in with_top2])) if with_top2 else None
    ht = [i for i in items if i["p_htft"] is not None and i["y_htft"] is not None]
    if ht:
        p, y = np.array([i["p_htft"] for i in ht]), np.array([i["y_htft"] for i in ht])
        m["htft"] = {"n": len(ht), "log_loss": log_loss(p, y), "brier": brier(p, y), "accuracy": accuracy(p, y)}
    else:
        m["htft"] = None
    return m


def _metrics_by_track(items: list[dict]) -> dict[str, dict]:
    tracks: dict[str, list] = {}
    for i in items:
        tracks.setdefault(i["track"], []).append(i)
    return {t: _metrics(v) for t, v in sorted(tracks.items())}


def _fmt(x: float | None, nd: int = 4) -> str:
    return "-" if x is None else f"{x:.{nd}f}"


def _metrics_md(tracks: dict[str, dict]) -> list[str]:
    lines = ["| track | n | 1X2 log-loss | Brier | accuracy | TOP2 hit | HT/FT n | HT/FT log-loss | HT/FT acc |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for t, m in tracks.items():
        h = m["htft"] or {}
        lines.append(f"| {t} | {m['n']} | {_fmt(m['log_loss'])} | {_fmt(m['brier'])} | {_fmt(m['accuracy'], 3)} | "
                     f"{_fmt(m['top2_hit'], 3)} | {h.get('n', 0)} | {_fmt(h.get('log_loss'))} | "
                     f"{_fmt(h.get('accuracy'), 3)} |")
    return lines


def _write(out_dir: Path, stem: str, payload: dict, md_lines: list[str]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{stem}.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (out_dir / f"{stem}.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")


def weekly_review(conn: sqlite3.Connection, days: int = 7, now: datetime | None = None,
                  out_dir: Path = REVIEWS_DIR) -> dict:
    now = now or datetime.now(timezone.utc)
    items = scored_predictions(conn, now - timedelta(days=days), now)
    result: dict = {"window_days": days, "generated_at": now.isoformat(timespec="seconds"),
                    "n_fixtures": len({i["fixture_id"] for i in items}), "tracks": _metrics_by_track(items)}
    if result["tracks"]:
        md = [f"# 每周赛后复盘 weekly review — {now.date()}", "",
              f"窗口：最近 {days} 天已完赛、且开球前有预测的比赛 {result['n_fixtures']} 场。", "",
              *_metrics_md(result["tracks"]), "", "仅供研究/参考，不做自动下注。"]
        _write(out_dir, f"weekly_{now.date().isoformat()}", result, md)
    return result


def fit_live_weights(conn: sqlite3.Connection, min_matches: int = MIN_LIVE_MATCHES,
                     now: datetime | None = None) -> dict:
    """Football weight of the log pool, fit on stored pre-kickoff predictions."""
    items = scored_predictions(conn)
    by_fixture: dict[int, dict[str, dict]] = {}
    for i in items:
        by_fixture.setdefault(i["fixture_id"], {})[i["track"]] = i
    pairs = [(v["football"], v["market"]) for v in by_fixture.values() if "football" in v and "market" in v]
    out: dict = {"fitted_at": (now or datetime.now(timezone.utc)).isoformat(timespec="seconds"),
                 "min_matches": min_matches}
    for kind, key, ykey in (("1x2", "p_1x2", "y_1x2"), ("htft", "p_htft", "y_htft")):
        usable = [(f, m) for f, m in pairs if f[key] is not None and m[key] is not None and f[ykey] is not None]
        entry: dict = {"n": len(usable), "w_football": None}
        if len(usable) >= min_matches:
            pf = np.array([f[key] for f, _ in usable])
            pm = np.array([m[key] for _, m in usable])
            y = np.array([f[ykey] for f, _ in usable])
            w, ll = fit_pool_weight(pf, pm, y)
            entry.update({"w_football": w, "log_loss_pooled": ll,
                          "log_loss_football": log_loss(pf, y), "log_loss_market": log_loss(pm, y)})
        else:
            entry["reason"] = f"need >= {min_matches} fixtures with football + market predictions"
        out[kind] = entry
    return out


def monthly_review(conn: sqlite3.Connection | None = None, now: datetime | None = None,
                   out_dir: Path = REVIEWS_DIR, weights_path: Path = LIVE_WEIGHTS_PATH,
                   run_backtest_: bool = True, log=print) -> dict:
    now = now or datetime.now(timezone.utc)
    result: dict = {"generated_at": now.isoformat(timespec="seconds")}
    if run_backtest_:
        from .backtest.walkforward import run_backtest, track_metrics, write_report

        bt = run_backtest(log=log)
        md, js = write_report(bt)
        result["backtest"] = {"report_md": str(md), "report_json": str(js),
                              "ensemble_weight_football": bt.weights,
                              "overall": {k: v["log_loss"] for k, v in track_metrics(bt.predictions).items()}}
    if conn is not None:
        live = fit_live_weights(conn, now=now)
        weights_path.parent.mkdir(parents=True, exist_ok=True)
        weights_path.write_text(json.dumps(live, indent=2) + "\n", encoding="utf-8")
        result["live_weights"] = live
        result["last_30_days"] = _metrics_by_track(scored_predictions(conn, now - timedelta(days=30), now))

    md = [f"# 月度复盘 monthly review — {now.date()}", ""]
    if "backtest" in result:
        md += ["## 走步回测（已刷新 reports/backtest_epl.md）", "",
               "| track | log-loss |", "|---|---:|",
               *[f"| {k} | {v:.4f} |" for k, v in result["backtest"]["overall"].items()], ""]
    if "live_weights" in result:
        md += ["## 线上集成权重（reports/ensemble_weights.json，epl predict 优先使用）", "",
               "| 市场 | 样本 | w_football | 说明 |", "|---|---:|---:|---|"]
        for kind in ("1x2", "htft"):
            e = result["live_weights"][kind]
            md.append(f"| {kind} | {e['n']} | {_fmt(e['w_football'], 2)} | {e.get('reason', 'fitted on live log-loss')} |")
        md += ["", "## 最近 30 天线上表现", "", *_metrics_md(result["last_30_days"]), ""]
    md += ["仅供研究/参考，不做自动下注。"]
    _write(out_dir, f"monthly_{now.date().isoformat()}", result, md)
    return result
