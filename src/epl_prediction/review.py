"""Self-upgrade hooks (MVP scaffold).

* weekly  : score stored predictions against finished fixtures (post-match review).
* monthly : re-run the walk-forward backtest, which refreshes the ensemble weight
            read by ``epl predict``.

TODO(v1): monthly re-tuning of xi / l2, calibration checks (reliability
curves), drift alerts, and an HT/FT football path to review alongside the market.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

from .backtest.metrics import summary
from .config import REPORTS_DIR
from .sportmonks.parse import FINISHED_STATES


def weekly_review(conn: sqlite3.Connection, days: int = 7, now: datetime | None = None,
                  out_dir: Path = REPORTS_DIR / "reviews") -> dict:
    now = now or datetime.now(timezone.utc)
    lo = int((now - timedelta(days=days)).timestamp())
    rows = conn.execute(
        f"""SELECT p.fixture_id, p.track, p.p_home, p.p_draw, p.p_away, f.ft_home, f.ft_away
            FROM predictions p JOIN fixtures f ON f.id = p.fixture_id
            WHERE f.state_id IN ({','.join(map(str, FINISHED_STATES))}) AND f.starting_at_timestamp >= ?
              AND p.id IN (SELECT MAX(id) FROM predictions GROUP BY fixture_id, track)""", (lo,)).fetchall()
    result: dict = {"window_days": days, "generated_at": now.isoformat(timespec="seconds"), "tracks": {}}
    by_track: dict[str, list] = {}
    for r in rows:
        y = 0 if r["ft_home"] > r["ft_away"] else 1 if r["ft_home"] == r["ft_away"] else 2
        by_track.setdefault(r["track"], []).append(([r["p_home"], r["p_draw"], r["p_away"]], y))
    for track, items in by_track.items():
        probs = np.array([p for p, _ in items])
        result["tracks"][track] = summary(probs, np.array([y for _, y in items]))
    if result["tracks"]:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / f"weekly_{now.date().isoformat()}.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def monthly_review(log=print) -> dict:
    from .backtest.walkforward import run_backtest, write_report

    result = run_backtest(log=log)
    md, js = write_report(result)
    return {"report_md": str(md), "report_json": str(js), "ensemble_weight_football": result.weights}
