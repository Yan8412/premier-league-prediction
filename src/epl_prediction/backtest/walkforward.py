"""Walk-forward backtest on football-data.co.uk EPL CSVs.

For every calendar week (Mon-Sun) in the test seasons:
  1. fit Dixon-Coles on all matches strictly before that Monday (time-decayed);
  2. predict that week's matches (football track);
  3. de-vig the bookmakers' closing (and opening) 1X2 odds -> market track;
  4. pool both with the log opinion pool. The pool weight for season S is fit
     on the out-of-sample predictions of all earlier test seasons only
     (first test season uses the prior weight 0.5).
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .. import __version__
from ..config import REPORTS_DIR
from ..data.football_data import load_all
from ..data.xg import attach_xg
from ..data.xg import load_store as load_xg_store
from ..models.dixon_coles import DEFAULT_XI, DixonColesModel, dc_score_matrix, invert_1x2
from ..models.ensemble import fit_pool_weight, log_pool_rows
from ..models.halftime import htft_index, htft_probs
from ..models.market import consensus_1x2
from ..models.score_matrix import matrix_to_1x2, reweight_to_1x2, top_k_scores
from .metrics import accuracy, brier, log_loss, summary

TRACKS = ("football", "market_close", "ensemble_close", "market_open", "ensemble_open")


@dataclass
class BacktestConfig:
    test_seasons: list[str] | None = None  # default: every season except the first (warm-up)
    xi: float = DEFAULT_XI
    l2: float = 1e-2
    xg_weight: float = 0.5
    max_days: int = 1100
    devig_method: str = "shin"
    prior_weight: float = 0.5
    data_dir: Path | None = None


@dataclass
class BacktestResult:
    config: BacktestConfig
    predictions: pd.DataFrame
    weights: dict[str, dict[str, float]] = field(default_factory=dict)
    runtime_s: float = 0.0
    n_xg_matches: int = 0


def _market_probs(odds_dicts: pd.Series, method: str) -> np.ndarray:
    out = np.full((len(odds_dicts), 3), np.nan)
    for i, books in enumerate(odds_dicts):
        est = consensus_1x2(books, method) if books else None
        if est is not None:
            out[i] = est.probs
    return out


def _top2_hit(matrix: np.ndarray, hg: int, ag: int) -> bool:
    return f"{int(hg)}-{int(ag)}" in {s["score"] for s in top_k_scores(matrix, 2)}


def run_backtest(cfg: BacktestConfig | None = None, log=print) -> BacktestResult:
    cfg = cfg or BacktestConfig()
    t0 = time.time()
    df = load_all(cfg.data_dir) if cfg.data_dir else load_all()
    seasons = sorted(df["season"].unique())
    test_seasons = cfg.test_seasons or seasons[1:]
    played = attach_xg(df[df["result"] >= 0].copy(), [load_xg_store()])

    test = played[played["season"].isin(test_seasons)].copy()
    test["week"] = test["date"] - pd.to_timedelta(test["date"].dt.weekday, unit="D")
    log(f"[backtest] {len(played)} played matches, testing {len(test)} in seasons {test_seasons}")

    rows = []
    model: DixonColesModel | None = None
    weeks = sorted(test["week"].unique())
    for k, week in enumerate(weeks):
        model = DixonColesModel(xi=cfg.xi, l2=cfg.l2, xg_weight=cfg.xg_weight).fit(
            played, as_of=week, max_days=cfg.max_days, warm_start=model)
        block = test[test["week"] == week]
        for _, r in block.iterrows():
            lam, mu = model.expected_goals(r["home"], r["away"])
            m = dc_score_matrix(lam, mu, model.rho, model.max_goals)
            rows.append({
                "season": r["season"], "date": r["date"], "home": r["home"], "away": r["away"],
                "hg": int(r["hg"]), "ag": int(r["ag"]), "result": int(r["result"]),
                "lam": lam, "mu": mu, "rho": model.rho,
                "home_known": r["home"] in model.teams, "away_known": r["away"] in model.teams,
                "matrix": m, "odds_close": r["odds_close"], "odds_open": r["odds_open"],
                "htft": htft_probs(lam, mu, model.ht_share_home, model.ht_share_away, model.rho_ht, m),
                "htft_class": (htft_index(int(r["hthg"]), int(r["htag"]), int(r["hg"]), int(r["ag"]))
                               if pd.notna(r["hthg"]) and pd.notna(r["htag"]) else -1),
            })
        if (k + 1) % 20 == 0 or k == len(weeks) - 1:
            log(f"[backtest] week {k + 1}/{len(weeks)} ({pd.Timestamp(week).date()}) fitted on {model.n_matches} matches")

    pred = pd.DataFrame(rows)
    pred_f = np.vstack([matrix_to_1x2(m) for m in pred["matrix"]])
    p_close = _market_probs(pred["odds_close"], cfg.devig_method)
    p_open = _market_probs(pred["odds_open"], cfg.devig_method)
    pred[["f_H", "f_D", "f_A"]] = pred_f
    pred[["mc_H", "mc_D", "mc_A"]] = p_close
    pred[["mo_H", "mo_D", "mo_A"]] = p_open
    pred["n_books_close"] = [len(d) for d in pred["odds_close"]]
    pred["top2_football"] = [_top2_hit(m, hg, ag) for m, hg, ag in zip(pred["matrix"], pred["hg"], pred["ag"])]

    weights: dict[str, dict[str, float]] = {}
    for kind, cols in (("close", ["mc_H", "mc_D", "mc_A"]), ("open", ["mo_H", "mo_D", "mo_A"])):
        ens = np.full((len(pred), 3), np.nan)
        weights[kind] = {}
        for s in test_seasons:
            hist = pred[(pred["season"] < s) & pred[cols[0]].notna()]
            if len(hist) >= 100:
                w, _ = fit_pool_weight(hist[["f_H", "f_D", "f_A"]].to_numpy(), hist[cols].to_numpy(), hist["result"].to_numpy())
            else:
                w = cfg.prior_weight
            weights[kind][s] = w
            idx = (pred["season"] == s).to_numpy() & pred[cols[0]].notna().to_numpy()
            ens[idx] = log_pool_rows(pred_f[idx], pred.loc[idx, cols].to_numpy(), w)
        pred[[f"e{kind[0]}_H", f"e{kind[0]}_D", f"e{kind[0]}_A"]] = ens

    hits_m, hits_e = [], []
    for i, r in pred.iterrows():
        if np.isnan(r["mc_H"]):
            hits_m.append(np.nan)
            hits_e.append(np.nan)
            continue
        lam_m, mu_m = invert_1x2([r["mc_H"], r["mc_D"], r["mc_A"]], rho=r["rho"])
        hits_m.append(_top2_hit(dc_score_matrix(lam_m, mu_m, r["rho"]), r["hg"], r["ag"]))
        m_e = reweight_to_1x2(r["matrix"], np.array([r["ec_H"], r["ec_D"], r["ec_A"]]))
        hits_e.append(_top2_hit(m_e, r["hg"], r["ag"]))
    pred["top2_market_close"] = hits_m
    pred["top2_ensemble_close"] = hits_e

    pred[[f"fh_{k}" for k in range(9)]] = np.vstack(pred["htft"].to_numpy())
    # HT/FT baseline: class frequencies of all earlier played matches (expanding window).
    hist_cls = played.dropna(subset=["hthg", "htag"])
    hist_idx = np.array([htft_index(*map(int, v)) for v in hist_cls[["hthg", "htag", "hg", "ag"]].to_numpy()])
    base = np.zeros((len(pred), 9))
    for s in test_seasons:
        prior = hist_idx[(hist_cls["season"] < s).to_numpy()]
        freq = (np.bincount(prior, minlength=9) + 1) / (len(prior) + 9)
        base[(pred["season"] == s).to_numpy()] = freq
    pred[[f"bh_{k}" for k in range(9)]] = base

    pred = pred.drop(columns=["matrix", "odds_close", "odds_open", "htft"])
    n_xg = int((played["hxg"].notna() & played["axg"].notna()).sum())
    return BacktestResult(cfg, pred, weights, time.time() - t0, n_xg)


def htft_metrics(pred: pd.DataFrame) -> dict[str, dict]:
    """Football HT/FT (9-class) vs the expanding base-rate baseline."""
    sub = pred[pred["htft_class"] >= 0]
    y = sub["htft_class"].to_numpy()
    out = {}
    for name, prefix in (("football", "fh"), ("base_rate", "bh")):
        probs = sub[[f"{prefix}_{k}" for k in range(9)]].to_numpy()
        out[name] = {"n": int(len(y)), "log_loss": log_loss(probs, y), "brier": brier(probs, y),
                     "accuracy": accuracy(probs, y)}
    return out


_TRACK_COLS = {
    "football": ("f", "top2_football"),
    "market_close": ("mc", "top2_market_close"),
    "ensemble_close": ("ec", "top2_ensemble_close"),
    "market_open": ("mo", None),
    "ensemble_open": ("eo", None),
}


def track_metrics(pred: pd.DataFrame, common_only: bool = True) -> dict[str, dict]:
    """Metrics per track. With ``common_only`` every track is scored on the same
    matches (those with both closing and opening odds) so numbers are comparable."""
    mask = np.ones(len(pred), dtype=bool)
    if common_only:
        for prefix, _ in _TRACK_COLS.values():
            mask &= pred[f"{prefix}_H"].notna().to_numpy()
    sub = pred[mask]
    y = sub["result"].to_numpy()
    out = {}
    for track, (prefix, top2) in _TRACK_COLS.items():
        stats = summary(sub[[f"{prefix}_H", f"{prefix}_D", f"{prefix}_A"]].to_numpy(), y)
        stats["top2_hit"] = float(sub[top2].astype(float).mean()) if top2 else None
        out[track] = stats
    return out


def write_report(result: BacktestResult, out_dir: Path = REPORTS_DIR, name: str = "backtest_epl") -> tuple[Path, Path]:
    pred = result.predictions
    overall = track_metrics(pred)
    per_season = {s: track_metrics(g) for s, g in pred.groupby("season")}
    unseen = pred[~(pred["home_known"] & pred["away_known"])]
    base_rates = pred["result"].value_counts(normalize=True).sort_index().tolist()
    cfg = result.config
    payload = {
        "model_version": f"epl-prediction {__version__}",
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "data": "football-data.co.uk E0 CSVs in data/football_data/",
        "config": {"test_seasons": sorted(pred["season"].unique().tolist()), "xi_per_day": cfg.xi, "l2": cfg.l2, "xg_weight": cfg.xg_weight,
                   "max_days": cfg.max_days, "devig_method_1x2": cfg.devig_method, "prior_weight": cfg.prior_weight},
        "n_matches": int(len(pred)),
        "n_training_matches_with_xg": result.n_xg_matches,
        "n_matches_scored_all_tracks": overall["football"]["n"],
        "n_matches_with_unseen_team": int(len(unseen)),
        "base_rates_HDA": base_rates,
        "ensemble_weight_football": result.weights,
        "overall": overall,
        "per_season": per_season,
        "htft_overall": htft_metrics(pred),
        "htft_per_season": {s: htft_metrics(g) for s, g in pred.groupby("season")},
        "runtime_s": round(result.runtime_s, 1),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"{name}.json"
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    def fmt(x, nd=4):
        return "-" if x is None else f"{x:.{nd}f}"

    lines = [
        "# EPL 走步回测报告 (Walk-forward backtest)",
        "",
        f"- 生成时间 generated_at: `{payload['generated_at']}`  模型版本: `{payload['model_version']}`",
        f"- 数据: football-data.co.uk E0（`data/football_data/`），测试赛季 {', '.join(payload['config']['test_seasons'])}",
        f"- 每周（周一为界）用此前全部比赛重拟合 Dixon-Coles：xi={cfg.xi}/天，L2={cfg.l2}，窗口 {cfg.max_days} 天，"
        f"xG 混合权重 {cfg.xg_weight}（有 xG 的比赛用 a·xG+(1-a)·进球 作为泊松目标；"
        f"训练集中有 xG 的比赛 {payload['n_training_matches_with_xg']} 场）",
        f"- 市场：各家庄 1X2 赔率逐家 {cfg.devig_method} 去水后取平均；closing=收盘赔率，open=开盘（周五前后采集）",
        "- 集成：对数意见池 p ∝ p_football^w · p_market^(1-w)，w 仅用更早赛季的样本外预测最小化 log-loss 拟合",
        f"- 共 {payload['n_matches']} 场；三轨同口径评分 {payload['n_matches_scored_all_tracks']} 场；"
        f"含升班/未见球队的比赛 {payload['n_matches_with_unseen_team']} 场",
        f"- 基准：均匀分布 log-loss = {np.log(3):.4f}；实际赛果频率 H/D/A = "
        + " / ".join(f"{x:.3f}" for x in base_rates),
        "",
        "## 总体 (Overall)",
        "",
        "| track | n | log-loss | Brier | RPS | accuracy | TOP2 exact-score hit |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for t, s in overall.items():
        lines.append(f"| {t} | {s['n']} | {fmt(s['log_loss'])} | {fmt(s['brier'])} | {fmt(s['rps'])} | "
                     f"{fmt(s['accuracy'], 3)} | {fmt(s['top2_hit'], 3)} |")
    lines += ["", "## 分赛季 (Per season, log-loss / TOP2 hit)", "",
              "| season | n | football | market_close | ensemble_close | w_football(close) | TOP2 football | TOP2 ensemble |",
              "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for s, m in per_season.items():
        lines.append(f"| {s} | {m['football']['n']} | {fmt(m['football']['log_loss'])} | "
                     f"{fmt(m['market_close']['log_loss'])} | {fmt(m['ensemble_close']['log_loss'])} | "
                     f"{fmt(result.weights['close'].get(s), 2)} | {fmt(m['football']['top2_hit'], 3)} | "
                     f"{fmt(m['ensemble_close']['top2_hit'], 3)} |")
    h = payload["htft_overall"]
    lines += ["", "## HT/FT（半场/全场 9 类，足球轨 path A）", "",
              "| model | n | log-loss | Brier | accuracy |", "|---|---:|---:|---:|---:|"]
    for key, label in (("football", "football (half-split DC)"), ("base_rate", "base rate (历史频率)")):
        lines.append(f"| {label} | {h[key]['n']} | {fmt(h[key]['log_loss'])} | {fmt(h[key]['brier'])} | "
                     f"{fmt(h[key]['accuracy'], 3)} |")
    lines += ["", f"均匀分布 9 类 log-loss = {np.log(9):.4f}。football-data 无 HT/FT 赔率，市场/集成 HT/FT 需 Sportmonks market 29 历史快照积累后评估。"]
    lines += [
        "",
        "## 说明",
        "",
        "- log-loss / Brier / RPS 越低越好；TOP2 = 真实比分落在模型概率最高的两个比分中的比例。",
        "- 市场 TOP2：由去水后的 1X2 反推 Dixon-Coles (λ, μ)；集成 TOP2：按集成 1X2 对足球模型比分矩阵分区重加权。",
        "- w_football = 0 表示历史 log-loss 认为纯足球模型对收盘赔率没有增量信息，此时集成 1X2 退化为市场轨"
        "（比分 TOP2 仍保留足球模型的比分形状）。`epl predict --football-weight` 可手动覆盖。",
        "- 仅供研究/参考，不构成投注建议，系统不做自动下注。",
        "",
        f"复现：`epl backtest`（耗时约 {payload['runtime_s']} 秒）。",
        "",
    ]
    md_path = out_dir / f"{name}.md"
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return md_path, json_path
