"""Dixon-Coles (1997) bivariate Poisson model with exponential time decay.

    log lambda_home = home + attack[h] + defence[a]
    log mu_away     =        attack[a] + defence[h]

``defence`` is "goals conceded" strength (higher = leakier). The low-score
dependence factor tau (rho) adjusts the 0-0, 1-0, 0-1 and 1-1 cells. Matches
are weighted by ``exp(-xi * days_before_cutoff)``.

Identifiability: a soft sum-to-zero penalty on attack plus a small L2 ridge
(also shrinks teams with little data). Fitted with L-BFGS-B on an analytic
gradient. Self-contained; ideas cross-checked against penaltyblog / mezzala
(both MIT) but no code is vendored.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from scipy.optimize import least_squares, minimize
from scipy.special import gammaln

from .halftime import DEFAULT_FIRST_HALF_SHARE, fit_half_params, htft_probs
from .score_matrix import RHO_BOUNDS, dc_score_matrix, matrix_to_1x2, normalise, tau_matrix  # noqa: F401

DEFAULT_XI = 0.0019  # per day, ~1 year half-life (DC paper: 0.0065 per half-week)


def invert_1x2(target_1x2: Sequence[float], rho: float = -0.05, max_goals: int = 10) -> tuple[float, float]:
    """Find (lambda, mu) whose Dixon-Coles matrix reproduces a 1X2 vector.

    1X2 has two degrees of freedom, so (lambda, mu) is identified. Used to give
    the market track a score distribution / TOP2 scores."""
    t = np.asarray(target_1x2, dtype=float)
    t = t / t.sum()

    def resid(x: np.ndarray) -> np.ndarray:
        p = matrix_to_1x2(dc_score_matrix(np.exp(x[0]), np.exp(x[1]), rho, max_goals))
        return np.array([p[0] - t[0], p[2] - t[2]])

    sol = least_squares(resid, x0=np.log([1.4, 1.1]), bounds=([-3, -3], [2, 2]), xtol=1e-12, ftol=1e-12)
    return float(np.exp(sol.x[0])), float(np.exp(sol.x[1]))


@dataclass
class DixonColesModel:
    xi: float = DEFAULT_XI
    l2: float = 1e-2  # chosen on 2020/21-2022/23 walk-forward only
    max_goals: int = 10
    # Weight of xG in the Poisson targets where xG exists: target = a*xG + (1-a)*goals.
    # The rho / low-score correction always uses actual goals.
    xg_weight: float = 0.5
    teams: list[str] = field(default_factory=list)
    attack: np.ndarray | None = None
    defence: np.ndarray | None = None
    home_adv: float = 0.25
    rho: float = -0.05
    ht_share_home: float = DEFAULT_FIRST_HALF_SHARE
    ht_share_away: float = DEFAULT_FIRST_HALF_SHARE
    rho_ht: float = 0.0
    n_matches: int = 0
    n_xg_matches: int = 0
    converged: bool = False

    @property
    def is_fitted(self) -> bool:
        return self.attack is not None

    # ------------------------------------------------------------------ fit
    def fit(
        self,
        matches: pd.DataFrame,
        as_of: pd.Timestamp | None = None,
        max_days: int | None = 1100,
        warm_start: "DixonColesModel | None" = None,
    ) -> "DixonColesModel":
        """``matches`` needs columns date, home, away, hg, ag (optional hthg, htag
        for the HT/FT half split). Only rows strictly
        before ``as_of`` (default: day after last match) are used."""
        df = matches.dropna(subset=["hg", "ag"])
        as_of = pd.Timestamp(as_of) if as_of is not None else df["date"].max() + pd.Timedelta(days=1)
        df = df[df["date"] < as_of]
        days = (as_of - df["date"]).dt.days.to_numpy(dtype=float)
        if max_days is not None:
            keep = days <= max_days
            df, days = df[keep], days[keep]
        if len(df) < 20:
            raise ValueError(f"not enough matches to fit Dixon-Coles ({len(df)})")

        teams = sorted(set(df["home"]) | set(df["away"]))
        idx = {t: k for k, t in enumerate(teams)}
        h = df["home"].map(idx).to_numpy()
        a = df["away"].map(idx).to_numpy()
        x_obs = df["hg"].to_numpy(dtype=float)
        y_obs = df["ag"].to_numpy(dtype=float)
        x, y = x_obs, y_obs
        self.n_xg_matches = 0
        if self.xg_weight > 0 and {"hxg", "axg"} <= set(df.columns):
            hx, ax = df["hxg"].to_numpy(dtype=float), df["axg"].to_numpy(dtype=float)
            has = ~(np.isnan(hx) | np.isnan(ax))
            a_ = self.xg_weight
            x = np.where(has, a_ * np.nan_to_num(hx) + (1 - a_) * x_obs, x_obs)
            y = np.where(has, a_ * np.nan_to_num(ax) + (1 - a_) * y_obs, y_obs)
            self.n_xg_matches = int(has.sum())
        w = np.exp(-self.xi * days)
        w = w / w.sum()
        n = len(teams)

        x0 = np.zeros(2 * n + 2)
        x0[-2], x0[-1] = 0.25, -0.05
        if warm_start is not None and warm_start.is_fitted:
            for t, k in idx.items():
                if t in warm_start.teams:
                    j = warm_start.teams.index(t)
                    x0[k], x0[n + k] = warm_start.attack[j], warm_start.defence[j]
            x0[-2], x0[-1] = warm_start.home_adv, warm_start.rho
        else:
            # Rough start from mean goals keeps L-BFGS-B iterations low.
            x0[n:2 * n] = np.log(max((x.mean() + y.mean()) / 2, 0.1))

        is00, is01, is10, is11 = ((x_obs == 0) & (y_obs == 0), (x_obs == 0) & (y_obs == 1),
                                  (x_obs == 1) & (y_obs == 0), (x_obs == 1) & (y_obs == 1))
        const = (gammaln(x + 1) + gammaln(y + 1)) @ w
        l2 = self.l2

        def objective(theta: np.ndarray) -> tuple[float, np.ndarray]:
            att, dfn, home, rho = theta[:n], theta[n:2 * n], theta[-2], theta[-1]
            eta1 = home + att[h] + dfn[a]
            eta2 = att[a] + dfn[h]
            lam, mu = np.exp(eta1), np.exp(eta2)

            tau = np.ones_like(lam)
            tau[is00] = 1.0 - lam[is00] * mu[is00] * rho
            tau[is01] = 1.0 + lam[is01] * rho
            tau[is10] = 1.0 + mu[is10] * rho
            tau[is11] = 1.0 - rho
            tau = np.clip(tau, 1e-10, None)

            ll = np.log(tau) + x * eta1 - lam + y * eta2 - mu
            sum_att = att.sum()
            loss = -(ll @ w) + const + l2 * (att @ att + dfn @ dfn) + sum_att**2

            # d ll / d eta1, d eta2, d rho
            g1 = x - lam
            g2 = y - mu
            grho = np.zeros_like(lam)
            g1[is00] += -lam[is00] * mu[is00] * rho / tau[is00]
            g2[is00] += -lam[is00] * mu[is00] * rho / tau[is00]
            grho[is00] = -lam[is00] * mu[is00] / tau[is00]
            g1[is01] += lam[is01] * rho / tau[is01]
            grho[is01] = lam[is01] / tau[is01]
            g2[is10] += mu[is10] * rho / tau[is10]
            grho[is10] = mu[is10] / tau[is10]
            grho[is11] = -1.0 / tau[is11]

            wg1, wg2 = w * g1, w * g2
            grad = np.empty_like(theta)
            grad[:n] = -(np.bincount(h, wg1, n) + np.bincount(a, wg2, n)) + 2 * l2 * att + 2 * sum_att
            grad[n:2 * n] = -(np.bincount(a, wg1, n) + np.bincount(h, wg2, n)) + 2 * l2 * dfn
            grad[-2] = -wg1.sum()
            grad[-1] = -(w @ grho)
            return float(loss), grad

        bounds = [(-3, 3)] * (2 * n) + [(-1, 1), RHO_BOUNDS]
        res = minimize(objective, x0, jac=True, method="L-BFGS-B", bounds=bounds,
                       options={"maxiter": 2000, "ftol": 1e-12, "gtol": 1e-8})
        theta = res.x
        self.teams = teams
        self.attack, self.defence = theta[:n].copy(), theta[n:2 * n].copy()
        self.home_adv, self.rho = float(theta[-2]), float(theta[-1])
        self.n_matches = int(len(df))
        self.converged = bool(res.success)
        if {"hthg", "htag"} <= set(df.columns):
            lam = np.exp(self.home_adv + self.attack[h] + self.defence[a])
            mu = np.exp(self.attack[a] + self.defence[h])
            self.ht_share_home, self.ht_share_away, self.rho_ht = fit_half_params(
                lam, mu, x_obs, y_obs, df["hthg"].to_numpy(dtype=float), df["htag"].to_numpy(dtype=float), w)
        return self

    # -------------------------------------------------------------- predict
    def _team_params(self, team: str) -> tuple[float, float]:
        if team in self.teams:
            k = self.teams.index(team)
            return float(self.attack[k]), float(self.defence[k])
        # Unseen (typically promoted) team: average of the three weakest teams.
        strength = self.attack - self.defence
        weakest = np.argsort(strength)[:3]
        return float(self.attack[weakest].mean()), float(self.defence[weakest].mean())

    def expected_goals(self, home: str, away: str) -> tuple[float, float]:
        if not self.is_fitted:
            raise RuntimeError("model is not fitted")
        ah, dh = self._team_params(home)
        aa, da = self._team_params(away)
        return float(np.exp(self.home_adv + ah + da)), float(np.exp(aa + dh))

    def score_matrix(self, home: str, away: str) -> np.ndarray:
        lam, mu = self.expected_goals(home, away)
        return dc_score_matrix(lam, mu, self.rho, self.max_goals)

    def predict_htft(self, home: str, away: str) -> np.ndarray:
        lam, mu = self.expected_goals(home, away)
        ft = dc_score_matrix(lam, mu, self.rho, self.max_goals)
        return htft_probs(lam, mu, self.ht_share_home, self.ht_share_away, self.rho_ht, ft)

    def predict_1x2(self, home: str, away: str) -> np.ndarray:
        return matrix_to_1x2(self.score_matrix(home, away))

    def known(self, teams: Iterable[str]) -> list[bool]:
        return [t in self.teams for t in teams]

    def to_dict(self) -> dict:
        return {
            "xi": self.xi,
            "l2": self.l2,
            "home_adv": self.home_adv,
            "rho": self.rho,
            "ht_share_home": self.ht_share_home,
            "ht_share_away": self.ht_share_away,
            "rho_ht": self.rho_ht,
            "n_matches": self.n_matches,
            "converged": self.converged,
            "xg_weight": self.xg_weight,
            "n_xg_matches": self.n_xg_matches,
            "teams": {t: {"attack": float(a), "defence": float(d)}
                      for t, a, d in zip(self.teams, self.attack, self.defence)},
        }
