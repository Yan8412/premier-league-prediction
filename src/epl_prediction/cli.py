"""Command line interface: ``epl <command>`` (or ``python -m epl_prediction``)."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

from .config import EPL_LEAGUE_ID, MARKET_FT_1X2, MARKET_HT_FT, load_dotenv, sportmonks_token

NO_TOKEN_MSG = ("SPORTMONKS_API_TOKEN is not set: skipping live Sportmonks call.\n"
                "Set it with `export SPORTMONKS_API_TOKEN=...` or copy .env.example to .env (never commit it).")


def _client():
    from .sportmonks.client import SportmonksClient

    return SportmonksClient()


def _conn(args):
    from .db import connect

    return connect(args.db)


def _parse_odds_arg(values: list[str] | None) -> dict:
    """``--odds H,D,A`` (repeatable, one per bookmaker)."""
    out = {}
    for k, v in enumerate(values or []):
        parts = [float(x) for x in v.split(",")]
        if len(parts) != 3:
            raise SystemExit(f"--odds expects H,D,A decimal odds, got {v!r}")
        out[f"manual{k + 1}"] = tuple(parts)
    return out


# ------------------------------------------------------------------- sync
def cmd_sync(args) -> int:
    if not sportmonks_token():
        print(NO_TOKEN_MSG, file=sys.stderr)
        return 2
    from .sportmonks import sync

    client, conn = _client(), _conn(args)
    what = args.what
    if what == "due":
        from .sportmonks.scheduler import run_due

        print(json.dumps(run_due(client, conn, horizon_days=args.horizon_days, refresh_hours=args.refresh_hours),
                         indent=2, default=str))
        return 0
    if what in ("seasons", "all"):
        rows = sync.sync_seasons(client, conn, args.league_id)
        print(f"seasons: {len(rows)} stored")
    if what in ("fixtures", "all"):
        if args.date_from and args.date_to:
            n = sync.sync_fixtures(client, conn, date.fromisoformat(args.date_from), date.fromisoformat(args.date_to),
                                   args.league_id, with_xg=args.with_xg)
        else:
            season_id = args.season_id
            if season_id is None:
                if sync.current_season(conn, args.league_id) is None:
                    sync.sync_seasons(client, conn, args.league_id)
                season_id = sync.current_season(conn, args.league_id)["id"]
            n = sync.sync_season_fixtures(client, conn, season_id, with_xg=args.with_xg)
        print(f"fixtures: {n} upserted")
    if what == "fixture":
        if not args.fixture_id:
            raise SystemExit("sync fixture needs --fixture-id")
        for fid in args.fixture_id:
            print(json.dumps(sync.sync_fixture(client, conn, fid), ensure_ascii=False))
    if what in ("odds", "all"):
        ids = args.fixture_id or sync.upcoming_fixture_ids(conn, args.upcoming_days)
        markets = [int(m) for m in args.markets.split(",")]
        stats = sync.sync_odds(client, conn, ids, markets)
        print(json.dumps(stats, indent=2, default=str))
        htft = stats["by_market"].get(str(MARKET_HT_FT))
        if htft and htft["fixtures_with_odds"] == 0 and stats["fixtures"]:
            print("note: no HT/FT (market 29) odds returned; market track will omit p_htft.", file=sys.stderr)
    rl = client.last_rate_limit
    if rl:
        print(f"rate limit: {rl}", file=sys.stderr)
    return 0


def cmd_smoke(args) -> int:
    """Live smoke test: league 8 + current season + one fixture's odds. Skips cleanly without a token."""
    if not sportmonks_token():
        print("SKIP live smoke: " + NO_TOKEN_MSG)
        return 0
    from .sportmonks.parse import parse_fixture

    client = _client()
    league = client.league(EPL_LEAGUE_ID, include="currentSeason")
    season = league.get("currentseason") or league.get("currentSeason") or {}
    print(f"league {league.get('id')} {league.get('name')} | current season {season.get('id')} {season.get('name')}")
    today = date.today()
    fixtures = client.fixtures_between(today, today + timedelta(days=14))
    print(f"fixtures next 14 days: {len(fixtures)}")
    if fixtures:
        fx = parse_fixture(fixtures[0])
        print(f"first: {fx['id']} {fx['home_team']} vs {fx['away_team']} @ {fx['starting_at']}")
        for mid in (MARKET_FT_1X2, MARKET_HT_FT):
            odds = client.prematch_odds(fx["id"], mid)
            books = {o.get("bookmaker_id") for o in odds}
            print(f"market {mid}: {len(odds)} selections from {len(books)} bookmakers")
    print(f"requests made: {client.requests_made}; rate limit: {client.last_rate_limit}")
    return 0


def cmd_audit(args) -> int:
    from .audit import audit_db

    report = audit_db(_conn(args))
    if not args.verbose_mapping:
        report.pop("team_name_mapping")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if report["ok"] else 1


def cmd_coverage(args) -> int:
    from .sportmonks.scheduler import coverage

    print(json.dumps(coverage(_conn(args)), indent=2, ensure_ascii=False))
    return 0


def cmd_xg(args) -> int:
    from .data import xg

    if args.action == "import":
        if args.understat:
            df = xg.UnderstatXG(Path(args.understat)).match_xg()
            source = "understat"
        elif args.csv:
            df = xg.CsvXG(Path(args.csv), args.source).match_xg()
            source = args.source
        else:
            raise SystemExit("xg import needs --understat FILE or --csv FILE")
        if df.empty:
            raise SystemExit("no finished matches with xG found in the file")
        print(f"imported {len(df)} matches -> {xg.save_to_store(df, source)}")
        return 0
    store = xg.load_store()
    hist = xg.attach_xg(load_all_played(), [store])
    by = hist.loc[hist["hxg"].notna()].groupby(hist["date"].dt.year)["xg_source"].value_counts()
    print(json.dumps({"store_rows": int(len(store)), "history_matches_with_xg": int(hist["hxg"].notna().sum()),
                      "by_year_and_source": {f"{y}/{s}": int(n) for (y, s), n in by.items()}}, indent=2))
    return 0


def load_all_played():
    from .data.football_data import load_all

    df = load_all()
    return df[df["result"] >= 0][["date", "home", "away", "hg", "ag", "hxg", "axg"]]


# ---------------------------------------------------------------- backtest
def cmd_backtest(args) -> int:
    from .backtest.walkforward import BacktestConfig, run_backtest, write_report

    cfg = BacktestConfig(xi=args.xi, l2=args.l2, xg_weight=args.xg_weight, devig_method=args.devig,
                         test_seasons=args.test_seasons.split(",") if args.test_seasons else None,
                         data_dir=Path(args.data_dir) if args.data_dir else None)
    result = run_backtest(cfg)
    md, js = write_report(result, name=args.name)
    print(md.read_text(encoding="utf-8"))
    print(f"wrote {md} and {js}")
    return 0


def cmd_data(args) -> int:
    from .data.football_data import download_season

    for y in [int(s) for s in args.seasons.split(",")]:
        print(f"downloaded {download_season(y)}")
    return 0


# ----------------------------------------------------------------- predict
def cmd_predict(args) -> int:
    import pandas as pd

    from . import predict as P
    from .config import db_path

    db_file = Path(args.db) if args.db else db_path()
    conn = None
    if not args.no_db and (args.fixture_id or args.matchday or db_file.exists()):
        conn = _conn(args)

    if args.home and args.away:
        fixtures = [P.FixtureInput(home=args.home, away=args.away, kickoff=args.date,
                                   odds_1x2=_parse_odds_arg(args.odds))]
    elif args.fixture_id or args.matchday:
        if conn is None:
            raise SystemExit("--fixture-id / --matchday need the SQLite DB (run `epl sync all` first)")
        if args.fixture_id:
            missing = [f for f in args.fixture_id if conn.execute("SELECT 1 FROM fixtures WHERE id=?", (f,)).fetchone() is None]
            if missing and sportmonks_token():
                from .sportmonks.sync import sync_fixture

                for fid in missing:
                    sync_fixture(_client(), conn, fid)
            ids = list(args.fixture_id)
        else:
            ids = P.next_matchday_fixture_ids(conn)
            if not ids:
                raise SystemExit("no upcoming fixtures in DB; run `epl sync fixtures` first")
        fixtures = P.fixture_inputs_from_db(conn, ids)
    else:
        raise SystemExit("give --fixture-id, --matchday next, or --home/--away")

    as_of = None
    if args.date:
        as_of = datetime.fromisoformat(args.date[:10])
    elif fixtures and fixtures[0].kickoff:
        as_of = datetime.fromisoformat(fixtures[0].kickoff[:10])
    history = P.load_history(conn)
    model = P.fit_football(history, pd.Timestamp(as_of) if as_of else None, xg_weight=args.xg_weight)
    weight = args.football_weight if args.football_weight is not None else P.load_pool_weight()
    generated_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    htft_weight = (args.football_weight_htft if args.football_weight_htft is not None
                   else P.load_pool_weight(kind="htft"))
    preds = [P.predict_fixture(model, fx, weight, args.devig, generated_at, htft_weight) for fx in fixtures]
    from .contract import validate_prediction

    for pred in preds:
        for err in validate_prediction(pred):
            print(f"contract warning ({pred['home']} v {pred['away']}): {err}", file=sys.stderr)
    if conn is not None and not args.no_store and any(p["fixture_id"] for p in preds):
        P.store_predictions(conn, [p for p in preds if p["fixture_id"]])

    text = json.dumps(preds if len(preds) > 1 else preds[0], indent=2, ensure_ascii=False)
    if args.output:
        Path(args.output).write_text(text + "\n", encoding="utf-8")
        print(f"wrote {args.output}")
    else:
        print(text)
    return 0


# ------------------------------------------------------------------ review
def cmd_review(args) -> int:
    from .review import monthly_review, weekly_review

    if args.period == "weekly":
        print(json.dumps(weekly_review(_conn(args), days=args.days), indent=2))
    else:
        print(json.dumps(monthly_review(_conn(args), run_backtest_=not args.skip_backtest), indent=2, default=str))
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="epl", description="EPL prediction MVP (research only, no auto-betting)")
    p.add_argument("--db", help="SQLite path (default: $EPL_DB_PATH or data/epl.db)")
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("sync", help="sync Sportmonks data into SQLite (needs SPORTMONKS_API_TOKEN)")
    s.add_argument("what", choices=["seasons", "fixtures", "fixture", "odds", "all", "due"],
                   help="due = scheduler tick for cron: refresh fixtures + fetch only due odds snapshots")
    s.add_argument("--league-id", type=int, default=EPL_LEAGUE_ID)
    s.add_argument("--season-id", type=int)
    s.add_argument("--from", dest="date_from", help="YYYY-MM-DD (fixtures)")
    s.add_argument("--to", dest="date_to", help="YYYY-MM-DD (fixtures)")
    s.add_argument("--fixture-id", type=int, action="append", help="repeatable")
    s.add_argument("--upcoming-days", type=int, default=7, help="odds: fixtures kicking off within N days")
    s.add_argument("--markets", default=f"{MARKET_FT_1X2},{MARKET_HT_FT}", help="odds market ids (default 1,29)")
    s.add_argument("--with-xg", action="store_true", help="fixtures: also fetch xGFixture (Sportmonks xG add-on)")
    s.add_argument("--horizon-days", type=int, default=14, help="due: look-ahead for opening snapshots")
    s.add_argument("--refresh-hours", type=float, default=24.0, help="due: interim refresh age")
    s.set_defaults(func=cmd_sync)

    au = sub.add_parser("audit", help="data-quality audit of the SQLite DB (league filter, team names)")
    au.add_argument("--verbose-mapping", action="store_true", help="include the full team-name mapping")
    au.set_defaults(func=cmd_audit)

    cv = sub.add_parser("coverage", help="odds snapshot tags per upcoming/recent fixture (markets 1, 29)")
    cv.set_defaults(func=cmd_coverage)

    sm = sub.add_parser("smoke", help="live Sportmonks smoke test (skips without token)")
    sm.set_defaults(func=cmd_smoke)

    x = sub.add_parser("xg", help="xG store: import Understat/CSV files, show coverage")
    x.add_argument("action", choices=["import", "status"])
    x.add_argument("--understat", help="Understat match-list JSON saved by you (no crawling; see README)")
    x.add_argument("--csv", help="CSV with date,home,away,hxg,axg")
    x.add_argument("--source", default="csv", help="label for --csv imports")
    x.set_defaults(func=cmd_xg)

    d = sub.add_parser("data", help="download football-data.co.uk EPL CSVs")
    d.add_argument("--seasons", required=True, help="start years, e.g. 2024,2025")
    d.set_defaults(func=cmd_data)

    b = sub.add_parser("backtest", help="walk-forward backtest on football-data CSVs")
    b.add_argument("--xi", type=float, default=0.0019)
    b.add_argument("--l2", type=float, default=1e-2)
    b.add_argument("--xg-weight", type=float, default=0.5, help="xG share of Poisson targets where xG exists (0 = goals only)")
    b.add_argument("--devig", default="shin", choices=["shin", "multiplicative"])
    b.add_argument("--test-seasons", help="comma list of season codes, e.g. 2324,2425")
    b.add_argument("--data-dir")
    b.add_argument("--name", default="backtest_epl", help="report file stem under reports/")
    b.set_defaults(func=cmd_backtest)

    pr = sub.add_parser("predict", help="emit JSON predictions (football / market / ensemble)")
    pr.add_argument("--fixture-id", type=int, action="append", help="Sportmonks fixture id (repeatable)")
    pr.add_argument("--matchday", choices=["next"], help="all fixtures of the next upcoming round in the DB")
    pr.add_argument("--home", help="offline mode: home team (football-data or Sportmonks name)")
    pr.add_argument("--away", help="offline mode: away team")
    pr.add_argument("--date", help="offline mode: match date YYYY-MM-DD (model uses only earlier results)")
    pr.add_argument("--odds", action="append", help="offline mode: H,D,A decimal odds, repeat per bookmaker")
    pr.add_argument("--football-weight", type=float, help="override log-pool weight on the football track")
    pr.add_argument("--football-weight-htft", type=float, help="override log-pool weight for HT/FT (default prior 0.5)")
    pr.add_argument("--xg-weight", type=float, help="xG share of Poisson targets where xG exists (default 0.5)")
    pr.add_argument("--devig", default="shin", choices=["shin", "multiplicative"])
    pr.add_argument("--no-db", action="store_true", help="do not read the SQLite DB")
    pr.add_argument("--no-store", action="store_true", help="do not store predictions in the DB")
    pr.add_argument("--output", "-o", help="write JSON to file")
    pr.set_defaults(func=cmd_predict)

    r = sub.add_parser("review", help="self-upgrade hooks: weekly post-match / monthly refresh")
    r.add_argument("period", choices=["weekly", "monthly"])
    r.add_argument("--days", type=int, default=7)
    r.add_argument("--skip-backtest", action="store_true", help="monthly: only refit live weights")
    r.set_defaults(func=cmd_review)
    return p


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    from .sportmonks.client import SportmonksError

    try:
        return args.func(args)
    except SportmonksError as exc:
        print(f"Sportmonks error: {exc}", file=sys.stderr)
        return 1
    except LookupError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
