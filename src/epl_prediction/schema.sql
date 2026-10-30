-- SQLite schema for the EPL prediction MVP (Sportmonks sync + stored predictions).
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS seasons (
    id            INTEGER PRIMARY KEY,
    league_id     INTEGER NOT NULL,
    name          TEXT,
    starting_at   TEXT,
    ending_at     TEXT,
    is_current    INTEGER,
    finished      INTEGER,
    synced_at     TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS teams (
    id          INTEGER PRIMARY KEY,
    name        TEXT,
    short_code  TEXT
);

CREATE TABLE IF NOT EXISTS fixtures (
    id                     INTEGER PRIMARY KEY,
    league_id              INTEGER,
    season_id              INTEGER,
    round_id               INTEGER,
    round_name             TEXT,
    name                   TEXT,
    starting_at            TEXT,      -- UTC "YYYY-MM-DD HH:MM:SS"
    starting_at_timestamp  INTEGER,
    state_id               INTEGER,   -- 1 = not started, 5/7/8 = finished
    home_team_id           INTEGER,
    home_team              TEXT,
    away_team_id           INTEGER,
    away_team              TEXT,
    ft_home                INTEGER,
    ft_away                INTEGER,
    ht_home                INTEGER,
    ht_away                INTEGER,
    result_info            TEXT,
    synced_at              TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_fixtures_season ON fixtures(season_id);
CREATE INDEX IF NOT EXISTS idx_fixtures_kickoff ON fixtures(starting_at_timestamp);

CREATE TABLE IF NOT EXISTS bookmakers (
    id    INTEGER PRIMARY KEY,
    name  TEXT
);

-- Every fetch is kept: the standard pre-match feed only returns current prices,
-- so "opening" and "T-6h"/"T-1h" odds exist only if we snapshot them ourselves.
CREATE TABLE IF NOT EXISTS odds_snapshots (
    snapshot_id              INTEGER PRIMARY KEY AUTOINCREMENT,
    fetched_at               TEXT NOT NULL,   -- UTC ISO-8601
    snapshot_tag             TEXT NOT NULL,   -- opening | T-6h | T-1h | interim | post_kickoff
    hours_to_kickoff         REAL,
    fixture_id               INTEGER NOT NULL,
    market_id                INTEGER NOT NULL,
    bookmaker_id             INTEGER,
    odd_id                   INTEGER,
    label                    TEXT,
    name                     TEXT,
    value                    REAL,
    stopped                  INTEGER,
    latest_bookmaker_update  TEXT
);
CREATE INDEX IF NOT EXISTS idx_odds_fixture ON odds_snapshots(fixture_id, market_id, fetched_at);

CREATE TABLE IF NOT EXISTS predictions (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    fixture_id     INTEGER,
    home_team      TEXT,
    away_team      TEXT,
    kickoff        TEXT,
    track          TEXT NOT NULL,   -- football | market | ensemble
    model_version  TEXT NOT NULL,
    generated_at   TEXT NOT NULL,
    p_home         REAL,
    p_draw         REAL,
    p_away         REAL,
    payload_json   TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_predictions_fixture ON predictions(fixture_id, track);

CREATE TABLE IF NOT EXISTS sync_log (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    run_at      TEXT NOT NULL,
    command     TEXT NOT NULL,
    detail_json TEXT
);
