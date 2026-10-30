"""Runtime configuration. Secrets come from the environment only."""

from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"
FOOTBALL_DATA_DIR = DATA_DIR / "football_data"
REPORTS_DIR = REPO_ROOT / "reports"
LIVE_WEIGHTS_PATH = REPORTS_DIR / "ensemble_weights.json"  # written by `epl review monthly`

SPORTMONKS_BASE_URL_DEFAULT = "https://api.sportmonks.com/v3/football"
EPL_LEAGUE_ID = 8

MARKET_FT_1X2 = 1
MARKET_HT_FT = 29


def load_dotenv(path: Path | None = None) -> None:
    """Minimal `.env` loader: KEY=VALUE lines, never overrides existing env vars."""
    path = path or REPO_ROOT / ".env"
    if not path.is_file():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and value and key not in os.environ:
            os.environ[key] = value


def sportmonks_token() -> str | None:
    token = os.environ.get("SPORTMONKS_API_TOKEN", "").strip()
    return token or None


def sportmonks_base_url() -> str:
    return os.environ.get("SPORTMONKS_BASE_URL", SPORTMONKS_BASE_URL_DEFAULT).rstrip("/")


def db_path() -> Path:
    return Path(os.environ.get("EPL_DB_PATH", str(DATA_DIR / "epl.db")))
