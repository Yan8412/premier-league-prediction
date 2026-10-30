"""Thin Sportmonks Football API v3 client.

Verified against https://docs.sportmonks.com/v3 (Oct 2026):

* Base URL ``https://api.sportmonks.com/v3/football``.
* Auth: token in the ``Authorization`` header (alternative: ``api_token``
  query param). We use the header so the token never appears in URLs/logs.
* Includes are ``;``-separated (``include=participants;scores``); filters are
  ``filters=name:v1,v2``.
* Pagination: response ``pagination`` has ``has_more``, ``next_page`` and
  ``next_cursor``. Live, both links are full URLs; we follow ``page`` from
  ``next_page`` (keeping ``per_page``), else the ``cursor`` token (without
  ``per_page``, which the API rejects alongside a cursor).
* Errors: 401 unauthenticated, 403 not in plan, 429 hourly per-entity rate
  limit (body ``rate_limit``: ``remaining``, ``resets_in_seconds``).

Endpoints used:
  GET /leagues/{id}?include=currentSeason;seasons
  GET /seasons/{id}
  GET /fixtures/{id}
  GET /fixtures/between/{start}/{end}?filters=fixtureLeagues:8   (max 100-day range)
  GET /odds/pre-match/fixtures/{id}/markets/{marketId}?include=bookmaker  (not paginated)
"""

from __future__ import annotations

import logging
import time
from datetime import date, timedelta
from typing import Any, Callable, Iterator
from urllib.parse import parse_qs, urlparse

import requests

from ..config import EPL_LEAGUE_ID, sportmonks_base_url, sportmonks_token

log = logging.getLogger(__name__)

FIXTURE_INCLUDES = "participants;scores;round"
MAX_RANGE_DAYS = 100


class SportmonksError(RuntimeError):
    def __init__(self, message: str, status: int | None = None, body: Any = None):
        super().__init__(message)
        self.status = status
        self.body = body


class MissingTokenError(SportmonksError):
    pass


class SportmonksClient:
    def __init__(
        self,
        token: str | None = None,
        base_url: str | None = None,
        session: requests.Session | None = None,
        timeout: float = 30.0,
        max_retries: int = 3,
        backoff_s: float = 2.0,
        sleep: Callable[[float], None] = time.sleep,
    ):
        self.token = token or sportmonks_token()
        if not self.token:
            raise MissingTokenError(
                "SPORTMONKS_API_TOKEN is not set. Export it or put it in .env (see .env.example)."
            )
        self.base_url = (base_url or sportmonks_base_url()).rstrip("/")
        self.session = session or requests.Session()
        self.timeout = timeout
        self.max_retries = max_retries
        self.backoff_s = backoff_s
        self.sleep = sleep
        self.last_rate_limit: dict | None = None
        self.requests_made = 0

    # ----------------------------------------------------------- transport
    def get(self, path: str, params: dict | None = None) -> dict:
        url = f"{self.base_url}/{path.lstrip('/')}"
        headers = {"Authorization": self.token, "Accept": "application/json"}
        for attempt in range(self.max_retries + 1):
            self.requests_made += 1
            try:
                resp = self.session.get(url, params=params or {}, headers=headers, timeout=self.timeout)
            except requests.RequestException as exc:
                if attempt >= self.max_retries:
                    raise SportmonksError(f"network error for {path}: {exc}") from exc
                self.sleep(self.backoff_s * 2**attempt)
                continue

            body = _json_or_text(resp)
            if isinstance(body, dict) and "rate_limit" in body:
                self.last_rate_limit = body["rate_limit"]

            if resp.status_code == 200:
                return body if isinstance(body, dict) else {"data": body}
            if resp.status_code == 429 or resp.status_code >= 500:
                if attempt >= self.max_retries:
                    break
                wait = self.backoff_s * 2**attempt
                if resp.status_code == 429 and isinstance(body, dict):
                    reset = (body.get("rate_limit") or {}).get("resets_in_seconds")
                    if isinstance(reset, (int, float)) and reset < 120:
                        wait = max(wait, float(reset) + 1)
                log.warning("Sportmonks %s on %s, retrying in %.0fs", resp.status_code, path, wait)
                self.sleep(wait)
                continue
            break

        message = body.get("message") if isinstance(body, dict) else str(body)[:300]
        hint = {401: "check SPORTMONKS_API_TOKEN", 403: "endpoint/add-on not in your plan (Odds add-on?)",
                429: "hourly rate limit reached"}.get(resp.status_code, "")
        raise SportmonksError(f"GET {path} -> HTTP {resp.status_code}: {message} {hint}".strip(),
                              status=resp.status_code, body=body)

    def paginate(self, path: str, params: dict | None = None, per_page: int = 50,
                 max_pages: int = 200) -> Iterator[dict]:
        params = dict(params or {})
        params["per_page"] = per_page
        for _ in range(max_pages):
            body = self.get(path, params)
            data = body.get("data") or []
            yield from (data if isinstance(data, list) else [data])
            pg = body.get("pagination") or {}
            if not pg.get("has_more"):
                return
            # Live API (Oct 2026) returns next_page and next_cursor as full URLs;
            # sending a URL as `cursor` gets HTTP 400.
            next_page = _query_value(pg.get("next_page"), "page")
            cursor = pg.get("next_cursor")
            if cursor and str(cursor).startswith("http"):
                cursor = _query_value(cursor, "cursor")
            if next_page:
                params.pop("cursor", None)
                params["page"] = int(next_page)
                params["per_page"] = per_page
            elif cursor:
                params.pop("per_page", None)  # per_page + cursor -> HTTP 400
                params.pop("page", None)
                params["cursor"] = cursor
            else:
                params["page"] = int(pg.get("current_page", params.get("page", 1))) + 1
                params["per_page"] = per_page
        log.warning("stopped paginating %s after %d pages", path, max_pages)

    # ---------------------------------------------------------- endpoints
    def league(self, league_id: int = EPL_LEAGUE_ID, include: str = "currentSeason;seasons") -> dict:
        return self.get(f"leagues/{league_id}", {"include": include})["data"]

    def season(self, season_id: int) -> dict:
        return self.get(f"seasons/{season_id}")["data"]

    def fixture(self, fixture_id: int, include: str = FIXTURE_INCLUDES) -> dict:
        return self.get(f"fixtures/{fixture_id}", {"include": include})["data"]

    def fixtures_between(self, start: date, end: date, league_id: int = EPL_LEAGUE_ID,
                         include: str = FIXTURE_INCLUDES) -> list[dict]:
        """All league fixtures in [start, end], split into <=100-day windows."""
        out: dict[int, dict] = {}
        cur = start
        while cur <= end:
            stop = min(end, cur + timedelta(days=MAX_RANGE_DAYS - 1))
            params = {"include": include, "filters": f"fixtureLeagues:{league_id}"}
            for fx in self.paginate(f"fixtures/between/{cur.isoformat()}/{stop.isoformat()}", params):
                if fx.get("league_id") == league_id:
                    out[fx["id"]] = fx
                else:
                    log.warning("dropping fixture %s with league_id=%s (expected %s)",
                                fx.get("id"), fx.get("league_id"), league_id)
            cur = stop + timedelta(days=1)
        return sorted(out.values(), key=lambda f: (f.get("starting_at_timestamp") or 0, f["id"]))

    def prematch_odds(self, fixture_id: int, market_id: int, include: str = "bookmaker",
                      bookmaker_ids: list[int] | None = None) -> list[dict]:
        params: dict[str, Any] = {"include": include}
        if bookmaker_ids:
            params["filters"] = "bookmakers:" + ",".join(str(b) for b in bookmaker_ids)
        data = self.get(f"odds/pre-match/fixtures/{fixture_id}/markets/{market_id}", params).get("data") or []
        return data if isinstance(data, list) else [data]


def _query_value(link: Any, key: str) -> str | None:
    """``key`` from a pagination link: a full URL, or the bare value itself."""
    if link is None or link == "":
        return None
    text = str(link)
    if not text.startswith("http"):
        return text if key != "page" or text.isdigit() else None
    values = parse_qs(urlparse(text).query).get(key)
    return values[0] if values else None


def _json_or_text(resp: requests.Response) -> Any:
    try:
        return resp.json()
    except ValueError:
        return resp.text
