"""Open-Meteo weather ingestion.

Free, key-less API with two endpoints we care about:

- ``archive-api.open-meteo.com`` for any date in the past
- ``api.open-meteo.com/v1/forecast`` for today + ~16 days out

Pulls hourly temperature, humidity, pressure, wind speed + direction, and
precipitation; selects the hour nearest kickoff; projects wind onto the
stadium's pass-axis bearing; writes one row per outdoor game to
``weather``. Indoor (dome / closed-roof) games get a row with
``is_dome=True`` and zeros for the weather fields so downstream features
can treat the column as never-null.
"""

from __future__ import annotations

import math
from datetime import date, datetime, timedelta

import httpx
import pandas as pd
from tenacity import retry, stop_after_attempt, wait_exponential

from nfl_model.config import settings
from nfl_model.data.warehouse import query, upsert_dataframe
from nfl_model.logging import get_logger

log = get_logger("data.sources.weather")

_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
_HOURLY_VARS = (
    "temperature_2m,relative_humidity_2m,pressure_msl,"
    "wind_speed_10m,wind_direction_10m,precipitation"
)


# Roofs treated as "indoors" — any of these short-circuits the API call.
_INDOOR_ROOFS = {"dome", "closed", "indoors", "indoor"}


def _is_indoor(roof: str | None) -> bool:
    if roof is None:
        return False
    return roof.lower() in _INDOOR_ROOFS


def _project_wind_on_pass_axis(speed_mph: float, dir_deg: float, axis_deg: float) -> float:
    """Project wind onto the stadium's long axis (the passing axis).

    Returns the signed wind component along the axis in the same units as
    ``speed_mph``. Positive = wind blowing in the +axis direction.
    Wind direction is reported as "where the wind is coming FROM" in
    meteorology, so we flip 180° to get the "blowing TO" vector before
    projecting.
    """
    if not (math.isfinite(speed_mph) and math.isfinite(dir_deg) and math.isfinite(axis_deg)):
        return 0.0
    blowing_to = (dir_deg + 180.0) % 360.0
    delta = math.radians(blowing_to - axis_deg)
    return float(speed_mph * math.cos(delta))


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1.0, min=2.0, max=10.0))
def _fetch_hourly(url: str, params: dict) -> dict:
    headers = {"User-Agent": settings.http_user_agent}
    with httpx.Client(timeout=settings.http_timeout_seconds, headers=headers) as client:
        resp = client.get(url, params=params)
        resp.raise_for_status()
        return resp.json()


def _hourly_to_dict(payload: dict, target: datetime) -> dict | None:
    hourly = payload.get("hourly") or {}
    times = hourly.get("time") or []
    if not times:
        return None
    # Find nearest hour
    target_naive = target.replace(tzinfo=None)
    parsed = [datetime.fromisoformat(t) for t in times]
    diffs = [abs((p - target_naive).total_seconds()) for p in parsed]
    idx = diffs.index(min(diffs))

    def at(name: str) -> float | None:
        seq = hourly.get(name)
        if seq is None or idx >= len(seq):
            return None
        return seq[idx]

    temp_c = at("temperature_2m")
    return {
        "temp_f": (temp_c * 9 / 5 + 32) if temp_c is not None else None,
        "humidity_pct": at("relative_humidity_2m"),
        "pressure_hpa": at("pressure_msl"),
        "wind_speed_mph": at("wind_speed_10m"),
        "wind_dir_deg": at("wind_direction_10m"),
        "precipitation_mm": at("precipitation"),
        "condition": None,
    }


def pull_weather_for_games(games_df: pd.DataFrame) -> int:
    """For each game in ``games_df``, write one row to ``weather``.

    Indoor games get a flat ``is_dome=True`` row with zero weather. Outdoor
    games hit Open-Meteo (archive for past dates, forecast for future)
    once per game and persist the hour-of-kickoff snapshot. Idempotent
    upsert keyed on ``game_id``.

    Resolves the venue via stadium_id; falls back to the home team's primary
    stadium when the stadium_id is unknown (covers neutral-site games at
    international venues by registering those in :mod:`.venues`).
    """
    if games_df.empty:
        return 0

    venues = query(
        "SELECT stadium_id, team, lat, lon, pass_axis_bearing_deg, roof FROM venues"
    )
    venue_by_id = {
        row["stadium_id"]: row for _, row in venues.iterrows()
    } if not venues.empty else {}
    # Per-team lookup for stadium_id misses. When multiple venues map to the
    # same team (rare; mostly intl), prefer the one whose stadium_id ends in
    # "00" (the home stadium).
    venue_by_team: dict[str, dict] = {}
    if not venues.empty:
        for _, row in venues.iterrows():
            team = row.get("team")
            if not team or pd.isna(team):
                continue
            existing = venue_by_team.get(team)
            if existing is None or str(row["stadium_id"]).endswith("00"):
                venue_by_team[team] = row

    # Need home_team for the fallback lookup. Join from games if not present.
    if "home_team" not in games_df.columns:
        from nfl_model.data.warehouse import query as _wh_query
        ids = games_df["game_id"].astype(str).unique().tolist()
        if ids:
            placeholders = ",".join("?" for _ in ids)
            home_lookup = _wh_query(
                f"SELECT game_id, home_team FROM games WHERE game_id IN ({placeholders})",
                tuple(ids),
            )
            games_df = games_df.merge(home_lookup, on="game_id", how="left")

    today = date.today()
    rows = []
    for _, g in games_df.iterrows():
        gid = str(g["game_id"])
        roof = g.get("roof") if pd.notna(g.get("roof")) else None
        if _is_indoor(roof):
            rows.append(
                {
                    "game_id": gid,
                    "temp_f": 70.0, "humidity_pct": 50.0, "pressure_hpa": 1013.0,
                    "wind_speed_mph": 0.0, "wind_dir_deg": 0.0,
                    "wind_on_pass_axis": 0.0,
                    "precipitation_mm": 0.0, "condition": "indoor",
                    "is_dome": True,
                }
            )
            continue

        venue = venue_by_id.get(g.get("stadium_id"))
        if venue is None:
            # Fall back to the home team's primary stadium when stadium_id is
            # unknown (covers nflverse PFR codes we haven't seeded yet).
            ht = g.get("home_team")
            if ht is not None and not pd.isna(ht):
                venue = venue_by_team.get(ht)
        if venue is None:
            continue
        lat, lon = float(venue["lat"]), float(venue["lon"])
        axis = float(venue["pass_axis_bearing_deg"] or 0.0)

        kickoff = g.get("kickoff_ts")
        if pd.notna(kickoff):
            kt = pd.to_datetime(kickoff).to_pydatetime()
        elif pd.notna(g.get("game_date")):
            d = pd.to_datetime(g["game_date"]).date()
            kt = datetime.combine(d, datetime.min.time()).replace(hour=13)
        else:
            continue

        url = _ARCHIVE_URL if kt.date() < today else _FORECAST_URL
        params = {
            "latitude": lat,
            "longitude": lon,
            "hourly": _HOURLY_VARS,
            "wind_speed_unit": "mph",
            "temperature_unit": "celsius",
            "timezone": "UTC",
            "start_date": kt.date().isoformat(),
            "end_date": kt.date().isoformat(),
        }
        try:
            payload = _fetch_hourly(url, params)
        except httpx.HTTPError as exc:
            log.warning("weather.fetch_failed", game_id=gid, error=str(exc))
            continue

        snap = _hourly_to_dict(payload, kt)
        if snap is None:
            continue

        wpa = _project_wind_on_pass_axis(
            snap["wind_speed_mph"] or 0.0,
            snap["wind_dir_deg"] or 0.0,
            axis,
        )
        rows.append(
            {
                "game_id": gid,
                **snap,
                "wind_on_pass_axis": wpa,
                "is_dome": False,
            }
        )

    if not rows:
        return 0
    df = pd.DataFrame(rows)
    upsert_dataframe(df, "weather", key_columns=["game_id"])
    return int(len(df))
