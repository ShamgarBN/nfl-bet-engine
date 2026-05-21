"""Schedule-context features: rest, short week, bye, travel, primetime, divisional.

Computes great-circle travel miles between consecutive home stadiums for
each team, time-zone shift, days of rest, and various binary flags.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from nfl_model.data.warehouse import query
from nfl_model.logging import get_logger

log = get_logger("features.schedule_context")

# Approximate geographic centroid for each team's home stadium for travel calc.
TEAM_HOME_LAT_LON: dict[str, tuple[float, float]] = {
    "BUF": (42.7738, -78.7869), "MIA": (25.9580, -80.2389),
    "NE":  (42.0909, -71.2643), "NYJ": (40.8135, -74.0744),
    "BAL": (39.2780, -76.6227), "CIN": (39.0954, -84.5161),
    "CLE": (41.5061, -81.6995), "PIT": (40.4468, -80.0158),
    "HOU": (29.6847, -95.4107), "IND": (39.7601, -86.1639),
    "JAX": (30.3239, -81.6373), "TEN": (36.1665, -86.7713),
    "DEN": (39.7439, -105.0201), "KC":  (39.0489, -94.4839),
    "LV":  (36.0908, -115.1830), "LAC": (33.9534, -118.3387),
    "DAL": (32.7473, -97.0945), "NYG": (40.8135, -74.0744),
    "PHI": (39.9008, -75.1675), "WAS": (38.9077, -76.8645),
    "CHI": (41.8623, -87.6167), "DET": (42.3400, -83.0456),
    "GB":  (44.5013, -88.0622), "MIN": (44.9737, -93.2581),
    "ATL": (33.7553, -84.4006), "CAR": (35.2258, -80.8528),
    "NO":  (29.9511, -90.0812), "TB":  (27.9759, -82.5033),
    "ARI": (33.5276, -112.2626), "LA":  (33.9534, -118.3387),
    "SF":  (37.4032, -121.9698), "SEA": (47.5952, -122.3316),
}


def _great_circle_miles(p1: tuple[float, float], p2: tuple[float, float]) -> float:
    """Great-circle distance in miles between two (lat, lon) points."""
    lat1, lon1 = math.radians(p1[0]), math.radians(p1[1])
    lat2, lon2 = math.radians(p2[0]), math.radians(p2[1])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    c = 2 * math.asin(math.sqrt(a))
    return 3956.0 * c


def build(games: pd.DataFrame) -> pd.DataFrame:
    if games.empty:
        return pd.DataFrame()

    # Build a per-team itinerary so we can compute days-of-rest and travel.
    rows: list[dict] = []
    for team in pd.unique(pd.concat([games["home_team"], games["away_team"]])):
        team_games = games[(games["home_team"] == team) | (games["away_team"] == team)].copy()
        team_games = team_games.sort_values("game_date")
        team_games["was_home"] = team_games["home_team"] == team
        team_games["opponent_home"] = np.where(
            team_games["was_home"], team_games["home_team"], team_games["away_team"]
        )
        prev_date = None
        prev_loc: tuple[float, float] | None = None
        for _, g in team_games.iterrows():
            curr_loc = TEAM_HOME_LAT_LON.get(g["home_team"], (0.0, 0.0))
            curr_date = pd.Timestamp(g["game_date"]).to_pydatetime().date() if pd.notna(g["game_date"]) else None
            rest_days = (
                (curr_date - prev_date).days if (curr_date and prev_date) else None
            )
            travel = _great_circle_miles(prev_loc, curr_loc) if prev_loc else 0.0
            rows.append(
                {
                    "game_id": g["game_id"],
                    "team": team,
                    "rest_days": rest_days,
                    "short_week": rest_days is not None and rest_days <= 4,
                    "long_rest": rest_days is not None and rest_days >= 10,
                    "travel_miles": travel,
                }
            )
            prev_date = curr_date
            prev_loc = curr_loc

    sched = pd.DataFrame(rows)

    home = games[["game_id", "home_team"]].rename(columns={"home_team": "team"}).merge(
        sched, on=["game_id", "team"], how="left"
    ).drop(columns=["team"])
    away = games[["game_id", "away_team"]].rename(columns={"away_team": "team"}).merge(
        sched, on=["game_id", "team"], how="left"
    ).drop(columns=["team"])
    home = home.rename(columns={c: f"home_sc_{c}" for c in home.columns if c != "game_id"})
    away = away.rename(columns={c: f"away_sc_{c}" for c in away.columns if c != "game_id"})
    out = home.merge(away, on="game_id", how="outer")

    # Game-level flags (not team-keyed)
    games_meta = games[["game_id", "weekday", "primetime", "divisional"]].copy()
    out = out.merge(games_meta, on="game_id", how="left")
    out["thursday_game"] = out["weekday"] == "Thursday"
    out["monday_game"] = out["weekday"] == "Monday"
    out["sunday_late"] = out["weekday"] == "Sunday"

    log.debug("schedule_context.built", n=len(out))
    return out
