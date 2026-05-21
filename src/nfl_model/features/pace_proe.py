"""Pace and pass-rate-over-expected features.

Rolling 4 / 8-game pace metrics: plays per drive, seconds per play, neutral
pass rate, PROE.
"""

from __future__ import annotations

import pandas as pd

from nfl_model.data.warehouse import query
from nfl_model.features._rolling import leakage_safe_rolling
from nfl_model.logging import get_logger

log = get_logger("features.pace_proe")

PACE_VALUE_COLS = ("plays_per_drive", "neutral_pass_rate", "proe")


def build(games: pd.DataFrame) -> pd.DataFrame:
    if games.empty:
        return pd.DataFrame()
    tg = query(
        """
        SELECT t.game_id, t.team, t.plays_per_drive, t.neutral_pass_rate, t.proe,
               g.game_date
        FROM team_game_stats t JOIN games g USING(game_id)
        ORDER BY g.game_date
        """
    )
    if tg.empty:
        return pd.DataFrame()
    tg["game_date"] = pd.to_datetime(tg["game_date"], errors="coerce")
    tg = tg.sort_values(["team", "game_date"]).reset_index(drop=True)
    rolled = leakage_safe_rolling(
        tg, group_col="team", sort_col="game_date",
        value_cols=list(PACE_VALUE_COLS), window=8, min_periods=2, weight="recency",
    )
    rolled["game_id"] = tg["game_id"].values
    rolled["team"] = tg["team"].values

    home = games[["game_id", "home_team"]].rename(columns={"home_team": "team"}).merge(
        rolled, on=["game_id", "team"], how="left"
    ).drop(columns=["team"])
    away = games[["game_id", "away_team"]].rename(columns={"away_team": "team"}).merge(
        rolled, on=["game_id", "team"], how="left"
    ).drop(columns=["team"])

    home = home.add_prefix("home_pace_").rename(columns={"home_pace_game_id": "game_id"})
    away = away.add_prefix("away_pace_").rename(columns={"away_pace_game_id": "game_id"})
    return home.merge(away, on="game_id", how="outer")
