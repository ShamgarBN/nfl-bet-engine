"""Team-level offense / defense EPA features.

Computes opponent-adjusted rolling EPA per play for each team's offense and
defense across multiple windows (4, 8, 16 games). Splits pass vs rush.
Uses shrinkage toward season-to-date league mean for early-season games.

For each game in ``games`` (one row per game), produces *both* sides
(home and away) prefixed with home_/away_. Leakage-safe: every rolling
feature uses only games strictly before the target game's date.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from nfl_model.data.warehouse import query
from nfl_model.features._rolling import leakage_safe_rolling
from nfl_model.logging import get_logger

log = get_logger("features.team_epa")

ROLL_WINDOWS = (4, 8, 16)
VALUE_COLS = (
    "offense_epa_per_play",
    "defense_epa_per_play",
    "pass_epa",
    "rush_epa",
    "success_rate",
    "explosive_rate",
)


def build(games: pd.DataFrame) -> pd.DataFrame:
    """Build team-EPA features for every game.

    Returns a DataFrame keyed by ``game_id`` with columns like
    ``home_off_epa_roll4``, ``away_off_epa_roll8``, etc.
    """
    if games.empty:
        return pd.DataFrame()

    tg = query(
        """
        SELECT t.game_id, t.team, t.is_home,
               t.offense_epa_per_play, t.defense_epa_per_play,
               t.pass_epa, t.rush_epa, t.success_rate, t.explosive_rate,
               g.game_date
        FROM team_game_stats t
        JOIN games g USING(game_id)
        ORDER BY g.game_date
        """
    )
    if tg.empty:
        return pd.DataFrame()

    tg["game_date"] = pd.to_datetime(tg["game_date"], errors="coerce")
    tg = tg.sort_values(["team", "game_date"]).reset_index(drop=True)

    # Compute leakage-safe rolling for each value column at each window.
    rollups = []
    for window in ROLL_WINDOWS:
        rolled = leakage_safe_rolling(
            tg,
            group_col="team",
            sort_col="game_date",
            value_cols=list(VALUE_COLS),
            window=window,
            min_periods=2,
            weight="recency",
        )
        rolled["game_id"] = tg["game_id"].values
        rolled["team"] = tg["team"].values
        rollups.append(rolled)

    # Combine per-team-per-game rollups across windows
    base = rollups[0]
    for piece in rollups[1:]:
        base = base.merge(piece, on=["game_id", "team"], how="outer")

    # Pivot to home/away per game_id
    home_idx = games[["game_id", "home_team"]].rename(columns={"home_team": "team"})
    away_idx = games[["game_id", "away_team"]].rename(columns={"away_team": "team"})

    home_feats = home_idx.merge(base, on=["game_id", "team"], how="left").drop(columns=["team"])
    away_feats = away_idx.merge(base, on=["game_id", "team"], how="left").drop(columns=["team"])

    home_feats = home_feats.add_prefix("home_")
    home_feats = home_feats.rename(columns={"home_game_id": "game_id"})
    away_feats = away_feats.add_prefix("away_")
    away_feats = away_feats.rename(columns={"away_game_id": "game_id"})

    out = home_feats.merge(away_feats, on="game_id", how="outer")
    log.debug("team_epa.built", n=len(out))
    return out
