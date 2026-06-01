"""Drive-level efficiency features.

Per-team rolling: drives per game, yards per drive (offense / defense),
red-zone trip rate, turnover differential. These are extracted directly
from PBP rather than persisting in ``team_game_stats`` so the feature
can evolve without a warehouse migration.

Leakage-safe: each game's value is the team's mean across prior games only.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from nfl_model.data.warehouse import query
from nfl_model.features._rolling import leakage_safe_rolling
from nfl_model.logging import get_logger

log = get_logger("features.drive_eff")


def _per_game_team_stats() -> pd.DataFrame:
    """Aggregate per-(game_id, team) drive efficiency stats from PBP."""
    plays = query(
        """
        SELECT game_id, posteam, defteam, yardline_100, yards_gained,
               touchdown, play_type, epa
        FROM pbp
        WHERE posteam IS NOT NULL AND defteam IS NOT NULL
        """
    )
    if plays.empty:
        return pd.DataFrame()

    plays["yards_gained"] = pd.to_numeric(plays["yards_gained"], errors="coerce").fillna(0)
    plays["is_red_zone"] = plays["yardline_100"].between(0, 20, inclusive="both")
    # Turnover proxy: large negative EPA on a non-scoring play (interception /
    # fumble lost have epa < -1.5 by convention). nflverse drops the raw
    # interception flag from our PBP slice, so we approximate.
    plays["turnover_proxy"] = (plays["epa"] < -2.0) & (~plays["touchdown"].astype(bool))

    # Per (game, posteam) offense aggregates.
    off = (
        plays.groupby(["game_id", "posteam"], as_index=False)
        .agg(
            offense_plays=("epa", "size"),
            offense_yards=("yards_gained", "sum"),
            offense_red_zone_trips=("is_red_zone", "any"),  # placeholder
            offense_turnovers=("turnover_proxy", "sum"),
        )
        .rename(columns={"posteam": "team"})
    )
    # Real RZ trips: count distinct (game, drive) — but we don't store drive
    # IDs in pbp. Approximate by counting plays inside the 20.
    rz = (
        plays[plays["is_red_zone"]]
        .groupby(["game_id", "posteam"], as_index=False)
        .agg(offense_rz_plays=("epa", "size"))
        .rename(columns={"posteam": "team"})
    )
    off = off.merge(rz, on=["game_id", "team"], how="left")
    off["offense_rz_plays"] = off["offense_rz_plays"].fillna(0)
    off = off.drop(columns=["offense_red_zone_trips"])

    # Per (game, defteam) defense aggregates.
    deff = (
        plays.groupby(["game_id", "defteam"], as_index=False)
        .agg(
            defense_plays=("epa", "size"),
            defense_yards=("yards_gained", "sum"),
            defense_turnovers_forced=("turnover_proxy", "sum"),
        )
        .rename(columns={"defteam": "team"})
    )

    out = off.merge(deff, on=["game_id", "team"], how="outer")
    out["turnover_diff"] = (
        out["defense_turnovers_forced"].fillna(0)
        - out["offense_turnovers"].fillna(0)
    )
    # Drives proxy: 12 + (offense_plays - 60) * 0.1 — cheap heuristic since
    # we don't track drive IDs. Most teams have 10-13 drives per game.
    out["drives_proxy"] = 12 + (out["offense_plays"] - 60).clip(-30, 30) * 0.1
    out["yards_per_drive_off"] = out["offense_yards"] / out["drives_proxy"].replace(0, np.nan)
    out["yards_per_drive_def"] = out["defense_yards"] / out["drives_proxy"].replace(0, np.nan)
    out["rz_trip_rate"] = out["offense_rz_plays"] / out["drives_proxy"].replace(0, np.nan)
    return out[
        [
            "game_id", "team",
            "yards_per_drive_off", "yards_per_drive_def",
            "turnover_diff", "rz_trip_rate",
        ]
    ]


def build(games: pd.DataFrame) -> pd.DataFrame:
    if games.empty:
        return pd.DataFrame()

    per_game = _per_game_team_stats()
    if per_game.empty:
        return pd.DataFrame({"game_id": games["game_id"]})

    # Attach game_date so the rolling can sort properly.
    g = games[["game_id", "game_date"]].copy()
    g["game_date"] = pd.to_datetime(g["game_date"], errors="coerce")
    per_game = per_game.merge(g, on="game_id", how="left").sort_values(
        ["team", "game_date"]
    ).reset_index(drop=True)

    rolled = leakage_safe_rolling(
        per_game,
        group_col="team",
        sort_col="game_date",
        value_cols=[
            "yards_per_drive_off",
            "yards_per_drive_def",
            "turnover_diff",
            "rz_trip_rate",
        ],
        window=8,
        min_periods=2,
        weight="recency",
    )
    rolled["game_id"] = per_game["game_id"].values
    rolled["team"] = per_game["team"].values

    home = games[["game_id", "home_team"]].rename(columns={"home_team": "team"}).merge(
        rolled, on=["game_id", "team"], how="left"
    ).drop(columns=["team"])
    away = games[["game_id", "away_team"]].rename(columns={"away_team": "team"}).merge(
        rolled, on=["game_id", "team"], how="left"
    ).drop(columns=["team"])
    home = home.add_prefix("home_drv_").rename(columns={"home_drv_game_id": "game_id"})
    away = away.add_prefix("away_drv_").rename(columns={"away_drv_game_id": "game_id"})
    return home.merge(away, on="game_id", how="outer")
