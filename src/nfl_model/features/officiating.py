"""Officiating-crew features.

Crew-level historical penalty rate, yards/game, and total-points moved by
crew. Built lazily; falls back to NaN when officiating table is sparse.
"""

from __future__ import annotations

import pandas as pd

from nfl_model.data.warehouse import query
from nfl_model.logging import get_logger

log = get_logger("features.officiating")


def build(games: pd.DataFrame) -> pd.DataFrame:
    if games.empty:
        return pd.DataFrame()

    off = query(
        """
        SELECT o.game_id, g.season, g.game_date, o.referee, o.crew_id,
               o.penalties_total, o.penalty_yards, o.total_points
        FROM officiating o JOIN games g USING(game_id)
        ORDER BY g.game_date
        """
    )
    if off.empty:
        return pd.DataFrame({"game_id": games["game_id"]})

    off["game_date"] = pd.to_datetime(off["game_date"], errors="coerce")
    off = off.sort_values(["referee", "game_date"]).reset_index(drop=True)

    # Leakage-safe rolling
    off["ref_pen_avg"] = off.groupby("referee")["penalties_total"].shift(1).expanding().mean().reset_index(level=0, drop=True)
    off["ref_pen_yards_avg"] = off.groupby("referee")["penalty_yards"].shift(1).expanding().mean().reset_index(level=0, drop=True)
    off["ref_total_pts_avg"] = off.groupby("referee")["total_points"].shift(1).expanding().mean().reset_index(level=0, drop=True)

    keep = ["game_id", "ref_pen_avg", "ref_pen_yards_avg", "ref_total_pts_avg"]
    out = games[["game_id"]].merge(off[keep], on="game_id", how="left")
    return out
