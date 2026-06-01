"""Offensive-line continuity features.

NFL OL play degrades meaningfully when starters rotate week-to-week.
For each game, we compute the fraction of the team's offensive line
that started last week as well, and a rolling 4-game average of that
fraction. Computed from ``snap_counts`` filtered to OL positions.

Leakage-safe: the calculation only looks at games strictly before the
target row.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from nfl_model.data.warehouse import query
from nfl_model.logging import get_logger

log = get_logger("features.ol_continuity")

_OL_POSITIONS = {"T", "G", "C", "OT", "OG", "OL"}
_STARTER_SNAP_THRESHOLD = 0.85


def _starters_per_game(team_filter: list[str] | None = None) -> pd.DataFrame:
    """Return one row per (game_id, team, player_id) for OL starters."""
    snaps = query(
        """
        SELECT s.game_id, s.team, s.player_id, s.position, s.offense_pct,
               g.game_date
        FROM snap_counts s
        JOIN games g USING(game_id)
        WHERE s.position IN ('T','G','C','OT','OG','OL')
        """
    )
    if snaps.empty:
        return snaps
    snaps["game_date"] = pd.to_datetime(snaps["game_date"], errors="coerce")
    snaps = snaps[snaps["offense_pct"].fillna(0) >= _STARTER_SNAP_THRESHOLD]
    return snaps


def build(games: pd.DataFrame) -> pd.DataFrame:
    if games.empty:
        return pd.DataFrame()

    starters = _starters_per_game()
    if starters.empty:
        return pd.DataFrame({"game_id": games["game_id"]})

    starters = starters.sort_values(["team", "game_date", "game_id"]).reset_index(drop=True)

    # Build (game_id, team) -> set of OL starter player_ids.
    sets = (
        starters.groupby(["game_id", "team", "game_date"])["player_id"]
        .apply(set)
        .reset_index()
        .rename(columns={"player_id": "ol_starters"})
        .sort_values(["team", "game_date", "game_id"])
        .reset_index(drop=True)
    )
    # Compare to the team's PREVIOUS game's starter set.
    sets["prev_starters"] = sets.groupby("team")["ol_starters"].shift(1)

    def _overlap(row: pd.Series) -> float:
        prev = row["prev_starters"]
        cur = row["ol_starters"]
        if not isinstance(prev, set) or not isinstance(cur, set) or not prev:
            return np.nan
        return len(prev & cur) / 5.0

    sets["ol_returning_share"] = sets.apply(_overlap, axis=1)

    # Rolling 4-game mean of returning share (leakage-safe via shift).
    sets["ol_continuity_r4"] = (
        sets.groupby("team")["ol_returning_share"]
        .apply(lambda s: s.shift(1).rolling(4, min_periods=1).mean())
        .reset_index(level=0, drop=True)
    )

    home = games[["game_id", "home_team"]].rename(columns={"home_team": "team"}).merge(
        sets[["game_id", "team", "ol_returning_share", "ol_continuity_r4"]],
        on=["game_id", "team"], how="left",
    ).drop(columns=["team"])
    away = games[["game_id", "away_team"]].rename(columns={"away_team": "team"}).merge(
        sets[["game_id", "team", "ol_returning_share", "ol_continuity_r4"]],
        on=["game_id", "team"], how="left",
    ).drop(columns=["team"])
    home = home.rename(columns={c: f"home_{c}" for c in ("ol_returning_share", "ol_continuity_r4")})
    away = away.rename(columns={c: f"away_{c}" for c in ("ol_returning_share", "ol_continuity_r4")})
    return home.merge(away, on="game_id", how="outer")
