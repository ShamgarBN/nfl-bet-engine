"""Coaching features.

Per-coach rolling: career win %, post-bye record, 2nd-half scoring trend.
For first cut we attach simple coach-tenure proxies; richer staff-level
features (OC/DC tenure, situational tendencies) layered in later.
"""

from __future__ import annotations

import pandas as pd

from nfl_model.data.warehouse import query
from nfl_model.logging import get_logger

log = get_logger("features.coaching")


def build(games: pd.DataFrame) -> pd.DataFrame:
    if games.empty:
        return pd.DataFrame()

    df = games[["game_id", "season", "home_team", "away_team", "home_coach", "away_coach"]].copy()

    coach_records = query(
        """
        SELECT season, home_team AS team, home_coach AS coach,
               (home_score > away_score) AS won
        FROM games
        WHERE home_coach IS NOT NULL AND home_score IS NOT NULL
        UNION ALL
        SELECT season, away_team AS team, away_coach AS coach,
               (away_score > home_score) AS won
        FROM games
        WHERE away_coach IS NOT NULL AND away_score IS NOT NULL
        """
    )
    if coach_records.empty:
        return pd.DataFrame({"game_id": df["game_id"]})

    coach_records["won"] = coach_records["won"].astype(float)
    cum = coach_records.sort_values(["coach", "season"]).groupby("coach")["won"].expanding().mean().shift(1)
    cum = cum.reset_index().rename(columns={"won": "coach_career_win_pct"})

    # Approximate by joining on coach + season alignment (good enough for first cut)
    summary = (
        coach_records.groupby(["coach", "season"]).agg(coach_season_games=("won", "count"),
                                                        coach_season_win_pct=("won", "mean"))
        .reset_index()
    )

    home = df[["game_id", "season", "home_coach"]].rename(columns={"home_coach": "coach"})
    home = home.merge(summary, on=["coach", "season"], how="left")
    home = home.rename(columns={c: f"home_{c}" for c in ["coach_season_games", "coach_season_win_pct"]})

    away = df[["game_id", "season", "away_coach"]].rename(columns={"away_coach": "coach"})
    away = away.merge(summary, on=["coach", "season"], how="left")
    away = away.rename(columns={c: f"away_{c}" for c in ["coach_season_games", "coach_season_win_pct"]})

    out = home.drop(columns=["coach", "season"]).merge(
        away.drop(columns=["coach", "season"]), on="game_id", how="outer"
    )
    return out
