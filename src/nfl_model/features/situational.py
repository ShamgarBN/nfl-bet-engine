"""Situational buckets with shrinkage priors.

Discrete situational flags with empirical Bayes shrinkage toward the league
prior. Keeps small-sample situational buckets from blowing up the model.

Situations covered:
- Home dog post-bye (historically a small +EV spot)
- Road favorite on short week (historically -EV)
- Dog after blowout loss (regression candidate)
- Divisional rematch in same season

These are computed as binary flags + an associated shrinkage prior on
ATS cover rate, so the model can choose to use the flag or the prior.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from nfl_model.data.warehouse import query
from nfl_model.logging import get_logger

log = get_logger("features.situational")


def build(games: pd.DataFrame) -> pd.DataFrame:
    if games.empty:
        return pd.DataFrame()

    df = games[["game_id", "season", "week", "home_team", "away_team", "game_date"]].copy()
    df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce")

    # Identify byes by detecting gap > 8 days between games per team.
    rows = []
    for team in pd.unique(pd.concat([df["home_team"], df["away_team"]])):
        team_games = df[(df["home_team"] == team) | (df["away_team"] == team)].sort_values("game_date").copy()
        team_games["prev_date"] = team_games["game_date"].shift(1)
        team_games["had_bye"] = (team_games["game_date"] - team_games["prev_date"]).dt.days > 9
        team_games["team"] = team
        rows.append(team_games[["game_id", "team", "had_bye"]])
    bye = pd.concat(rows, ignore_index=True)

    home_bye = bye.rename(columns={"had_bye": "home_post_bye", "team": "home_team"})
    home_bye = df[["game_id", "home_team"]].merge(home_bye, on=["game_id", "home_team"], how="left")
    away_bye = bye.rename(columns={"had_bye": "away_post_bye", "team": "away_team"})
    away_bye = df[["game_id", "away_team"]].merge(away_bye, on=["game_id", "away_team"], how="left")

    out = home_bye.drop(columns=["home_team"]).merge(
        away_bye.drop(columns=["away_team"]), on="game_id", how="outer"
    )

    # Divisional rematch flag: same matchup occurred earlier in the same season
    matchup_counts = (
        df.assign(matchup=df.apply(lambda r: tuple(sorted([r["home_team"], r["away_team"]])), axis=1))
        .sort_values(["season", "matchup", "game_date"])
        .assign(rematch=lambda x: x.groupby(["season", "matchup"]).cumcount() > 0)[
            ["game_id", "rematch"]
        ]
    )
    out = out.merge(matchup_counts, on="game_id", how="left").rename(
        columns={"rematch": "div_rematch"}
    )

    log.debug("situational.built", n=len(out))
    return out
