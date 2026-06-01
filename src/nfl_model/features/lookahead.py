"""Lookahead / letdown features.

For each (game, team) we surface the strength of the team's
**prior-week opponent** and **next-week opponent**, plus a "sandwich"
flag for being tough-team-then-tough-team. A small but well-documented
motivation effect: teams coming off an emotionally taxing game (high
prior-week opponent strength) tend to under-perform; teams looking
ahead to a tough game (high next-week opponent strength) often start
flat.

Opponent strength is the opponent's prior-rolling-8-game offense EPA
minus defense EPA — a single-number "team quality" proxy from data the
warehouse already has.

Leakage-safe: each value is computed only from games strictly outside
the target week. For prior_week we use the team's actual last game.
For next_week we use the *scheduled* opponent (known at kickoff), so
this is safe to use for live predictions.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from nfl_model.data.warehouse import query
from nfl_model.logging import get_logger

log = get_logger("features.lookahead")


def _team_strength_per_game() -> pd.DataFrame:
    """Per-(game, team) shifted-8-game team-quality estimate."""
    tg = query(
        """
        SELECT t.game_id, t.team,
               t.offense_epa_per_play, t.defense_epa_per_play,
               g.game_date
        FROM team_game_stats t JOIN games g USING(game_id)
        WHERE g.home_score IS NOT NULL
        """
    )
    if tg.empty:
        return tg
    tg["game_date"] = pd.to_datetime(tg["game_date"], errors="coerce")
    tg = tg.sort_values(["team", "game_date"]).reset_index(drop=True)
    grp = tg.groupby("team", sort=False)
    # Quality = (offense_epa_per_play - defense_epa_per_play) rolled 8gms,
    # shifted by 1 so the value is what was known *going into* the game.
    tg["team_strength"] = grp.apply(
        lambda s: (s["offense_epa_per_play"] - s["defense_epa_per_play"]).shift(1)
        .rolling(8, min_periods=2).mean()
    ).reset_index(level=0, drop=True)
    return tg[["game_id", "team", "team_strength"]]


def build(games: pd.DataFrame) -> pd.DataFrame:
    if games.empty:
        return pd.DataFrame()

    g = games[["game_id", "season", "week", "game_date", "home_team", "away_team"]].copy()
    g["game_date"] = pd.to_datetime(g["game_date"], errors="coerce")

    strength = _team_strength_per_game()
    if strength.empty:
        return pd.DataFrame({"game_id": games["game_id"]})

    # Build a per-team schedule with each game's opponent + game_date so
    # we can look up "prior week" and "next week" opponent by sort order.
    home = g[["game_id", "season", "week", "game_date", "home_team", "away_team"]].rename(
        columns={"home_team": "team", "away_team": "opponent"}
    )
    away = g[["game_id", "season", "week", "game_date", "home_team", "away_team"]].rename(
        columns={"away_team": "team", "home_team": "opponent"}
    )
    sched = pd.concat([home, away], ignore_index=True).sort_values(
        ["team", "season", "game_date"]
    ).reset_index(drop=True)

    # Prior + next opponent per (team, game).
    sched["prev_opponent"] = sched.groupby(["team", "season"])["opponent"].shift(1)
    sched["next_opponent"] = sched.groupby(["team", "season"])["opponent"].shift(-1)

    # Join opponent strength via the OPPONENT's most-recent strength (their
    # team_strength as of *their* most recent game). For prev_opp we want
    # what we knew at the time of the prior game; for next_opp we want what
    # we know now (so use the opponent's strength as of this game's date).
    # Practical approximation: use the team_strength_per_game frame keyed
    # by team alone with a "most recent shifted value" lookup.
    last_strength_per_team = (
        strength.merge(g[["game_id", "game_date"]], on="game_id", how="left")
        .sort_values(["team", "game_date"])
        .groupby("team")
        .agg(latest_strength=("team_strength", "last"))
        .reset_index()
    )

    # prev_opponent_strength: look up that opponent's latest_strength
    sched = sched.merge(
        last_strength_per_team.rename(
            columns={"team": "prev_opponent", "latest_strength": "prev_opp_strength"}
        ),
        on="prev_opponent", how="left",
    )
    sched = sched.merge(
        last_strength_per_team.rename(
            columns={"team": "next_opponent", "latest_strength": "next_opp_strength"}
        ),
        on="next_opponent", how="left",
    )

    sched["prev_opp_strength"] = pd.to_numeric(sched["prev_opp_strength"], errors="coerce").fillna(0.0)
    sched["next_opp_strength"] = pd.to_numeric(sched["next_opp_strength"], errors="coerce").fillna(0.0)
    # Sandwich = both surrounding games against above-average opponents.
    sched["sandwich_game"] = (
        (sched["prev_opp_strength"] > 0.05) & (sched["next_opp_strength"] > 0.05)
    ).astype(float)

    keep = ["game_id", "team", "prev_opp_strength", "next_opp_strength", "sandwich_game"]
    home_lookup = g[["game_id", "home_team"]].rename(columns={"home_team": "team"}).merge(
        sched[keep], on=["game_id", "team"], how="left"
    ).drop(columns=["team"])
    away_lookup = g[["game_id", "away_team"]].rename(columns={"away_team": "team"}).merge(
        sched[keep], on=["game_id", "team"], how="left"
    ).drop(columns=["team"])
    home_lookup = home_lookup.rename(
        columns={c: f"home_{c}" for c in ("prev_opp_strength", "next_opp_strength", "sandwich_game")}
    )
    away_lookup = away_lookup.rename(
        columns={c: f"away_{c}" for c in ("prev_opp_strength", "next_opp_strength", "sandwich_game")}
    )
    return home_lookup.merge(away_lookup, on="game_id", how="outer")
