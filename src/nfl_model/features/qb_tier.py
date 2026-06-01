"""QB tier rating.

Categorical rating for the team's projected starting QB based on the
prior season's EPA-per-dropback percentile. Five tiers:

- 1 = Elite (top 10%)
- 2 = High (top 30%)
- 3 = Average (middle 40%)
- 4 = Below avg (next 20%)
- 5 = Replacement (bottom 10%) — covers rookies + unknowns

Computed from ``qb_game_stats`` aggregated per (season-1, player_id),
attached to the team's projected starter via ``games.home_qb`` /
``games.away_qb`` matching against ``qb_game_stats.player_name``.

This is a simple ordinal feature but it captures the single most
important roster-level distinction in NFL game outcomes.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from nfl_model.data.warehouse import query
from nfl_model.logging import get_logger

log = get_logger("features.qb_tier")


def _prior_season_qb_tiers() -> pd.DataFrame:
    """Return DataFrame with columns ``[season, player_name, qb_tier]``.

    For each (player_name, season=S), tier is the percentile bucket of
    that player's epa_per_play across all attempts in season S-1.
    """
    df = query(
        """
        SELECT q.player_id, q.player_name, q.team, q.attempts, q.epa_per_play,
               g.season
        FROM qb_game_stats q
        JOIN games g USING(game_id)
        WHERE q.is_starter = TRUE AND q.attempts >= 10
        """
    )
    if df.empty:
        return pd.DataFrame()

    # Aggregate per (player, season): attempt-weighted mean EPA, total atts.
    df["epa_x_atts"] = df["epa_per_play"] * df["attempts"]
    agg = (
        df.groupby(["player_id", "player_name", "season"], as_index=False)
        .agg(total_atts=("attempts", "sum"), epa_sum=("epa_x_atts", "sum"))
    )
    agg = agg[agg["total_atts"] >= 150]  # need a real season's worth of pass
    agg["epa_per_play"] = agg["epa_sum"] / agg["total_atts"]
    agg = agg.drop(columns=["epa_sum"])

    # Per-season percentile rank → tier.
    def _tier_from_pct(pct: float) -> int:
        if pct >= 0.90:
            return 1
        if pct >= 0.70:
            return 2
        if pct >= 0.30:
            return 3
        if pct >= 0.10:
            return 4
        return 5

    agg["pct"] = agg.groupby("season")["epa_per_play"].rank(pct=True)
    agg["qb_tier"] = agg["pct"].apply(_tier_from_pct)

    # Tier we look up for season S is the player's tier from season S-1.
    agg["lookup_season"] = agg["season"] + 1

    return agg[["lookup_season", "player_name", "qb_tier"]].rename(
        columns={"lookup_season": "season"}
    )


def build(games: pd.DataFrame) -> pd.DataFrame:
    if games.empty:
        return pd.DataFrame()

    tiers = _prior_season_qb_tiers()
    if tiers.empty:
        return pd.DataFrame(
            {
                "game_id": games["game_id"],
                "home_qb_tier": 5,
                "away_qb_tier": 5,
            }
        )

    home = games[["game_id", "season", "home_qb"]].rename(columns={"home_qb": "player_name"})
    home = home.merge(tiers, on=["season", "player_name"], how="left").drop(
        columns=["player_name", "season"]
    )
    home["qb_tier"] = home["qb_tier"].fillna(5)  # unknown / rookie defaults to lowest tier
    home = home.rename(columns={"qb_tier": "home_qb_tier"})

    away = games[["game_id", "season", "away_qb"]].rename(columns={"away_qb": "player_name"})
    away = away.merge(tiers, on=["season", "player_name"], how="left").drop(
        columns=["player_name", "season"]
    )
    away["qb_tier"] = away["qb_tier"].fillna(5)
    away = away.rename(columns={"qb_tier": "away_qb_tier"})

    out = home.merge(away, on="game_id", how="outer")
    return out
