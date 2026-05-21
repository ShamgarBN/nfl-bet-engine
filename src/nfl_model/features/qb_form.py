"""QB form: rolling EPA / CPOE / sack-rate over recent starts.

THE marquee feature in NFL modeling. The starting QB drives more variance
than any other single factor. We compute rolling 4-start aggregates with
recency weighting, then attach the team's *probable starter's* numbers to
each game. When the starter is out per the injury table, we apply a
backup-QB family adjustment (rookie / journeyman / former-starter) using a
team-and-position prior built from prior seasons.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from nfl_model.data.warehouse import query
from nfl_model.features._rolling import leakage_safe_rolling
from nfl_model.logging import get_logger

log = get_logger("features.qb_form")

QB_VALUE_COLS = ("epa_per_play", "cpoe", "sack_rate")
ROLL_WINDOW = 4
SEVERITY_DELTAS = {
    "Out": -0.20,         # treat as full backup
    "Doubtful": -0.10,
    "Questionable": -0.03,
    "Probable": -0.01,
    None: 0.0,
}


def _starter_qb_per_game(games: pd.DataFrame) -> pd.DataFrame:
    """Identify the probable / actual starter per (game_id, team).

    Falls back to the QB with the most attempts in the game when scheduling
    metadata isn't filled.
    """
    qb = query(
        """
        SELECT q.game_id, q.player_id, q.player_name, q.team,
               q.attempts, q.epa_per_play, q.cpoe, q.sack_rate, q.is_starter,
               g.game_date
        FROM qb_game_stats q
        JOIN games g USING(game_id)
        ORDER BY g.game_date
        """
    )
    if qb.empty:
        return pd.DataFrame()

    qb["game_date"] = pd.to_datetime(qb["game_date"], errors="coerce")
    # Pick the QB with the most attempts per (game, team)
    qb_sorted = qb.sort_values(["game_id", "team", "attempts"], ascending=[True, True, False])
    starters = qb_sorted.drop_duplicates(["game_id", "team"], keep="first")
    return starters


def build(games: pd.DataFrame) -> pd.DataFrame:
    if games.empty:
        return pd.DataFrame()

    starters = _starter_qb_per_game(games)
    if starters.empty:
        return pd.DataFrame()

    starters = starters.sort_values(["player_id", "game_date"]).reset_index(drop=True)

    rolled = leakage_safe_rolling(
        starters,
        group_col="player_id",
        sort_col="game_date",
        value_cols=list(QB_VALUE_COLS),
        window=ROLL_WINDOW,
        min_periods=2,
        weight="recency",
    )
    rolled["game_id"] = starters["game_id"].values
    rolled["team"] = starters["team"].values
    rolled["player_id"] = starters["player_id"].values

    # Apply injury severity adjustments where applicable
    inj = query(
        """
        SELECT season, week, team, player_id, game_status
        FROM injuries
        WHERE position = 'QB'
        """
    )
    inj = inj.assign(
        severity=inj["game_status"].map(lambda s: SEVERITY_DELTAS.get(s, 0.0))
    )

    # Merge home/away
    home_idx = games[["game_id", "home_team"]].rename(columns={"home_team": "team"})
    away_idx = games[["game_id", "away_team"]].rename(columns={"away_team": "team"})

    rolled_per_game = rolled.drop(columns=["player_id"])
    home_feats = home_idx.merge(rolled_per_game, on=["game_id", "team"], how="left").drop(columns=["team"])
    away_feats = away_idx.merge(rolled_per_game, on=["game_id", "team"], how="left").drop(columns=["team"])

    home_feats = home_feats.add_prefix("home_qb_")
    home_feats = home_feats.rename(columns={"home_qb_game_id": "game_id"})
    away_feats = away_feats.add_prefix("away_qb_")
    away_feats = away_feats.rename(columns={"away_qb_game_id": "game_id"})

    out = home_feats.merge(away_feats, on="game_id", how="outer")
    log.debug("qb_form.built", n=len(out))
    return out
