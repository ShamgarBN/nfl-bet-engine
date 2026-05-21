"""Kicker quality features.

Per-team rolling FG% by distance bucket, recent miss streak, weather-
adjusted FG% (small but real for tight spreads). Pulled as a per-team
aggregate from the snap-counts/PBP since dedicated kicker tables aren't
stored separately yet.
"""

from __future__ import annotations

import pandas as pd

from nfl_model.data.warehouse import query
from nfl_model.logging import get_logger

log = get_logger("features.kicking")


def build(games: pd.DataFrame) -> pd.DataFrame:
    """Stub kicker features.

    Real implementation pulls FG attempts + makes by distance from PBP
    and computes leakage-safe rolling. For first cut we return an empty
    DataFrame so the assembler skips this builder cleanly.
    """
    if games.empty:
        return pd.DataFrame()
    # Placeholder: surface a single column so the model has *something* and
    # ablation can measure marginal value of nothing-vs-something.
    return pd.DataFrame({
        "game_id": games["game_id"],
        "home_kicker_fg_pct_roll": pd.NA,
        "away_kicker_fg_pct_roll": pd.NA,
    })
