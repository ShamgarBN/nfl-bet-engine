"""Coach-vs-coach head-to-head features.

For each game between (home_coach, away_coach), the prior history of
this exact matchup — straight-up record and against-the-spread record
from the home coach's perspective. Some coach pairs have one-sided
histories (e.g. Belichick vs. various staffs in earlier eras); the
market doesn't always fully price the matchup.

Leakage-safe: every game's value reflects only meetings *strictly
before* the current game's date.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from nfl_model.data.warehouse import query
from nfl_model.logging import get_logger

log = get_logger("features.coach_matchup")


def _matchup_long() -> pd.DataFrame:
    """Long-format frame: one row per finalized game where both coaches known."""
    g = query(
        """
        SELECT g.game_id, g.game_date,
               g.home_coach, g.away_coach,
               g.home_score, g.away_score,
               o.spread_close
        FROM games g
        LEFT JOIN odds_history o
          ON o.game_id = g.game_id AND o.book = 'consensus'
        WHERE g.home_coach IS NOT NULL AND g.away_coach IS NOT NULL
          AND g.home_score IS NOT NULL AND g.away_score IS NOT NULL
        """
    )
    if g.empty:
        return g
    g["game_date"] = pd.to_datetime(g["game_date"], errors="coerce")
    g["home_margin"] = pd.to_numeric(g["home_score"], errors="coerce") - pd.to_numeric(
        g["away_score"], errors="coerce"
    )
    g["spread_close"] = pd.to_numeric(g["spread_close"], errors="coerce")
    g["home_won"] = (g["home_margin"] > 0).astype(float)
    cover_diff = g["home_margin"] + g["spread_close"]
    g["home_covered"] = np.where(
        g["spread_close"].notna(),
        (cover_diff > 0).astype(float),
        np.nan,
    )

    # Canonicalise the coach pair to (coach_lo, coach_hi) so the same
    # matchup keys lookups regardless of who happens to be home.
    pair_lo = np.minimum(g["home_coach"], g["away_coach"])
    pair_hi = np.maximum(g["home_coach"], g["away_coach"])
    g["pair_lo"] = pair_lo
    g["pair_hi"] = pair_hi
    # Whose perspective is "home" in this row, from the canonical pair?
    g["lo_is_home"] = (g["home_coach"] == g["pair_lo"]).astype(float)
    # lo_won: did the lo-coach win this meeting?
    g["lo_won"] = np.where(g["lo_is_home"] == 1, g["home_won"], 1.0 - g["home_won"])
    # lo_covered: did the lo-coach cover?
    g["lo_covered"] = np.where(
        g["lo_is_home"] == 1, g["home_covered"], 1.0 - g["home_covered"]
    )
    return g


def build(games: pd.DataFrame) -> pd.DataFrame:
    if games.empty:
        return pd.DataFrame()

    long = _matchup_long()
    if long.empty:
        return pd.DataFrame({"game_id": games["game_id"]})

    # Per canonical pair, expanding (shifted) mean of lo_won + lo_covered =
    # the *lo coach's* prior H2H rates against the *hi coach* going into
    # this row.
    long = long.sort_values(["pair_lo", "pair_hi", "game_date", "game_id"]).reset_index(drop=True)
    grp = long.groupby(["pair_lo", "pair_hi"], sort=False)
    long["pair_meetings"] = grp["lo_won"].transform(lambda s: s.shift(1).expanding().count())
    long["lo_h2h_winpct"] = grp["lo_won"].transform(lambda s: s.shift(1).expanding().mean())
    long["lo_h2h_atspct"] = grp["lo_covered"].transform(lambda s: s.shift(1).expanding().mean())

    # Convert to the home perspective for the row.
    long["home_coach_h2h_meetings"] = long["pair_meetings"]
    long["home_coach_h2h_winpct"] = np.where(
        long["lo_is_home"] == 1, long["lo_h2h_winpct"], 1.0 - long["lo_h2h_winpct"]
    )
    long["home_coach_h2h_atspct"] = np.where(
        long["lo_is_home"] == 1, long["lo_h2h_atspct"], 1.0 - long["lo_h2h_atspct"]
    )

    out_cols = ["game_id", "home_coach_h2h_meetings",
                "home_coach_h2h_winpct", "home_coach_h2h_atspct"]
    return long[out_cols].copy()
