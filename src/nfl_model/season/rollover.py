"""NFL season lifecycle predicates.

The NFL regular season runs early September through early January, with
playoffs through the Super Bowl in early February. Off-season is
February through August.

This module is the single source of truth for "what season is it?" so
callers don't have to invent ad-hoc heuristics.
"""

from __future__ import annotations

from datetime import date as date_cls
from typing import Literal

from nfl_model.data.warehouse import query
from nfl_model.logging import get_logger

log = get_logger("season.rollover")

# A regular season has 272 games (32 teams * 17 games / 2). We use a
# slightly lower threshold to gracefully handle minor data lags.
_MIN_FINALIZED_REGULAR_SEASON = 270
# Full season including 13 playoff games (incl. SB) = 285. Use 280 as a
# safe lower bound.
_MIN_FINALIZED_FULL_SEASON = 280

# Calendar boundaries (NFL season runs into the following calendar year).
_REGULAR_SEASON_END_MONTH = 1     # Final regular-season Sunday is early January
_SUPER_BOWL_MONTH = 2

SeasonState = Literal["in_progress", "ended", "off_season", "pre_season"]


def is_regular_season_over(season: int, *, today: date_cls | None = None) -> bool:
    """True iff ``season``'s regular season is in the books.

    NFL season X runs into calendar year X+1 (regular season ends in
    early January of X+1). Two-signal check guards against partial pulls.
    """
    today = today or date_cls.today()
    boundary = date_cls(season + 1, _REGULAR_SEASON_END_MONTH, 8)
    if today < boundary:
        return False

    df = query(
        "SELECT COUNT(*) AS n FROM games "
        "WHERE season = ? AND home_score IS NOT NULL AND game_type = 'REG'",
        (int(season),),
    )
    if df.empty:
        return False
    return int(df.iloc[0]["n"]) >= _MIN_FINALIZED_REGULAR_SEASON


def is_full_season_over(season: int, *, today: date_cls | None = None) -> bool:
    """True iff the playoffs (incl. Super Bowl) are also done."""
    today = today or date_cls.today()
    boundary = date_cls(season + 1, _SUPER_BOWL_MONTH, 15)
    if today < boundary:
        return False

    df = query(
        "SELECT COUNT(*) AS n FROM games WHERE season = ? AND home_score IS NOT NULL",
        (int(season),),
    )
    if df.empty:
        return False
    return int(df.iloc[0]["n"]) >= _MIN_FINALIZED_FULL_SEASON


def last_completed_season(today: date_cls | None = None) -> int | None:
    """Most recent season whose regular season is in the books, or None."""
    today = today or date_cls.today()
    # NFL convention: today's "current" season is last calendar year if Jan-Feb.
    starting = today.year - 1 if today.month <= 2 else today.year
    for season in range(starting, starting - 8, -1):
        if is_regular_season_over(season, today=today):
            return season
    return None
