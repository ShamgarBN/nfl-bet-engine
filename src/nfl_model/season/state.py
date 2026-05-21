"""NFL season state detection.

The NFL regular season runs ~Sep through early Jan, with playoffs through
the Super Bowl in early Feb. Off-season is Feb-Aug.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from nfl_model.logging import get_logger

log = get_logger("season.state")


@dataclass(frozen=True)
class SeasonStatus:
    """Snapshot of where we are in the NFL season life cycle."""

    season: int
    state: str  # 'pre_season' | 'in_progress' | 'ended' | 'off_season'
    finalized_games: int
    description: str


def _current_season_year(today: date | None = None) -> int:
    """Return the season year for ``today``.

    NFL convention: the 2024 season runs Sep 2024 through Feb 2025. So
    Jan-Feb dates belong to the previous season's playoffs.
    """
    today = today or date.today()
    if today.month <= 2:
        return today.year - 1
    return today.year


def detect_season_state(today: date | None = None) -> SeasonStatus:
    """Detect the current season state from the warehouse.

    Returns a coarse classification used by the weekly-train job to know
    when to trigger the end-of-season sweep.
    """
    from nfl_model.data.warehouse import query, table_row_count

    today = today or date.today()
    season = _current_season_year(today)

    if table_row_count("games") == 0:
        return SeasonStatus(
            season=season,
            state="pre_season",
            finalized_games=0,
            description="Warehouse empty; pull data first.",
        )

    df = query(
        "SELECT COUNT(*) AS n_finalized "
        "FROM games WHERE season = ? AND home_score IS NOT NULL",
        (season,),
    )
    n_finalized = int(df.iloc[0]["n_finalized"])

    # NFL has 272 regular-season games + 11 playoff games + 1 Super Bowl = 284.
    # Use 280 as a "season is over" threshold to handle quirks.
    if n_finalized == 0:
        if today.month >= 9 or today.month == 1:
            return SeasonStatus(
                season=season,
                state="pre_season",
                finalized_games=0,
                description="Season has been scheduled but no games finalized yet.",
            )
        return SeasonStatus(
            season=season,
            state="off_season",
            finalized_games=0,
            description="Off-season period.",
        )

    if n_finalized < 280:
        return SeasonStatus(
            season=season,
            state="in_progress",
            finalized_games=n_finalized,
            description=f"{n_finalized}/284 games finalized.",
        )

    return SeasonStatus(
        season=season,
        state="ended",
        finalized_games=n_finalized,
        description="Season complete; ready for end-of-season sweep.",
    )
