"""Season state detection + end-of-season sweep."""

from __future__ import annotations

from nfl_model.season.end_of_season import (
    EndOfSeasonReport,
    run_end_of_season_sweep,
)
from nfl_model.season.rollover import (
    is_full_season_over,
    is_regular_season_over,
    last_completed_season,
)
from nfl_model.season.state import SeasonStatus, detect_season_state

__all__ = [
    "EndOfSeasonReport",
    "SeasonStatus",
    "detect_season_state",
    "is_full_season_over",
    "is_regular_season_over",
    "last_completed_season",
    "run_end_of_season_sweep",
]
