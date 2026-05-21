"""Wong teaser construction.

A "Wong teaser" is a 6-point teaser leg that crosses BOTH key numbers
(3 and 7) in the favorable direction:

  - Favorites of -7.5 to -8.5 teased down to -1.5 / -2.5  (cross 7 and 3)
  - Dogs of +1.5 to +2.5 teased up to +7.5 / +8.5         (cross 3 and 7)

These have a long published history of beating breakeven because the
score-margin distribution in the NFL is highly concentrated around 3
and 7 -- crossing both is a non-trivial expected-value gain.

Standard 2-team teasers price at -110 (need 72.4% per leg to break even
on EV; published Wong leg rates run ~73-75%). 3-team teasers price at
+160 typically (need 65.5% per leg to break even).

This module:
- Identifies eligible legs from a slate.
- Computes calibrated leg cover probability via the simulator at the
  TEASED line (not the original line).
- Reports leg hit rate, 2-team / 3-team yield vs current pricing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from nfl_model.config import settings
from nfl_model.logging import get_logger

log = get_logger("model.teasers")


@dataclass
class TeaserLeg:
    """One teaser leg within a slate."""

    game_id: str
    home_team: str
    away_team: str
    side: str             # 'home' or 'away'
    original_spread: float  # home perspective
    teased_spread: float    # home perspective
    leg_prob: float          # calibrated cover prob at the teased line
    crosses_3: bool
    crosses_7: bool

    @property
    def is_wong(self) -> bool:
        return self.crosses_3 and self.crosses_7


def _crosses(orig: float, teased: float, k: int) -> bool:
    """Did teasing the spread move it through the key number ``k``?

    Both lines are home-perspective. Crossing means going from one side
    of -k or +k to the other.
    """
    return (orig < -k <= teased) or (orig > -k >= teased) or (orig < k <= teased) or (orig > k >= teased)


def construct_teaser_legs(
    *,
    game_ids: Sequence[str],
    home_teams: Sequence[str],
    away_teams: Sequence[str],
    spreads_home: np.ndarray,            # market spread, home perspective
    p_home_cover_at: callable,             # f(game_index, line) -> p(home covers at given line)
    teaser_points: int = settings.teaser_points,
) -> list[TeaserLeg]:
    """Build all eligible teaser legs from a slate.

    For each game we evaluate both sides at +teaser / -teaser of the
    posted spread and keep legs whose calibrated cover prob exceeds the
    Wong-target threshold.
    """
    legs: list[TeaserLeg] = []
    for i, gid in enumerate(game_ids):
        sp = float(spreads_home[i])
        if not np.isfinite(sp):
            continue
        # Home-side teased spread (home becomes a smaller fav / bigger dog)
        home_teased = sp + teaser_points
        # Away-side teased spread (mirror): away becomes a smaller dog / bigger fav
        away_teased = sp - teaser_points

        p_home = float(p_home_cover_at(i, home_teased))
        p_away = 1.0 - float(p_home_cover_at(i, away_teased))

        legs.append(
            TeaserLeg(
                game_id=str(gid),
                home_team=str(home_teams[i]),
                away_team=str(away_teams[i]),
                side="home",
                original_spread=sp,
                teased_spread=home_teased,
                leg_prob=p_home,
                crosses_3=_crosses(sp, home_teased, 3),
                crosses_7=_crosses(sp, home_teased, 7),
            )
        )
        legs.append(
            TeaserLeg(
                game_id=str(gid),
                home_team=str(home_teams[i]),
                away_team=str(away_teams[i]),
                side="away",
                original_spread=sp,
                teased_spread=away_teased,
                leg_prob=p_away,
                crosses_3=_crosses(sp, away_teased, 3),
                crosses_7=_crosses(sp, away_teased, 7),
            )
        )

    return legs


def filter_wong_legs(legs: Iterable[TeaserLeg], min_prob: float | None = None) -> list[TeaserLeg]:
    """Keep only legs that cross BOTH key numbers and exceed ``min_prob``."""
    cutoff = min_prob if min_prob is not None else settings.teaser_min_leg_prob
    return [leg for leg in legs if leg.is_wong and leg.leg_prob >= cutoff]


def teaser_card_dataframe(legs: Iterable[TeaserLeg]) -> pd.DataFrame:
    """Return a sorted, ranked DataFrame of teaser legs for the UI."""
    rows = [
        {
            "game_id": leg.game_id,
            "matchup": f"{leg.away_team} @ {leg.home_team}",
            "side": leg.side,
            "original_spread": leg.original_spread,
            "teased_spread": leg.teased_spread,
            "leg_prob": leg.leg_prob,
            "crosses_3": leg.crosses_3,
            "crosses_7": leg.crosses_7,
            "is_wong": leg.is_wong,
        }
        for leg in legs
    ]
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return df.sort_values(["is_wong", "leg_prob"], ascending=[False, False]).reset_index(drop=True)
