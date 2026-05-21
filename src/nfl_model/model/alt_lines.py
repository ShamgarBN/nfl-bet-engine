"""Alt-spread / buy-through-key-number EV.

When the calibrated cover probability on a side justifies extra juice to
buy through a key number (e.g. moving from -2.5 to -1.5 through 2, or
+2.5 to +3.5 through 3), surface that as an alternate-line opportunity.

For each side at each key-cross alt line, compute:
- Required juice to break even (from calibrated p_cover)
- Current book juice (from input alt-line book pricing)
- EV-positive flag if calibrated juice > book juice (i.e. book is offering
  better than fair pricing)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from nfl_model.config import settings
from nfl_model.logging import get_logger

log = get_logger("model.alt_lines")


@dataclass
class AltLineOpportunity:
    game_id: str
    side: str            # 'home' or 'away'
    original_line: float
    alt_line: float
    p_cover: float
    fair_juice: int       # American odds reflecting p_cover
    book_juice: int | None  # what the sportsbook is offering, if known
    ev_positive: bool


def _prob_to_american(p: float) -> int:
    """Convert calibrated probability to American-odds break-even price."""
    p = max(min(p, 0.999), 0.001)
    if p >= 0.5:
        return -int(round(100 * p / (1 - p)))
    return int(round(100 * (1 - p) / p))


def _crosses_key(orig: float, alt: float, keys: tuple[int, ...]) -> bool:
    for k in keys:
        if (orig < -k <= alt) or (orig > -k >= alt) or (orig < k <= alt) or (orig > k >= alt):
            return True
    return False


def evaluate_alt_lines(
    *,
    game_ids: list[str],
    spreads_home: np.ndarray,
    p_home_cover_at: callable,
    keys: tuple[int, ...] = settings.key_numbers_spread,
    book_juice_lookup: dict | None = None,
    step: float = 1.0,
    n_steps: int = 4,
) -> list[AltLineOpportunity]:
    """Enumerate alt lines around posted spread for each game and score EV.

    Args:
        game_ids: list of game identifiers, length N.
        spreads_home: posted home-perspective spread per game.
        p_home_cover_at: callable (i, line) -> calibrated p(home covers at line).
        keys: key numbers we consider crossing through.
        book_juice_lookup: optional dict of {(game_id, side, alt_line): book_price}.
        step: step size between candidate alt lines (in points).
        n_steps: number of steps in each direction.
    """
    out: list[AltLineOpportunity] = []
    for i, gid in enumerate(game_ids):
        sp = float(spreads_home[i])
        if not np.isfinite(sp):
            continue
        for k in range(1, n_steps + 1):
            for side, alt_home, alt_away in (
                ("home", sp + k * step, sp - k * step),
                ("away", sp - k * step, sp + k * step),
            ):
                alt_line = alt_home if side == "home" else alt_away
                if not _crosses_key(sp, alt_line, keys):
                    continue
                if side == "home":
                    p = float(p_home_cover_at(i, alt_line))
                else:
                    p = 1.0 - float(p_home_cover_at(i, alt_line))
                fair = _prob_to_american(p)
                book = None
                ev_positive = False
                if book_juice_lookup:
                    book = book_juice_lookup.get((gid, side, alt_line))
                    if book is not None:
                        # Book is good for us if its implied prob is below ours
                        from nfl_model.features.market import _american_to_implied
                        import pandas as pd
                        book_p = float(_american_to_implied(pd.Series([book])).iloc[0])
                        ev_positive = p > book_p

                out.append(
                    AltLineOpportunity(
                        game_id=str(gid),
                        side=side,
                        original_line=sp,
                        alt_line=alt_line,
                        p_cover=p,
                        fair_juice=fair,
                        book_juice=book,
                        ev_positive=ev_positive,
                    )
                )
    return out
