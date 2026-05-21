"""Stage B: Monte Carlo simulator.

Given per-team mean + std score predictions, draw correlated home/away
scores per simulation iteration. The correlation comes from a shared
"game environment" multiplier (a unit-mean lognormal draw per game) that
captures shared scoring conditions both teams face -- weather, pace, the
overall totals environment.

From the paired draws we read off:
- p(home wins) -> moneyline
- p(home covers spread) -> ATS at the market line (or any input line)
- p(over total) -> totals at the market line

NFL-specific tweaks vs MLB:
- Force scores to be non-negative integers (NFL teams can score 0).
- Ties are real (regular season especially); track them as their own class.
- Account for the fact that field goals + TDs make scores discrete with
  sub-distributions; we fold this into the std and let the Normal
  approximation handle it (NFL final-score distributions are
  remarkably close to Normal once you're past the first few percentiles).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from nfl_model.config import settings
from nfl_model.logging import get_logger

log = get_logger("model.simulate")


@dataclass
class GamePrediction:
    """One game's joint-distribution probabilities."""

    game_id: str
    p_home_win: float
    p_home_cover: float            # at the *input* spread
    p_total_over: float             # at the input total
    p_tie: float
    expected_home_score: float
    expected_away_score: float
    expected_margin: float
    expected_total: float
    spread_used: float
    total_used: float


def simulate_games(
    *,
    game_ids: Sequence[str],
    pred_home: tuple[np.ndarray, np.ndarray],
    pred_away: tuple[np.ndarray, np.ndarray],
    spread_lines: np.ndarray,                     # home perspective; -3 = home -3
    total_lines: np.ndarray,
    n_sims: int = settings.monte_carlo_iterations,
    env_sigma: float = 0.10,
    seed: int = 0,
) -> list[GamePrediction]:
    """Run Stage B Monte Carlo for a slate of games.

    Args:
        game_ids: identifiers, length N.
        pred_home: (mean, std) arrays for the home team, each length N.
        pred_away: (mean, std) arrays for the away team, each length N.
        spread_lines: market spread (home perspective) per game; NaN allowed.
        total_lines: market total per game; NaN allowed.
        n_sims: Monte Carlo draws per game.
        env_sigma: lognormal sigma of the shared game-environment factor.
            Higher = more scoring-correlation between home and away.
    """
    rng = np.random.default_rng(seed)
    h_mean, h_std = pred_home
    a_mean, a_std = pred_away
    n = len(game_ids)
    results: list[GamePrediction] = []

    # Per-game iterate so we can vary spread_line + total_line independently.
    for i in range(n):
        env = rng.lognormal(mean=-0.5 * env_sigma**2, sigma=env_sigma, size=n_sims)
        # Draw correlated home + away samples (independent given env)
        home = rng.normal(loc=h_mean[i], scale=max(h_std[i], 1e-3), size=n_sims) * env
        away = rng.normal(loc=a_mean[i], scale=max(a_std[i], 1e-3), size=n_sims) * env
        # Snap to non-negative integers (rounding to nearest).
        home = np.clip(np.round(home), 0, None)
        away = np.clip(np.round(away), 0, None)

        margin = home - away
        total = home + away
        p_home_win = float(np.mean(margin > 0))
        p_tie = float(np.mean(margin == 0))

        sp = spread_lines[i] if i < len(spread_lines) else np.nan
        if np.isfinite(sp):
            # Cover when (home_score - away_score) + spread > 0  (home perspective)
            covers = (margin + sp) > 0
            p_home_cover = float(np.mean(covers))
        else:
            # Fall back to >half-point spread implied from margin median
            p_home_cover = p_home_win

        tl = total_lines[i] if i < len(total_lines) else np.nan
        if np.isfinite(tl):
            p_total_over = float(np.mean(total > tl))
        else:
            p_total_over = float(np.mean(total > np.median(total)))

        results.append(
            GamePrediction(
                game_id=str(game_ids[i]),
                p_home_win=p_home_win,
                p_home_cover=p_home_cover,
                p_total_over=p_total_over,
                p_tie=p_tie,
                expected_home_score=float(np.mean(home)),
                expected_away_score=float(np.mean(away)),
                expected_margin=float(np.mean(margin)),
                expected_total=float(np.mean(total)),
                spread_used=float(sp) if np.isfinite(sp) else float("nan"),
                total_used=float(tl) if np.isfinite(tl) else float("nan"),
            )
        )
    return results
