"""Optuna walk-forward hyperparameter + ensemble-weight tuning.

Defines:
- ``run_tuning``: search LightGBM params + ensemble blend weights to
  maximize top-decile ATS subject to a min-sample constraint.
- ``run_bakeoff``: tournament of named candidate configs; auto-promote
  per-market winners.

The objective always uses walk-forward predictions so the search itself
doesn't leak future data.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from nfl_model.logging import get_logger

log = get_logger("backtest.tune")


def run_tuning(start_season: int, end_season: int, *, n_trials: int = 50) -> dict[str, Any]:
    """Optuna search over a small grid.

    Stub: implementation runs the walk-forward backtest under different
    ensemble blends and LightGBM params, picks the best. For first cut
    we return a placeholder dict and let the iteration phase wire up a
    real Optuna study.
    """
    log.info("tune.placeholder", start=start_season, end=end_season, trials=n_trials)
    return {
        "best": {
            "ensemble_weights": {"spread": 0.55, "ml": 0.65, "total": 0.50},
            "lgbm": {"learning_rate": 0.04, "num_leaves": 31, "min_data_in_leaf": 25},
        },
        "best_score_top10_acc": float("nan"),
        "n_trials": n_trials,
    }


def run_bakeoff(start_season: int, end_season: int) -> dict[str, tuple[str, float]]:
    """Run a small tournament of candidate configs; return per-market winners.

    Stub: returns a dictionary mapping market -> (winner_name, score).
    The full implementation lives in the iteration loop and will:
    - Fork the assembled features and walkforward into N candidate
      pipelines.
    - Score each candidate per market on a held-out tail of the
      walk-forward window.
    - Save winning configs to ``models/candidates/``.
    """
    log.info("bakeoff.placeholder", start=start_season, end=end_season)
    return {
        "spread": ("simulator+direct(0.55)", float("nan")),
        "ml": ("simulator+direct(0.65)", float("nan")),
        "total": ("simulator+direct(0.50)", float("nan")),
    }
