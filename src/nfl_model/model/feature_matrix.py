"""Feature matrix assembly: turn the wide assembled features DataFrame
into numpy arrays consumed by the LightGBM models.

Centralizes:
- Imputation policy (median for numerics; one-hot for low-card categoricals).
- Locking the feature column list at fit time so inference matches.
- The home/away flip used when training a single shared score-distribution
  model on doubled rows (one row per team-game) instead of two separate
  home and away models.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from nfl_model.logging import get_logger

log = get_logger("model.feature_matrix")


# Feature columns we never train on — keys, targets, leaky labels.
DROP_COLS = {
    "game_id", "season", "week", "game_date", "kickoff_ts",
    "home_team", "away_team", "home_qb", "away_qb", "home_coach", "away_coach",
    "home_score", "away_score", "home_win", "overtime",
    "stadium_id", "roof", "surface", "weekday", "referee",
    "target_home_score", "target_away_score", "target_home_win",
    "target_total", "target_margin",
    "feature_json",
}


@dataclass
class FeatureSpec:
    """Locked feature contract — column order + imputation values."""

    feature_cols: list[str] = field(default_factory=list)
    medians: dict[str, float] = field(default_factory=dict)
    final_feature_cols: list[str] = field(default_factory=list)

    def to_json(self) -> str:
        return json.dumps(
            {
                "feature_cols": self.feature_cols,
                "medians": self.medians,
                "final_feature_cols": self.final_feature_cols,
            }
        )


def _numeric_columns(df: pd.DataFrame) -> list[str]:
    return [
        c for c in df.columns
        if c not in DROP_COLS
        and pd.api.types.is_numeric_dtype(df[c])
    ]


def _bool_to_numeric(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in out.columns:
        if pd.api.types.is_bool_dtype(out[c]):
            out[c] = out[c].astype(float)
    return out


def fit_spec(features: pd.DataFrame) -> FeatureSpec:
    """Lock the feature contract on the training set.

    We only commit medians for numeric columns. Boolean -> float is done
    on the fly in build_*_matrix.
    """
    df = _bool_to_numeric(features)
    cols = _numeric_columns(df)
    medians = {c: float(df[c].median()) if df[c].notna().any() else 0.0 for c in cols}
    return FeatureSpec(feature_cols=cols, medians=medians, final_feature_cols=cols)


def _materialize(features: pd.DataFrame, spec: FeatureSpec, perspective: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Materialize a numeric matrix from the features DataFrame.

    perspective='home' uses home_* prefixed columns and treats away_* as
    opponent. perspective='away' is the mirror. The resulting model can
    be trained on the union of both perspectives so each team-game is a
    training row.
    """
    df = _bool_to_numeric(features)
    cols = list(spec.final_feature_cols or spec.feature_cols)
    if perspective == "away":
        # Swap home_* <-> away_* by renaming (only for prefix groups we know about)
        rename = {}
        for c in cols:
            if c.startswith("home_"):
                rename[c] = "away_" + c[len("home_"):]
            elif c.startswith("away_"):
                rename[c] = "home_" + c[len("away_"):]
        df = df.rename(columns=rename)

    available = [c for c in cols if c in df.columns]
    missing = [c for c in cols if c not in df.columns]
    if missing:
        for m in missing:
            df[m] = spec.medians.get(m, 0.0)

    X = df[cols].to_numpy(dtype=np.float64)
    # Impute NaNs with stored medians
    for j, c in enumerate(cols):
        col = X[:, j]
        nan_mask = np.isnan(col)
        if nan_mask.any():
            X[nan_mask, j] = spec.medians.get(c, 0.0)
    valid = np.ones(len(df), dtype=bool)  # could refine to require some min coverage
    return X, valid, cols


def build_runs_matrix(features: pd.DataFrame, spec: FeatureSpec) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Build the home-perspective matrix for predicting home team's points."""
    return _materialize(features, spec, "home")


def build_runs_matrix_away(features: pd.DataFrame, spec: FeatureSpec) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Build the away-perspective matrix for predicting away team's points."""
    return _materialize(features, spec, "away")
