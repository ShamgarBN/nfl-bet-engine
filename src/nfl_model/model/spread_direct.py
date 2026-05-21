"""Direct ATS classifier (separate from the simulator).

LightGBM binary classifier predicting "did the home team cover the
posted closing spread". Ensembled with the simulator's spread probability
in ``ensemble.py`` (weights walk-forward tuned).

The point of having a direct classifier in addition to the simulator:
the simulator gives you a coherent joint distribution but is implicitly
optimizing for score MSE. The direct classifier optimizes for the binary
spread-cover loss directly, which captures things like late-game garbage
time more efficiently. Blending them produces better calibrated edges
than either one alone.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

from nfl_model.config import settings
from nfl_model.logging import get_logger
from nfl_model.model.feature_matrix import build_runs_matrix, fit_spec

log = get_logger("model.spread_direct")


@dataclass
class SpreadClassifier:
    booster: lgb.Booster
    feature_cols: list[str]

    def predict_proba_home_cover(self, X: np.ndarray) -> np.ndarray:
        return self.booster.predict(X)


def train_spread_classifier(features: pd.DataFrame) -> dict[str, Any]:
    """Train and persist the ATS direct classifier."""
    df = features.copy()
    spread = pd.to_numeric(df.get("mkt_spread_close"), errors="coerce")
    home = pd.to_numeric(df["target_home_score"], errors="coerce")
    away = pd.to_numeric(df["target_away_score"], errors="coerce")
    margin = home - away
    y = np.where(np.isfinite(spread), (margin + spread) > 0, np.nan).astype(float)
    mask = np.isfinite(y)
    df = df[mask]
    y = y[mask]

    if len(df) < 200:
        log.warning("spread_direct.too_few_rows", n=len(df))
        return {"trained": False, "reason": "too few rows with market spread"}

    spec = fit_spec(df)
    X, _, feat_cols = build_runs_matrix(df, spec)
    spec.final_feature_cols = feat_cols

    cut = int(len(X) * 0.85)
    Xt, Xv = X[:cut], X[cut:]
    yt, yv = y[:cut], y[cut:]
    train_ds = lgb.Dataset(Xt, label=yt, feature_name=feat_cols)
    val_ds = lgb.Dataset(Xv, label=yv, reference=train_ds)
    params = {
        "objective": "binary", "metric": "binary_logloss",
        "learning_rate": 0.04, "num_leaves": 31, "min_data_in_leaf": 25,
        "feature_fraction": 0.8, "bagging_fraction": 0.85, "bagging_freq": 5,
        "lambda_l2": 1.0, "verbose": -1, "deterministic": True, "seed": 0,
    }
    booster = lgb.train(
        params, train_ds, num_boost_round=2000, valid_sets=[val_ds],
        callbacks=[lgb.early_stopping(80, verbose=False), lgb.log_evaluation(0)],
    )

    settings.model_dir.mkdir(parents=True, exist_ok=True)
    out_path = settings.model_dir / "spread_direct.joblib"
    joblib.dump(
        {"model_str": booster.model_to_string(), "feature_cols": feat_cols},
        out_path, compress=("zlib", 3),
    )
    return {"trained": True, "n": int(len(df)), "n_features": len(feat_cols), "path": str(out_path)}


def load_spread_classifier(path: Path | None = None) -> SpreadClassifier:
    p = path or (settings.model_dir / "spread_direct.joblib")
    payload = joblib.load(p)
    return SpreadClassifier(
        booster=lgb.Booster(model_str=payload["model_str"]),
        feature_cols=payload["feature_cols"],
    )
