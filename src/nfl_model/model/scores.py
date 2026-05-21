"""Stage A: per-team score distribution models.

LightGBM regressor predicts each team's expected points; a paired
quantile regressor (or residual std) gives us a width estimate. Together
they specify a Normal-like score distribution per team-game which Stage B
samples from.

Multi-seed bagging: train ``n_seeds`` models with different bootstrap
seeds and average their mean / std predictions to reduce variance.
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

from nfl_model.config import settings
from nfl_model.logging import get_logger
from nfl_model.model.feature_matrix import (
    FeatureSpec,
    build_runs_matrix,
    build_runs_matrix_away,
    fit_spec,
)

log = get_logger("model.scores")


@dataclass
class ScoresModel:
    """A bagged LightGBM mean+std team-score predictor."""

    mean_boosters: list[lgb.Booster]
    std_boosters: list[lgb.Booster]
    feature_cols: list[str]

    def predict_distribution(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return (mean, std) arrays for each row in X."""
        means = np.mean([m.predict(X) for m in self.mean_boosters], axis=0)
        stds = np.mean([m.predict(X) for m in self.std_boosters], axis=0)
        # Floor std to a sane minimum (NFL teams almost never score within
        # tiny variance of their projection)
        stds = np.clip(stds, a_min=4.0, a_max=None)
        return means, stds


def _train_lgb(
    X: np.ndarray,
    y: np.ndarray,
    feature_cols: list[str],
    *,
    objective: str = "regression",
    eval_X: np.ndarray | None = None,
    eval_y: np.ndarray | None = None,
    seed: int = 0,
) -> lgb.Booster:
    """Train one LightGBM booster."""
    params = {
        "objective": objective,
        "learning_rate": 0.04,
        "num_leaves": 31,
        "min_data_in_leaf": 25,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.85,
        "bagging_freq": 5,
        "lambda_l2": 1.0,
        "verbose": -1,
        "seed": seed,
        "deterministic": True,
    }
    train_ds = lgb.Dataset(X, label=y, feature_name=feature_cols)
    if eval_X is not None and eval_y is not None and len(eval_X) > 0:
        valid_ds = lgb.Dataset(eval_X, label=eval_y, reference=train_ds)
        booster = lgb.train(
            params, train_ds, num_boost_round=2000,
            valid_sets=[valid_ds],
            callbacks=[
                lgb.early_stopping(stopping_rounds=80, verbose=False),
                lgb.log_evaluation(period=0),
            ],
        )
    else:
        booster = lgb.train(params, train_ds, num_boost_round=600)
    return booster


def train_score_model(
    X: np.ndarray,
    y: np.ndarray,
    feature_cols: list[str],
    *,
    eval_X: np.ndarray | None = None,
    eval_y: np.ndarray | None = None,
    n_seeds: int = 3,
) -> ScoresModel:
    """Train a multi-seed bagged mean + residual-std model on points."""
    mean_boosters = []
    for seed in range(n_seeds):
        b = _train_lgb(X, y, feature_cols, eval_X=eval_X, eval_y=eval_y, seed=seed)
        mean_boosters.append(b)

    # Residual std: train a regressor on |y - mean_pred|
    yhat = np.mean([b.predict(X) for b in mean_boosters], axis=0)
    resid = np.abs(y - yhat)
    std_boosters = []
    for seed in range(n_seeds):
        b = _train_lgb(
            X, resid, feature_cols, eval_X=eval_X,
            eval_y=np.abs(eval_y - yhat[: len(eval_y)]) if eval_X is not None and eval_y is not None else None,
            seed=seed + 100,
        )
        std_boosters.append(b)

    return ScoresModel(
        mean_boosters=mean_boosters,
        std_boosters=std_boosters,
        feature_cols=feature_cols,
    )


def _save_booster_list(boosters: list[lgb.Booster]) -> list[bytes]:
    return [b.model_to_string().encode("utf-8") for b in boosters]


def _load_booster_list(blobs: list[bytes]) -> list[lgb.Booster]:
    out = []
    for blob in blobs:
        out.append(lgb.Booster(model_str=blob.decode("utf-8")))
    return out


def save_score_model(model: ScoresModel, name: str) -> Path:
    """Persist a ScoresModel to ``models/<name>.joblib``."""
    settings.model_dir.mkdir(parents=True, exist_ok=True)
    path = settings.model_dir / f"{name}.joblib"
    payload = {
        "mean_blobs": _save_booster_list(model.mean_boosters),
        "std_blobs": _save_booster_list(model.std_boosters),
        "feature_cols": model.feature_cols,
    }
    joblib.dump(payload, path, compress=("zlib", 3))
    log.info("scores.saved", path=str(path), n_mean=len(model.mean_boosters))
    return path


def load_score_model(name: str) -> ScoresModel:
    path = settings.model_dir / f"{name}.joblib"
    payload = joblib.load(path)
    return ScoresModel(
        mean_boosters=_load_booster_list(payload["mean_blobs"]),
        std_boosters=_load_booster_list(payload["std_blobs"]),
        feature_cols=payload["feature_cols"],
    )


def train_score_models(features: pd.DataFrame) -> dict[str, Any]:
    """Train both home + away score-distribution models and persist them."""
    spec = fit_spec(features)
    feat_path = settings.model_dir / "feature_spec.joblib"
    settings.model_dir.mkdir(parents=True, exist_ok=True)

    sorted_feats = features.sort_values("game_date").reset_index(drop=True)
    val_cut = int(len(sorted_feats) * 0.85)
    train_part = sorted_feats.iloc[:val_cut]
    val_part = sorted_feats.iloc[val_cut:]

    Xh_t, _, feat_cols = build_runs_matrix(train_part, spec)
    Xa_t, _, _ = build_runs_matrix_away(train_part, spec)
    Xh_v, _, _ = build_runs_matrix(val_part, spec)
    Xa_v, _, _ = build_runs_matrix_away(val_part, spec)

    spec.final_feature_cols = feat_cols
    joblib.dump(spec, feat_path, compress=("zlib", 3))

    yh_t = train_part["target_home_score"].astype(float).to_numpy()
    ya_t = train_part["target_away_score"].astype(float).to_numpy()
    yh_v = val_part["target_home_score"].astype(float).to_numpy()
    ya_v = val_part["target_away_score"].astype(float).to_numpy()

    home_model = train_score_model(Xh_t, yh_t, feat_cols, eval_X=Xh_v, eval_y=yh_v)
    away_model = train_score_model(Xa_t, ya_t, feat_cols, eval_X=Xa_v, eval_y=ya_v)
    save_score_model(home_model, "scores_home")
    save_score_model(away_model, "scores_away")
    return {"n": len(features), "n_features": len(feat_cols)}
