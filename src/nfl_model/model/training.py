"""Production training entry point.

Coordinates Stage A (team score distributions), direct ATS / total
classifiers, isotonic calibrators, and ensemble-blend tuning. Persists
artifacts under ``models/``.
"""

from __future__ import annotations

from typing import Any

from nfl_model.logging import get_logger

log = get_logger("model.training")


def train_production_models(*, train_start: int, through_season: int) -> dict[str, Any]:
    """Train the full production stack on [train_start, through_season]."""
    from nfl_model.features.assemble import build_features_table
    from nfl_model.model.calibrate import fit_all_calibrators
    from nfl_model.model.scores import train_score_models
    from nfl_model.model.spread_direct import train_spread_classifier
    from nfl_model.model.totals import train_totals_classifier

    features = build_features_table(train_start, through_season)
    if features.empty:
        return {"trained": False, "reason": "no features"}

    finalized = features.dropna(subset=["target_home_score", "target_away_score"])
    if finalized.empty:
        return {"trained": False, "reason": "no finalized scores"}

    info: dict[str, Any] = {"n_train": int(len(finalized))}
    info["scores"] = train_score_models(finalized)
    info["spread"] = train_spread_classifier(finalized)
    info["totals"] = train_totals_classifier(finalized)
    info["calibrators"] = fit_all_calibrators(finalized)
    info["trained"] = True
    return info
