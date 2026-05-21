"""Isotonic per-market calibrators.

Trained on the simulator's OOF distribution -- the same distribution that
the inference path produces -- so training and serving stay aligned.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.isotonic import IsotonicRegression

from nfl_model.config import settings
from nfl_model.logging import get_logger

log = get_logger("model.calibrate")


@dataclass
class Calibrator:
    market: str
    isotonic: IsotonicRegression

    def transform(self, probs: np.ndarray) -> np.ndarray:
        clipped = np.clip(probs, 1e-6, 1 - 1e-6)
        return self.isotonic.predict(clipped)


def fit_calibrator(market: str, probs: np.ndarray, outcomes: np.ndarray) -> Calibrator:
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(probs, outcomes)
    return Calibrator(market=market, isotonic=iso)


def save_calibrator(cal: Calibrator) -> Path:
    settings.model_dir.mkdir(parents=True, exist_ok=True)
    path = settings.model_dir / f"calibrator_{cal.market}.joblib"
    joblib.dump(cal, path, compress=("zlib", 3))
    log.info("calibrator.saved", market=cal.market, path=str(path))
    return path


def load_calibrator(market: str) -> Calibrator | None:
    path = settings.model_dir / f"calibrator_{market}.joblib"
    if not path.exists():
        return None
    return joblib.load(path)


def fit_all_calibrators(features: Any) -> dict[str, Any]:
    """Stub: real fit happens in walk-forward training when OOF probs exist."""
    log.info("calibrate.fit_all.skipped", reason="walk-forward path produces OOF; train via backtest")
    return {"fit_at_walkforward": True}
