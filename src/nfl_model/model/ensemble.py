"""Simulator + direct-classifier ensemble.

Final probability for each market is a weighted blend:

    p_final = w * p_simulator + (1 - w) * p_direct_classifier

where ``w`` is per-market and walk-forward tuned to minimize log-loss on
held-out games. Defaults bias toward the simulator (which has stronger
priors) but the tuner can override per market.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from nfl_model.config import settings
from nfl_model.logging import get_logger

log = get_logger("model.ensemble")

DEFAULT_WEIGHTS: dict[str, float] = {
    "spread": 0.55,
    "ml": 0.65,    # ML probabilities are dominated by the simulator's joint
    "total": 0.50,
}


def _weights_path() -> Path:
    return settings.model_dir / "ensemble_weights.json"


def load_weights() -> dict[str, float]:
    p = _weights_path()
    if not p.exists():
        return dict(DEFAULT_WEIGHTS)
    try:
        return {**DEFAULT_WEIGHTS, **json.loads(p.read_text())}
    except Exception:  # noqa: BLE001
        return dict(DEFAULT_WEIGHTS)


def save_weights(weights: dict[str, float]) -> Path:
    settings.model_dir.mkdir(parents=True, exist_ok=True)
    p = _weights_path()
    p.write_text(json.dumps(weights, indent=2))
    log.info("ensemble.weights.saved", weights=weights)
    return p


def blend(market: str, p_sim: np.ndarray, p_direct: np.ndarray | None) -> np.ndarray:
    """Blend simulator + direct probabilities for ``market``."""
    if p_direct is None:
        return np.asarray(p_sim, dtype=float)
    weights = load_weights()
    w = float(weights.get(market, 0.5))
    return w * np.asarray(p_sim, dtype=float) + (1.0 - w) * np.asarray(p_direct, dtype=float)
