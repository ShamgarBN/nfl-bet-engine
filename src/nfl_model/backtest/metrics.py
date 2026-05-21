"""Per-pick metrics: ATS%, ROI, Brier, log-loss, CLV, teaser-leg %."""

from __future__ import annotations

import numpy as np
import pandas as pd


def brier_score(probs: np.ndarray, outcomes: np.ndarray) -> float:
    p = np.asarray(probs, dtype=float)
    y = np.asarray(outcomes, dtype=float)
    mask = np.isfinite(p) & np.isfinite(y)
    if mask.sum() == 0:
        return float("nan")
    return float(((p[mask] - y[mask]) ** 2).mean())


def log_loss(probs: np.ndarray, outcomes: np.ndarray, eps: float = 1e-6) -> float:
    p = np.clip(np.asarray(probs, dtype=float), eps, 1 - eps)
    y = np.asarray(outcomes, dtype=float)
    mask = np.isfinite(p) & np.isfinite(y)
    if mask.sum() == 0:
        return float("nan")
    return -float(np.mean(y[mask] * np.log(p[mask]) + (1 - y[mask]) * np.log(1 - p[mask])))


def ats_accuracy(picks: pd.Series, outcomes: pd.Series) -> float:
    valid = picks.notna() & outcomes.notna() & (outcomes != "push")
    if valid.sum() == 0:
        return float("nan")
    return float((picks[valid] == outcomes[valid]).mean())


def ml_roi_units(correct: pd.Series, decimal_odds: pd.Series) -> float:
    """ROI in units (1u risked per pick)."""
    units = np.where(correct, decimal_odds - 1.0, -1.0)
    units = pd.Series(units).where(decimal_odds.notna(), other=0.0)
    return float(units.mean()) if len(units) else float("nan")
