"""Shared helpers for leakage-safe rolling features.

All rolling feature builders use these. The fundamental rule: features for
a game played at time T must use only data available *before* T. We
enforce this by sorting by date and using ``shift()`` before any rolling
aggregation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def leakage_safe_rolling(
    df: pd.DataFrame,
    *,
    group_col: str,
    sort_col: str,
    value_cols: list[str],
    window: int,
    min_periods: int = 1,
    weight: str | None = None,
) -> pd.DataFrame:
    """Compute a leakage-safe rolling mean for ``value_cols`` per ``group_col``.

    Args:
        df: Dataframe with one row per (group, time).
        group_col: e.g. 'team' or 'player_id'.
        sort_col: e.g. 'game_date'.
        value_cols: columns to roll.
        window: rolling window in rows.
        min_periods: minimum rows to compute a rolling stat.
        weight: 'recency' to weight more recent rows higher (1, 2, ..., n);
                None for unweighted mean.

    Returns:
        Dataframe with the same index as ``df`` and one column per
        ``value_col`` containing the rolling-mean *prior* to the row's
        own observation.
    """
    out = pd.DataFrame(index=df.index)
    df = df.sort_values([group_col, sort_col])

    for col in value_cols:
        shifted = df.groupby(group_col)[col].shift(1)
        if weight == "recency":
            def _weighted_mean(x: pd.Series) -> float:
                vals = x.dropna()
                if vals.empty:
                    return float("nan")
                w = np.arange(1, len(vals) + 1, dtype=float)
                return float(np.sum(vals.to_numpy() * w) / w.sum())

            rolled = shifted.groupby(df[group_col]).rolling(
                window, min_periods=min_periods
            ).apply(_weighted_mean, raw=False).reset_index(level=0, drop=True)
        else:
            rolled = shifted.groupby(df[group_col]).rolling(
                window, min_periods=min_periods
            ).mean().reset_index(level=0, drop=True)
        out[f"{col}_roll{window}"] = rolled

    return out
