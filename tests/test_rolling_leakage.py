"""Leakage test: rolling features must NOT use the current row's own value.

The whole model rests on this invariant. If we leak the current game's
EPA into the feature row for that game, the model "predicts" it via a
shortcut and the published accuracy is meaningless.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from nfl_model.features._rolling import leakage_safe_rolling


def test_leakage_safe_rolling_does_not_use_current_row():
    """First row must be NaN (no prior data); subsequent rows must
    only use values from earlier rows.
    """
    df = pd.DataFrame(
        {
            "team": ["A", "A", "A", "A", "B", "B"],
            "date": pd.to_datetime(
                ["2024-09-01", "2024-09-08", "2024-09-15", "2024-09-22",
                 "2024-09-01", "2024-09-08"]
            ),
            "epa": [0.10, 0.20, -0.05, 0.30, 0.00, 0.40],
        }
    )

    result = leakage_safe_rolling(
        df,
        group_col="team",
        sort_col="date",
        value_cols=["epa"],
        window=2,
        min_periods=1,
    )

    # First A and first B must be NaN
    assert pd.isna(result.iloc[0]["epa_roll2"])
    assert pd.isna(result.iloc[4]["epa_roll2"])

    # Second A row uses only first A's value (0.10)
    assert result.iloc[1]["epa_roll2"] == 0.10

    # Third A uses first + second (mean of 0.10 and 0.20)
    np.testing.assert_almost_equal(result.iloc[2]["epa_roll2"], 0.15)

    # Fourth A uses second + third (mean of 0.20 and -0.05)
    np.testing.assert_almost_equal(result.iloc[3]["epa_roll2"], 0.075)


def test_leakage_safe_rolling_recency_weighted():
    """Recency weighting gives later observations more weight."""
    df = pd.DataFrame(
        {
            "team": ["A", "A", "A"],
            "date": pd.to_datetime(["2024-09-01", "2024-09-08", "2024-09-15"]),
            "epa": [0.0, 0.0, 1.0],  # 0, 0, 1 prior to t=4
        }
    )
    df = pd.concat(
        [df, pd.DataFrame(
            {"team": ["A"], "date": [pd.Timestamp("2024-09-22")], "epa": [0.0]}
        )],
        ignore_index=True,
    )
    result = leakage_safe_rolling(
        df, group_col="team", sort_col="date",
        value_cols=["epa"], window=3, min_periods=1, weight="recency",
    )
    # The 4th row uses prior 3 (0, 0, 1.0). With recency weights (1, 2, 3)
    # the weighted mean = (1*0 + 2*0 + 3*1) / 6 = 0.5
    np.testing.assert_almost_equal(result.iloc[3]["epa_roll3"], 0.5)
