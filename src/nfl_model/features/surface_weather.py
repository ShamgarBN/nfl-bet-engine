"""Surface (turf/grass), roof (dome/outdoor) and weather features.

The big one is **wind on pass axis**: speed projected onto the stadium's
long-axis bearing so the model knows how much the wind is helping or
hurting a deep ball -- not just absolute speed. Outdoor games with
||wind_on_pass_axis|| above 15 mph have measurably lower passing EPA
historically.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from nfl_model.config import settings
from nfl_model.data.warehouse import query
from nfl_model.logging import get_logger

log = get_logger("features.surface_weather")


def build(games: pd.DataFrame) -> pd.DataFrame:
    if games.empty:
        return pd.DataFrame()

    weather = query(
        """
        SELECT game_id, temp_f, humidity_pct, pressure_hpa,
               wind_speed_mph, wind_dir_deg, wind_on_pass_axis,
               precipitation_mm, is_dome
        FROM weather
        """
    )

    out = games[["game_id", "roof", "surface"]].copy()
    out = out.merge(weather, on="game_id", how="left")
    out["sw_is_dome"] = out["is_dome"].fillna(out["roof"].isin(["dome", "closed", "indoors"]))
    out["sw_is_turf"] = out["surface"].isin(["turf", "fieldturf", "a_turf"]).fillna(False)
    out["sw_temp_f"] = out["temp_f"]
    out["sw_humidity_pct"] = out["humidity_pct"]
    out["sw_wind_speed_mph"] = out["wind_speed_mph"].fillna(0.0)
    out["sw_wind_on_pass_axis"] = out["wind_on_pass_axis"].fillna(0.0)
    out["sw_wind_perpendicular"] = np.sqrt(
        (out["sw_wind_speed_mph"].fillna(0.0) ** 2
         - out["sw_wind_on_pass_axis"].fillna(0.0) ** 2).clip(lower=0)
    )
    out["sw_extreme_wind"] = (out["sw_wind_speed_mph"].abs() >= settings.extreme_wind_mph).fillna(False)
    out["sw_precipitation_mm"] = out["precipitation_mm"].fillna(0.0)
    # Targeted flags for sub-buckets that move totals more than the raw
    # values alone. Cold-weather passing is meaningfully worse below freezing;
    # heavy precip suppresses both scoring and kicking; "passing-unfriendly"
    # captures any of cold / wet / windy that the model could otherwise miss.
    out["sw_freezing"] = (out["sw_temp_f"].fillna(70.0) <= 32.0).astype(bool)
    out["sw_hot"] = (out["sw_temp_f"].fillna(70.0) >= 85.0).astype(bool)
    out["sw_heavy_precip"] = (out["sw_precipitation_mm"] >= 5.0).astype(bool)
    out["sw_passing_unfriendly"] = (
        out["sw_freezing"] | out["sw_heavy_precip"] | out["sw_extreme_wind"]
    )
    # Dome games are always passing-friendly; mask explicitly so the model
    # doesn't accidentally read stale rain/wind values for indoor venues.
    dome_mask = out["sw_is_dome"].astype(bool)
    for c in ("sw_extreme_wind", "sw_freezing", "sw_heavy_precip", "sw_passing_unfriendly"):
        out.loc[dome_mask, c] = False

    keep = [
        "game_id", "sw_is_dome", "sw_is_turf", "sw_temp_f", "sw_humidity_pct",
        "sw_wind_speed_mph", "sw_wind_on_pass_axis", "sw_wind_perpendicular",
        "sw_extreme_wind", "sw_precipitation_mm",
        "sw_freezing", "sw_hot", "sw_heavy_precip", "sw_passing_unfriendly",
    ]
    return out[keep]
