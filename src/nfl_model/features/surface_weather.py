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
    out["sw_extreme_wind"] = (out["sw_wind_speed_mph"].abs() >= settings.extreme_wind_mph).fillna(False)
    out["sw_precipitation_mm"] = out["precipitation_mm"].fillna(0.0)

    keep = [
        "game_id", "sw_is_dome", "sw_is_turf", "sw_temp_f", "sw_humidity_pct",
        "sw_wind_speed_mph", "sw_wind_on_pass_axis", "sw_extreme_wind",
        "sw_precipitation_mm",
    ]
    return out[keep]
