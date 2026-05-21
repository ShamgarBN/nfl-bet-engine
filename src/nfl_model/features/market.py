"""Market features: closing line, line movement, de-vigged implied probabilities,
reverse-line-movement, key-number-cross flags, steam-move detection.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from nfl_model.config import settings
from nfl_model.data.warehouse import query
from nfl_model.logging import get_logger

log = get_logger("features.market")


def _american_to_implied(odds: pd.Series) -> pd.Series:
    """Convert American odds to implied probability (no de-vig)."""
    odds = pd.to_numeric(odds, errors="coerce")
    pos = odds > 0
    out = pd.Series(np.nan, index=odds.index, dtype=float)
    out[pos] = 100.0 / (odds[pos] + 100.0)
    out[~pos] = (-odds[~pos]) / ((-odds[~pos]) + 100.0)
    return out


def _devig_two_way(p_a: pd.Series, p_b: pd.Series) -> pd.Series:
    """De-vig two-way market by normalizing implied probabilities."""
    s = p_a + p_b
    return p_a / s.replace(0, np.nan)


def _key_cross_flag(open_v: pd.Series, close_v: pd.Series, keys: tuple[int, ...]) -> pd.Series:
    """Flag any line movement that crossed one of the key numbers."""
    out = pd.Series(False, index=open_v.index)
    for k in keys:
        out |= ((open_v < -k) & (close_v > -k)) | ((open_v > -k) & (close_v < -k))
        out |= ((open_v < k) & (close_v > k)) | ((open_v > k) & (close_v < k))
    return out


def build(games: pd.DataFrame) -> pd.DataFrame:
    if games.empty:
        return pd.DataFrame()

    odds = query(
        """
        SELECT game_id, spread_open, spread_close,
               spread_close_home_price, spread_close_away_price,
               ml_open_home, ml_open_away, ml_close_home, ml_close_away,
               total_open, total_close, total_close_over, total_close_under,
               home_bets_pct, home_handle_pct, reverse_line_move, steam_move
        FROM odds_history
        WHERE book = 'consensus'
        """
    )
    if odds.empty:
        return pd.DataFrame({"game_id": games["game_id"]})

    out = odds.copy()
    out["mkt_spread_open"] = pd.to_numeric(out["spread_open"], errors="coerce")
    out["mkt_spread_close"] = pd.to_numeric(out["spread_close"], errors="coerce")
    out["mkt_spread_move"] = out["mkt_spread_close"] - out["mkt_spread_open"]

    p_home = _american_to_implied(out["ml_close_home"])
    p_away = _american_to_implied(out["ml_close_away"])
    out["mkt_ml_implied_home"] = _devig_two_way(p_home, p_away)
    out["mkt_ml_implied_away"] = 1.0 - out["mkt_ml_implied_home"]

    out["mkt_total_open"] = pd.to_numeric(out["total_open"], errors="coerce")
    out["mkt_total_close"] = pd.to_numeric(out["total_close"], errors="coerce")
    out["mkt_total_move"] = out["mkt_total_close"] - out["mkt_total_open"]

    out["mkt_key_cross_spread"] = _key_cross_flag(
        out["mkt_spread_open"], out["mkt_spread_close"], settings.key_numbers_spread
    )
    out["mkt_reverse_line_move"] = out["reverse_line_move"].fillna(False)
    out["mkt_steam_move"] = out["steam_move"].fillna(False)

    keep = [
        "game_id",
        "mkt_spread_open", "mkt_spread_close", "mkt_spread_move",
        "mkt_ml_implied_home", "mkt_ml_implied_away",
        "mkt_total_open", "mkt_total_close", "mkt_total_move",
        "mkt_key_cross_spread", "mkt_reverse_line_move", "mkt_steam_move",
    ]
    return out[keep]
