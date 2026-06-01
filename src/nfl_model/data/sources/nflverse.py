"""nflverse / nflreadpy ingestion.

One module, six pull functions: schedules, pbp, injuries, snap_counts,
officials, rosters. Each returns a pandas DataFrame shaped for direct
:func:`upsert_dataframe` writes against the warehouse tables.

We deliberately do NOT aggregate here — raw PBP is written to ``pbp`` and
the derived per-team / per-QB stats live in :mod:`.aggregations` so the
two concerns evolve independently.
"""

from __future__ import annotations

from typing import Iterable

import nflreadpy as nfl
import numpy as np
import pandas as pd

from nfl_model.data.warehouse import upsert_dataframe
from nfl_model.logging import get_logger

log = get_logger("data.sources.nflverse")


# Position string returned for "the head referee" within the officials feed.
# Other entries (Umpire, Down Judge, etc.) are not what the officiating
# feature builder wants.
_HEAD_REF_POS = "Referee"


def _to_pandas(df) -> pd.DataFrame:
    """Coerce a polars or pandas DataFrame to pandas."""
    if isinstance(df, pd.DataFrame):
        return df
    if hasattr(df, "to_pandas"):
        return df.to_pandas()
    raise TypeError(f"unsupported dataframe type: {type(df)!r}")


# --------------------------------------------------------------------------- #
# Schedules -> games                                                           #
# --------------------------------------------------------------------------- #


def pull_schedules(seasons: Iterable[int]) -> pd.DataFrame:
    """Load schedules for ``seasons`` and write to the ``games`` table.

    nflreadpy's schedules feed has every column the games table needs in
    a single fetch — scores, QBs, coaches, referee, stadium, surface,
    roof, kickoff date/time, plus the closing market lines (spread,
    total, moneylines) which we'll also pipe into ``odds_history`` from
    a separate ingestor.

    Returns the cleaned DataFrame for chaining (e.g. an odds ingest pass
    immediately afterwards).
    """
    seasons = list(seasons)
    raw = _to_pandas(nfl.load_schedules(seasons=seasons))
    if raw.empty:
        log.warning("schedules.empty", seasons=seasons)
        return raw

    out = pd.DataFrame()
    out["game_id"] = raw["game_id"].astype(str)
    out["season"] = pd.to_numeric(raw["season"], errors="coerce").astype("Int64")
    out["week"] = pd.to_numeric(raw["week"], errors="coerce").astype("Int64")
    out["game_type"] = raw["game_type"].astype(str)

    out["game_date"] = pd.to_datetime(raw["gameday"], errors="coerce").dt.date
    # Build a combined datetime when both date and gametime are present
    kickoff = pd.to_datetime(
        raw["gameday"].astype(str) + " " + raw["gametime"].fillna("").astype(str),
        errors="coerce",
    )
    out["kickoff_ts"] = kickoff

    out["weekday"] = raw["weekday"].astype(str) if "weekday" in raw.columns else None

    out["home_team"] = raw["home_team"].astype(str)
    out["away_team"] = raw["away_team"].astype(str)
    out["home_qb"] = raw.get("home_qb_name")
    out["away_qb"] = raw.get("away_qb_name")
    out["home_coach"] = raw.get("home_coach")
    out["away_coach"] = raw.get("away_coach")

    out["stadium_id"] = raw.get("stadium_id")
    out["stadium"] = raw.get("stadium")
    out["roof"] = raw.get("roof")
    out["surface"] = raw.get("surface")

    # Primetime: any non-Sunday-1pm slot. The schedule's gametime is local;
    # we proxy primetime as weekday in {Thu, Sun, Mon} *and* gametime in the
    # evening. Cheap heuristic; refined in features/schedule_context.
    times = raw["gametime"].astype(str).fillna("")
    hr = pd.to_numeric(times.str.slice(0, 2), errors="coerce")
    out["primetime"] = ((hr >= 19) | (hr == 0)).fillna(False)

    out["divisional"] = (
        raw["div_game"].astype(float).fillna(0).astype(bool)
        if "div_game" in raw.columns
        else False
    )

    out["referee"] = raw.get("referee")

    out["home_score"] = pd.to_numeric(raw["home_score"], errors="coerce").astype("Int64")
    out["away_score"] = pd.to_numeric(raw["away_score"], errors="coerce").astype("Int64")

    finalized = out["home_score"].notna() & out["away_score"].notna()
    home_win_vals = (out["home_score"].astype("Float64") > out["away_score"].astype("Float64"))
    out["home_win"] = home_win_vals.where(finalized, other=pd.NA).astype("boolean")
    out["overtime"] = (
        raw["overtime"].astype(float).fillna(0).astype(bool)
        if "overtime" in raw.columns
        else False
    )

    upsert_dataframe(out, "games", key_columns=["game_id"])
    return out


# --------------------------------------------------------------------------- #
# Schedules -> odds_history (consensus pre-baked from the schedule feed)       #
# --------------------------------------------------------------------------- #


def pull_odds_from_schedules(seasons: Iterable[int]) -> pd.DataFrame:
    """Derive a ``book='consensus'`` row per game from the schedule odds cols.

    nflreadpy's schedule feed ships closing-line snapshots for spread,
    total, moneyline, plus the spread / total American prices. This gives
    us a free baseline odds_history row per game. Real per-book history
    can be layered in later from SBR archives.
    """
    seasons = list(seasons)
    raw = _to_pandas(nfl.load_schedules(seasons=seasons))
    if raw.empty:
        return raw

    out = pd.DataFrame()
    out["game_id"] = raw["game_id"].astype(str)
    out["book"] = "consensus"

    # nflverse encodes ``spread_line`` as "points the home team is favored by"
    # (positive when home is the favorite). The rest of the model — market.py,
    # the simulator, walkforward.py — uses the standard sportsbook convention
    # where a negative spread means the home team is laying points (e.g.
    # KC -3.5). We negate at ingestion so downstream callers can rely on the
    # familiar convention without remembering this gotcha.
    home_favored_by = pd.to_numeric(raw.get("spread_line"), errors="coerce")
    out["spread_open"] = -home_favored_by
    out["spread_close"] = -home_favored_by
    out["spread_close_home_price"] = pd.to_numeric(
        raw.get("home_spread_odds"), errors="coerce"
    ).astype("Int64")
    out["spread_close_away_price"] = pd.to_numeric(
        raw.get("away_spread_odds"), errors="coerce"
    ).astype("Int64")

    out["ml_open_home"] = pd.to_numeric(raw.get("home_moneyline"), errors="coerce").astype("Int64")
    out["ml_open_away"] = pd.to_numeric(raw.get("away_moneyline"), errors="coerce").astype("Int64")
    out["ml_close_home"] = out["ml_open_home"]
    out["ml_close_away"] = out["ml_open_away"]

    out["total_open"] = pd.to_numeric(raw.get("total_line"), errors="coerce")
    out["total_close"] = out["total_open"]
    out["total_close_over"] = pd.to_numeric(raw.get("over_odds"), errors="coerce").astype("Int64")
    out["total_close_under"] = pd.to_numeric(raw.get("under_odds"), errors="coerce").astype("Int64")

    out["home_bets_pct"] = np.nan
    out["home_handle_pct"] = np.nan
    out["reverse_line_move"] = False
    out["steam_move"] = False

    out = out.dropna(subset=["game_id"])
    upsert_dataframe(out, "odds_history", key_columns=["game_id", "book"])
    return out


# --------------------------------------------------------------------------- #
# Play-by-play -> pbp (thin subset)                                            #
# --------------------------------------------------------------------------- #


def pull_pbp(seasons: Iterable[int]) -> pd.DataFrame:
    """Pull PBP for ``seasons`` and write the columns we aggregate over.

    The full nflverse PBP has 372+ columns; we persist a 23-column subset
    that's sufficient to rebuild every Stage-A feature. Keeping pbp around
    means the ``ablate`` command can re-derive team_game_stats on the fly
    without re-pulling from the network.
    """
    seasons = list(seasons)
    pbp = _to_pandas(nfl.load_pbp(seasons=seasons))
    if pbp.empty:
        log.warning("pbp.empty", seasons=seasons)
        return pbp

    # 2pt conversions and timeouts have epa but distort per-play aggregates;
    # the cleaner set is "actual scrimmage plays".
    keep_play = pbp["play_type"].isin({"pass", "run", "qb_kneel", "qb_spike", "field_goal"})
    pbp = pbp[keep_play].copy()

    explosive = ((pbp["yards_gained"].fillna(0) >= 20) & pbp["play_type"].eq("pass")) | (
        (pbp["yards_gained"].fillna(0) >= 10) & pbp["play_type"].eq("run")
    )

    out = pd.DataFrame()
    out["season"] = pd.to_numeric(pbp["season"] if "season" in pbp.columns else pbp.get("season_year"), errors="coerce").astype("Int64")
    out["week"] = pd.to_numeric(pbp["week"], errors="coerce").astype("Int64")
    out["game_id"] = pbp["game_id"].astype(str)
    out["play_id"] = pd.to_numeric(pbp["play_id"], errors="coerce").astype("Int64")
    out["posteam"] = pbp.get("posteam")
    out["defteam"] = pbp.get("defteam")
    out["down"] = pd.to_numeric(pbp.get("down"), errors="coerce").astype("Int64")
    out["ydstogo"] = pd.to_numeric(pbp.get("ydstogo"), errors="coerce").astype("Int64")
    out["yardline_100"] = pd.to_numeric(pbp.get("yardline_100"), errors="coerce").astype("Int64")
    out["play_type"] = pbp.get("play_type")
    out["passer_id"] = pbp.get("passer_player_id")
    out["rusher_id"] = pbp.get("rusher_player_id")
    out["receiver_id"] = pbp.get("receiver_player_id")
    out["epa"] = pd.to_numeric(pbp.get("epa"), errors="coerce")
    out["cpoe"] = pd.to_numeric(pbp.get("cpoe"), errors="coerce")
    out["success"] = pbp.get("success").astype(float).fillna(0).astype(bool) if "success" in pbp.columns else False
    out["explosive"] = explosive.fillna(False).astype(bool)
    out["air_yards"] = pd.to_numeric(pbp.get("air_yards"), errors="coerce")
    out["yards_gained"] = pd.to_numeric(pbp.get("yards_gained"), errors="coerce").astype("Int64")
    out["sack"] = pbp.get("sack").astype(float).fillna(0).astype(bool) if "sack" in pbp.columns else False
    out["touchdown"] = pbp.get("touchdown").astype(float).fillna(0).astype(bool) if "touchdown" in pbp.columns else False
    out["field_goal"] = pbp["play_type"].eq("field_goal")
    out["fg_distance"] = pd.to_numeric(pbp.get("kick_distance"), errors="coerce").astype("Int64")
    out["fg_made"] = (pbp.get("field_goal_result") == "made") if "field_goal_result" in pbp.columns else False
    out["is_pass"] = pbp.get("pass_attempt").astype(float).fillna(0).astype(bool) if "pass_attempt" in pbp.columns else pbp["play_type"].eq("pass")
    out["is_rush"] = pbp.get("rush_attempt").astype(float).fillna(0).astype(bool) if "rush_attempt" in pbp.columns else pbp["play_type"].eq("run")
    out["wp"] = pd.to_numeric(pbp.get("wp"), errors="coerce")

    out = out.dropna(subset=["game_id", "play_id"])
    upsert_dataframe(out, "pbp", key_columns=["game_id", "play_id"])
    return out


# --------------------------------------------------------------------------- #
# Injuries                                                                     #
# --------------------------------------------------------------------------- #


def pull_injuries(seasons: Iterable[int]) -> pd.DataFrame:
    """Weekly injury report by player. Maps to ``injuries`` table."""
    seasons = list(seasons)
    raw = _to_pandas(nfl.load_injuries(seasons=seasons))
    if raw.empty:
        return raw

    out = pd.DataFrame()
    out["season"] = pd.to_numeric(raw["season"], errors="coerce").astype("Int64")
    out["week"] = pd.to_numeric(raw["week"], errors="coerce").astype("Int64")
    out["team"] = raw["team"].astype(str)
    out["player_id"] = raw["gsis_id"].astype(str)
    out["player_name"] = raw.get("full_name")
    out["position"] = raw.get("position")
    out["game_status"] = raw.get("report_status")
    out["practice_status"] = raw.get("practice_status")
    out["report_date"] = pd.to_datetime(raw.get("date_modified"), errors="coerce").dt.date

    out = out.dropna(subset=["season", "week", "team", "player_id"])
    out = out.drop_duplicates(subset=["season", "week", "team", "player_id"], keep="last")
    upsert_dataframe(out, "injuries", key_columns=["season", "week", "team", "player_id"])
    return out


# --------------------------------------------------------------------------- #
# Snap counts                                                                  #
# --------------------------------------------------------------------------- #


def pull_snap_counts(seasons: Iterable[int]) -> pd.DataFrame:
    """Per-player snap counts for each game. Maps to ``snap_counts``."""
    seasons = list(seasons)
    raw = _to_pandas(nfl.load_snap_counts(seasons=seasons))
    if raw.empty:
        return raw

    out = pd.DataFrame()
    out["game_id"] = raw["game_id"].astype(str)
    out["player_id"] = raw["pfr_player_id"].astype(str)
    out["player_name"] = raw.get("player")
    out["team"] = raw.get("team")
    out["position"] = raw.get("position")
    out["offense_pct"] = pd.to_numeric(raw.get("offense_pct"), errors="coerce")
    out["defense_pct"] = pd.to_numeric(raw.get("defense_pct"), errors="coerce")
    out["st_pct"] = pd.to_numeric(raw.get("st_pct"), errors="coerce")

    out = out.dropna(subset=["game_id", "player_id"])
    out = out.drop_duplicates(subset=["game_id", "player_id"], keep="last")
    upsert_dataframe(out, "snap_counts", key_columns=["game_id", "player_id"])
    return out


# --------------------------------------------------------------------------- #
# Officials -> officiating (one row per game, head ref only)                   #
# --------------------------------------------------------------------------- #


def pull_officials(seasons: Iterable[int]) -> pd.DataFrame:
    """Pull the head referee per game; one row per game in ``officiating``.

    The officials feed keys games by the legacy nfl.com game_id format
    (e.g. ``2024090500``); the rest of the warehouse uses the modern
    nflverse format (``2024_01_BAL_KC``). We rejoin via the schedule's
    ``old_game_id`` column so the resulting officiating rows share PKs
    with ``games``.
    """
    seasons = list(seasons)
    raw = _to_pandas(nfl.load_officials(seasons=seasons))
    if raw.empty:
        return raw

    refs = raw[raw["position"] == _HEAD_REF_POS].copy()
    if refs.empty:
        return refs

    # Build a lookup: legacy_id -> modern game_id from the schedules feed.
    sched = _to_pandas(nfl.load_schedules(seasons=seasons))
    if not sched.empty and "old_game_id" in sched.columns:
        id_map = dict(
            zip(
                sched["old_game_id"].astype(str),
                sched["game_id"].astype(str),
                strict=False,
            )
        )
        refs["game_id"] = refs["game_id"].astype(str).map(id_map).fillna(
            refs["game_id"].astype(str)
        )

    out = pd.DataFrame()
    out["game_id"] = refs["game_id"].astype(str)
    out["referee"] = refs["official_name"].astype(str)
    out["crew_id"] = refs.get("official_id").astype(str)
    # penalties_total / penalty_yards / total_points are aggregations of
    # other tables and get filled in by aggregations.refresh_officiating_counts.
    out["penalties_total"] = pd.NA
    out["penalty_yards"] = pd.NA
    out["total_points"] = pd.NA
    out = out.drop_duplicates(subset=["game_id"], keep="last")
    upsert_dataframe(out, "officiating", key_columns=["game_id"])
    return out
