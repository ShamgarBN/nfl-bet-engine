"""Derive per-team and per-QB game stats from raw play-by-play.

Run after :mod:`.nflverse.pull_pbp` writes plays to the ``pbp`` table.
Produces the ``team_game_stats`` and ``qb_game_stats`` rows that every
EPA / pace / success-rate / QB feature in :mod:`nfl_model.features`
reads from.

Defensive EPA is the *negative* of the opponent's offensive EPA on the
same play — a positive defense_epa_per_play means the defense is
allowing more (worse) than league average. We invert sign so that
"higher = better defense" the way coaches and feature consumers expect.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from nfl_model.data.warehouse import query, upsert_dataframe
from nfl_model.logging import get_logger

log = get_logger("data.sources.aggregations")


_NEUTRAL_WP_LO = 0.20
_NEUTRAL_WP_HI = 0.80


def _safe_div(num: pd.Series, den: pd.Series) -> pd.Series:
    return np.where(den > 0, num / den.replace(0, np.nan), np.nan)


def refresh_team_game_stats(season: int | None = None) -> int:
    """Aggregate per-team per-game stats from the ``pbp`` table.

    Writes rows to ``team_game_stats`` keyed (game_id, team). Returns the
    number of (game, team) rows written. Idempotent — re-running with the
    same season replaces any prior aggregations.
    """
    where = ""
    params: tuple = ()
    if season is not None:
        where = "WHERE season = ?"
        params = (int(season),)
    plays = query(
        f"""
        SELECT game_id, posteam, defteam, season, week, down, ydstogo, yardline_100,
               play_type, epa, success, explosive, is_pass, is_rush, sack, wp,
               yards_gained
        FROM pbp {where}
        """,
        params,
    )
    if plays.empty:
        log.warning("aggregations.team.no_pbp", season=season)
        return 0

    plays = plays[plays["posteam"].notna() & plays["defteam"].notna()].copy()

    # ---- Offense: one row per (game_id, posteam) ----
    off = plays.groupby(["game_id", "posteam"], dropna=True).agg(
        offense_epa_per_play=("epa", "mean"),
        success_rate=("success", "mean"),
        explosive_rate=("explosive", "mean"),
        pass_epa=("epa", lambda s: s[plays.loc[s.index, "is_pass"]].mean()),
        rush_epa=("epa", lambda s: s[plays.loc[s.index, "is_rush"]].mean()),
    ).reset_index().rename(columns={"posteam": "team"})

    # Pace: plays_per_drive needs raw drive counts per game (so we group by game)
    # We don't have drive_id here; approximate with a flat plays/game / 12-drive
    # baseline, replaced when the assemble.py builder needs higher precision.
    plays_per_game = (
        plays.groupby(["game_id", "posteam"]).size().rename("plays").reset_index()
        .rename(columns={"posteam": "team"})
    )
    off = off.merge(plays_per_game, on=["game_id", "team"], how="left")
    off["plays_per_drive"] = off["plays"] / 12.0
    off = off.drop(columns=["plays"])

    # Neutral pass rate + PROE (pass-rate over expected). Neutral = .20 < wp < .80
    neutral_mask = plays["wp"].between(_NEUTRAL_WP_LO, _NEUTRAL_WP_HI, inclusive="both")
    neutral = plays[neutral_mask & (plays["play_type"].isin(["pass", "run"]))]
    if not neutral.empty:
        pace = neutral.groupby(["game_id", "posteam"]).agg(
            neutral_pass_rate=("is_pass", "mean"),
        ).reset_index().rename(columns={"posteam": "team"})
        off = off.merge(pace, on=["game_id", "team"], how="left")
    else:
        off["neutral_pass_rate"] = np.nan

    # PROE: pass rate minus league mean pass rate at the same down-distance bucket.
    # Cheap version: subtract league mean per-week.
    if "neutral_pass_rate" in off.columns:
        league_mean = off.groupby("game_id")["neutral_pass_rate"].transform("mean")
        off["proe"] = off["neutral_pass_rate"] - league_mean
    else:
        off["proe"] = np.nan

    # 3rd-down conversion %: from third_down columns we recreated via plays
    third_downs = plays[plays["down"] == 3].copy()
    if not third_downs.empty:
        # success on 3rd down = gained ydstogo
        third_downs["converted"] = third_downs["yards_gained"].fillna(0) >= third_downs["ydstogo"].fillna(99)
        td_rate = third_downs.groupby(["game_id", "posteam"]).agg(
            third_down_pct=("converted", "mean"),
        ).reset_index().rename(columns={"posteam": "team"})
        off = off.merge(td_rate, on=["game_id", "team"], how="left")
    else:
        off["third_down_pct"] = np.nan

    # Red-zone TD %: TDs on plays inside the 20
    red_zone = plays[plays["yardline_100"].between(0, 20)].copy()
    if not red_zone.empty:
        red_zone["rz_td"] = red_zone["play_type"].isin(["pass", "run"]) & (
            red_zone["yards_gained"].fillna(0) >= red_zone["yardline_100"]
        )
        rz_rate = red_zone.groupby(["game_id", "posteam"]).agg(
            red_zone_td_pct=("rz_td", "mean"),
        ).reset_index().rename(columns={"posteam": "team"})
        off = off.merge(rz_rate, on=["game_id", "team"], how="left")
    else:
        off["red_zone_td_pct"] = np.nan

    # ---- Defense: aggregate over plays where this team is the defteam ----
    # Defensive EPA is the negative of the offense's EPA against this defense.
    defense = plays.groupby(["game_id", "defteam"], dropna=True).agg(
        defense_epa_per_play=("epa", lambda s: -s.mean()),
    ).reset_index().rename(columns={"defteam": "team"})

    out = off.merge(defense, on=["game_id", "team"], how="outer")

    # ---- Mark is_home by joining to games ----
    games = query("SELECT game_id, home_team FROM games")
    if not games.empty:
        out = out.merge(games, on="game_id", how="left")
        out["is_home"] = out["team"] == out["home_team"]
        out = out.drop(columns=["home_team"])
    else:
        out["is_home"] = False

    upsert_dataframe(out, "team_game_stats", key_columns=["game_id", "team"])
    log.info("aggregations.team.refreshed", season=season, rows=len(out))
    return int(len(out))


def refresh_qb_game_stats(season: int | None = None) -> int:
    """Aggregate per-QB stats from PBP. One row per (game_id, passer_id).

    ``is_starter`` is the QB with the most pass attempts for their team
    that game. (Backup QBs get is_starter=False even if they took meaningful
    snaps after a starter injury — the qb_form feature later weights by
    attempts so this is OK.)
    """
    where = ""
    params: tuple = ()
    if season is not None:
        where = "WHERE season = ?"
        params = (int(season),)

    plays = query(
        f"""
        SELECT game_id, posteam, passer_id, is_pass, epa, cpoe, sack
        FROM pbp {where}
        """,
        params,
    )
    if plays.empty:
        log.warning("aggregations.qb.no_pbp", season=season)
        return 0

    passes = plays[plays["is_pass"] & plays["passer_id"].notna()].copy()
    if passes.empty:
        return 0

    agg = passes.groupby(["game_id", "passer_id"]).agg(
        team=("posteam", "first"),
        attempts=("is_pass", "sum"),
        epa_per_play=("epa", "mean"),
        cpoe=("cpoe", "mean"),
        sacks=("sack", "sum"),
    ).reset_index().rename(columns={"passer_id": "player_id"})

    agg["sack_rate"] = agg["sacks"] / agg["attempts"].replace(0, np.nan)
    agg = agg.drop(columns=["sacks"])

    # Mark starter as the QB with the most attempts per (game, team).
    agg["is_starter"] = False
    idx_starter = (
        agg.sort_values("attempts", ascending=False)
        .drop_duplicates(["game_id", "team"], keep="first")
        .index
    )
    agg.loc[idx_starter, "is_starter"] = True

    # Player name: nflreadpy ships it in PBP as passer_player_name; we can
    # join from rosters or skip it. Easiest: pull from passes' first occurrence.
    # We didn't store passer_player_name in pbp; for now leave NULL — the UI
    # falls back to the schedule's home_qb / away_qb fields.
    agg["player_name"] = pd.NA

    upsert_dataframe(
        agg[
            [
                "game_id",
                "player_id",
                "player_name",
                "team",
                "attempts",
                "epa_per_play",
                "cpoe",
                "sack_rate",
                "is_starter",
            ]
        ],
        "qb_game_stats",
        key_columns=["game_id", "player_id"],
    )
    log.info("aggregations.qb.refreshed", season=season, rows=len(agg))
    return int(len(agg))


def refresh_officiating_counts(season: int | None = None) -> int:
    """Backfill penalties_total / penalty_yards / total_points on officiating.

    Penalty data isn't currently persisted to ``pbp`` (we stripped most
    columns). ``total_points`` is trivial from the ``games`` table. This is
    a small backfill so feature builders that read these columns don't get
    all-NaN — penalty richness is queued for a future PFR scrape.
    """
    where = ""
    params: tuple = ()
    if season is not None:
        where = "WHERE g.season = ?"
        params = (int(season),)
    df = query(
        f"""
        SELECT g.game_id,
               COALESCE(g.home_score, 0) + COALESCE(g.away_score, 0) AS total_points
        FROM games g
        WHERE g.home_score IS NOT NULL AND g.away_score IS NOT NULL
        {('AND' if where else '') + where.replace('WHERE','')}
        """,
        params,
    )
    if df.empty:
        return 0

    # Upsert just total_points + leave penalty fields untouched (NULL).
    df["penalties_total"] = pd.NA
    df["penalty_yards"] = pd.NA
    # referee + crew_id are already populated by pull_officials; leave them.
    upsert_dataframe(
        df[["game_id", "total_points"]],
        "officiating",
        key_columns=["game_id"],
    )
    return int(len(df))
