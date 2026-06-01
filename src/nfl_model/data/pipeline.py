"""Top-level ingestion orchestrator.

Two public entry points used by the CLI and the automation jobs:

- :func:`ensure_venues` — idempotent seed of the static stadium table.
- :func:`pull_season(year, with_weather, with_lines)` — fetch every
  upstream source for a single season into the warehouse, in the right
  order so foreign-key joins (officiating → games, weather → venues,
  team_game_stats → pbp) succeed.

Each phase swallows its own errors and records the count it wrote, so a
partial failure (e.g. Open-Meteo rate-limited) doesn't take down the whole
pull — the caller sees a dict and can decide what to do.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from nfl_model.data.sources import (
    aggregations,
    nflverse,
    venues as venues_src,
    weather as weather_src,
)
from nfl_model.data.warehouse import init_schema, query
from nfl_model.logging import get_logger

log = get_logger("data.pipeline")


def ensure_venues() -> int:
    """Initialize the schema and seed the static venues table."""
    init_schema()
    return venues_src.ensure_venues()


def pull_season(
    season: int,
    *,
    with_weather: bool = True,
    with_lines: bool = True,
) -> dict[str, Any]:
    """Pull every source for ``season`` into the warehouse.

    Order matters: schedules first (everything else FKs to game_id),
    then PBP (so the aggregation step has plays to crunch), then the
    per-game tables (injuries / snaps / officials), then derived
    aggregates, then weather + lines.

    Returns a dict mapping source name → row count written (or an error
    payload). The CLI prints this table; the morning_sync job logs it.
    """
    init_schema()
    counts: dict[str, Any] = {"season": season}

    # -- 1) Schedules → games --------------------------------------------
    try:
        sched = nflverse.pull_schedules([season])
        counts["games"] = int(len(sched))
    except Exception as exc:  # noqa: BLE001
        log.exception("pipeline.schedules_failed", season=season)
        counts["games"] = {"error": str(exc)}
        sched = pd.DataFrame()

    # -- 2) PBP → pbp ----------------------------------------------------
    try:
        pbp = nflverse.pull_pbp([season])
        counts["pbp"] = int(len(pbp))
    except Exception as exc:  # noqa: BLE001
        log.exception("pipeline.pbp_failed", season=season)
        counts["pbp"] = {"error": str(exc)}

    # -- 3) Per-game tables: injuries, snaps, officials ------------------
    for name, fn in (
        ("injuries", nflverse.pull_injuries),
        ("snap_counts", nflverse.pull_snap_counts),
        ("officials", nflverse.pull_officials),
    ):
        try:
            df = fn([season])
            counts[name] = int(len(df))
        except Exception as exc:  # noqa: BLE001
            log.exception(f"pipeline.{name}_failed", season=season)
            counts[name] = {"error": str(exc)}

    # -- 4) Derived aggregations on top of PBP ---------------------------
    try:
        counts["team_game_stats"] = aggregations.refresh_team_game_stats(season=season)
    except Exception as exc:  # noqa: BLE001
        log.exception("pipeline.team_game_stats_failed", season=season)
        counts["team_game_stats"] = {"error": str(exc)}
    try:
        counts["qb_game_stats"] = aggregations.refresh_qb_game_stats(season=season)
    except Exception as exc:  # noqa: BLE001
        log.exception("pipeline.qb_game_stats_failed", season=season)
        counts["qb_game_stats"] = {"error": str(exc)}
    try:
        counts["officiating_totals"] = aggregations.refresh_officiating_counts(season=season)
    except Exception as exc:  # noqa: BLE001
        log.exception("pipeline.officiating_totals_failed", season=season)
        counts["officiating_totals"] = {"error": str(exc)}

    # -- 5) Lines: closing-line snapshots from the schedule feed ---------
    if with_lines:
        try:
            odds = nflverse.pull_odds_from_schedules([season])
            counts["odds_history"] = int(len(odds))
        except Exception as exc:  # noqa: BLE001
            log.exception("pipeline.odds_failed", season=season)
            counts["odds_history"] = {"error": str(exc)}

    # -- 6) Weather ------------------------------------------------------
    if with_weather:
        try:
            games_for_weather = query(
                "SELECT game_id, kickoff_ts, game_date, stadium_id, roof "
                "FROM games WHERE season = ?",
                (int(season),),
            )
            n = weather_src.pull_weather_for_games(games_for_weather)
            counts["weather"] = int(n)
        except Exception as exc:  # noqa: BLE001
            log.exception("pipeline.weather_failed", season=season)
            counts["weather"] = {"error": str(exc)}

    log.info("pipeline.done", **{k: v for k, v in counts.items() if not isinstance(v, dict)})
    return counts
