"""Top-level feature assembler.

Joins all per-game feature builders into a single wide DataFrame keyed by
game_id, then persists to the warehouse ``features`` table.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from nfl_model.data.warehouse import query, upsert_dataframe
from nfl_model.logging import get_logger

log = get_logger("features.assemble")


def build_features_table(start_season: int, end_season: int) -> pd.DataFrame:
    """Build features for every game in [start_season, end_season].

    Lazily imports each builder so the module surface stays small while
    individual phases progress.
    """
    games = query(
        """
        SELECT g.game_id, g.season, g.week, g.game_date, g.kickoff_ts,
               g.home_team, g.away_team, g.home_qb, g.away_qb,
               g.home_coach, g.away_coach,
               g.home_score, g.away_score, g.home_win, g.overtime,
               g.stadium_id, g.roof, g.surface, g.weekday, g.primetime,
               g.divisional, g.referee
        FROM games g
        WHERE g.season BETWEEN ? AND ?
        ORDER BY g.season, g.week, g.game_date
        """,
        (start_season, end_season),
    )
    if games.empty:
        log.warning("features.no_games", start=start_season, end=end_season)
        return pd.DataFrame()

    # Lazy import + apply each feature builder. Each returns a DataFrame
    # keyed by game_id that we left-join to the games skeleton.
    from nfl_model.features import (
        coach_matchup,
        coaching,
        drive_eff,
        injuries as inj_feat,
        kicking,
        lookahead,
        market,
        officiating as off_feat,
        ol_continuity,
        pace_proe,
        qb_form,
        qb_tier,
        schedule_context,
        situational,
        star_players,
        success_explosive,
        surface_weather,
        team_epa,
    )

    frames: list[pd.DataFrame] = [games]
    for module in (
        team_epa, qb_form, inj_feat, pace_proe, schedule_context,
        surface_weather, market, coaching, success_explosive,
        off_feat, kicking, situational,
        # New for v2: HC-as-entity, star-player WOWY, drive efficiency,
        # OL continuity, QB tier.
        drive_eff, ol_continuity, qb_tier, star_players,
        # New for v3: head-to-head coach matchup history.
        coach_matchup,
        # New for v4: lookahead / letdown opponent strength.
        lookahead,
    ):
        try:
            piece = module.build(games)
        except Exception as exc:  # noqa: BLE001
            log.warning("features.builder_failed", module=module.__name__, err=str(exc))
            continue
        if piece is None or piece.empty:
            continue
        if "game_id" not in piece.columns:
            log.warning("features.builder_no_game_id", module=module.__name__)
            continue
        frames.append(piece)

    out = frames[0]
    for piece in frames[1:]:
        out = out.merge(piece, on="game_id", how="left", suffixes=("", "_dup"))
        # Drop accidental duplicate columns
        dup_cols = [c for c in out.columns if c.endswith("_dup")]
        if dup_cols:
            out = out.drop(columns=dup_cols)

    # Targets
    out["target_home_score"] = pd.to_numeric(out["home_score"], errors="coerce").astype("Int64")
    out["target_away_score"] = pd.to_numeric(out["away_score"], errors="coerce").astype("Int64")
    out["target_home_win"] = out["home_win"]
    out["target_total"] = out["target_home_score"].astype("Int64") + out["target_away_score"].astype("Int64")
    out["target_margin"] = out["target_home_score"].astype("Int64") - out["target_away_score"].astype("Int64")

    # Persist a JSON-blob version to the warehouse for fast retrieval by the model
    feature_cols = [
        c for c in out.columns
        if c not in {
            "game_id", "season", "week", "game_date", "kickoff_ts",
            "home_team", "away_team", "home_qb", "away_qb",
            "home_coach", "away_coach",
            "home_score", "away_score", "home_win", "overtime",
            "stadium_id", "roof", "surface", "weekday", "primetime",
            "divisional", "referee",
            "target_home_score", "target_away_score", "target_home_win",
            "target_total", "target_margin",
        }
    ]
    if feature_cols:
        feat_records = out[feature_cols].apply(
            lambda r: json.dumps(
                {k: (None if (isinstance(v, float) and np.isnan(v)) else v) for k, v in r.items()},
                default=str,
            ),
            axis=1,
        )
        warehouse_df = pd.DataFrame(
            {
                "game_id": out["game_id"],
                "game_date": out["game_date"],
                "season": out["season"],
                "week": out["week"],
                "target_home_win": out["target_home_win"],
                "target_home_score": out["target_home_score"],
                "target_away_score": out["target_away_score"],
                "target_total": out["target_total"],
                "target_margin": out["target_margin"],
                "feature_json": feat_records,
            }
        )
        warehouse_df = warehouse_df.dropna(subset=["game_id"]).drop_duplicates("game_id", keep="last")
        upsert_dataframe(warehouse_df, "features", key_columns=["game_id"])

    log.info("features.built", n_games=len(out), n_features=len(feature_cols))
    return out
