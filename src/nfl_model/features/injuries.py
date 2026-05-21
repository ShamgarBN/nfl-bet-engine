"""Injury impact features.

For each game, computes:
- ``{home,away}_inj_count_starters_out``: number of probable starters with
  game_status in {Out, Doubtful}
- ``{home,away}_inj_qb_out``: explicit QB-out flag (highest-impact)
- ``{home,away}_inj_ol_starters_out``: OL-specific outs (drives sack rate)
- ``{home,away}_inj_top_skill_out``: top-skill (RB1/WR1/WR2/TE1) outs
- ``{home,away}_inj_secondary_out``: top-2 CB outs

Snap-share-weighted impact score: each out is weighted by the player's
trailing 4-game offensive/defensive snap share from the snap_counts table.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from nfl_model.data.warehouse import query
from nfl_model.logging import get_logger

log = get_logger("features.injuries")

POS_OL = {"OT", "OG", "C", "OL", "T", "G"}
POS_SKILL = {"WR", "RB", "TE"}
POS_CB = {"CB", "DB"}
OUT_STATUSES = {"Out", "Doubtful"}


def build(games: pd.DataFrame) -> pd.DataFrame:
    if games.empty:
        return pd.DataFrame()

    inj = query(
        """
        SELECT i.season, i.week, i.team, i.player_id, i.position, i.game_status
        FROM injuries i
        """
    )
    if inj.empty:
        return pd.DataFrame({"game_id": games["game_id"]})

    inj = inj[inj["game_status"].isin(OUT_STATUSES)].copy()

    # Pull recent snap shares to weight each out's impact
    snaps = query(
        """
        SELECT s.game_id, s.player_id, s.team, s.position,
               s.offense_pct, s.defense_pct, g.season, g.week, g.game_date
        FROM snap_counts s
        JOIN games g USING(game_id)
        """
    )
    snaps["game_date"] = pd.to_datetime(snaps["game_date"], errors="coerce")
    snaps = snaps.sort_values(["player_id", "game_date"])

    snap_share = (
        snaps.groupby("player_id")[["offense_pct", "defense_pct"]]
        .apply(lambda x: x.shift(1).rolling(4, min_periods=1).mean())
        .reset_index(level=0, drop=True)
    )
    snaps[["off_share_4", "def_share_4"]] = snap_share

    inj_with_share = inj.merge(
        snaps[["player_id", "season", "week", "off_share_4", "def_share_4"]],
        on=["player_id", "season", "week"],
        how="left",
    )

    def _summarize(grp: pd.DataFrame) -> pd.Series:
        starters_out = (grp["off_share_4"] >= 0.40).sum() + (grp["def_share_4"] >= 0.40).sum()
        qb_out = ((grp["position"] == "QB")).any()
        ol_out = grp["position"].isin(POS_OL).sum()
        skill_out = grp["position"].isin(POS_SKILL).sum()
        cb_out = grp["position"].isin(POS_CB).sum()
        impact = (
            grp["off_share_4"].fillna(0).sum() + grp["def_share_4"].fillna(0).sum()
        )
        return pd.Series(
            {
                "inj_starters_out": int(starters_out),
                "inj_qb_out": bool(qb_out),
                "inj_ol_out": int(ol_out),
                "inj_skill_out": int(skill_out),
                "inj_cb_out": int(cb_out),
                "inj_impact_score": float(impact),
            }
        )

    summary = (
        inj_with_share.groupby(["season", "week", "team"], as_index=False)
        .apply(_summarize, include_groups=False)
        .reset_index(drop=True)
    )

    home = games[["game_id", "season", "week", "home_team"]].rename(
        columns={"home_team": "team"}
    ).merge(summary, on=["season", "week", "team"], how="left").drop(columns=["team"])
    away = games[["game_id", "season", "week", "away_team"]].rename(
        columns={"away_team": "team"}
    ).merge(summary, on=["season", "week", "team"], how="left").drop(columns=["team"])

    home = home.rename(columns={c: f"home_{c}" for c in home.columns if c.startswith("inj_")})
    away = away.rename(columns={c: f"away_{c}" for c in away.columns if c.startswith("inj_")})

    out = home.merge(away, on=["game_id", "season", "week"], how="outer")
    log.debug("injuries.built", n=len(out))
    return out.drop(columns=["season", "week"], errors="ignore")
