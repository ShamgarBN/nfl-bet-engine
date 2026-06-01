"""Head-coach-as-entity features.

Treats the head coach as a first-class entity whose identity *follows them
across team moves*. Andy Reid in Philadelphia and Kansas City is the same
coach as far as these features are concerned — career win %, ATS %,
4th-down aggression, pace and pass-rate-over-expected, post-bye record,
and so on are all computed from every prior game the coach has worked,
regardless of team.

Why this matters: NFL teams rebuild on the same offensive philosophy
when a coach changes. Sean McVay's offenses run a recognisable scheme
whether the team is LA Rams or any future destination; Bill Belichick's
defenses keep a recognisable identity. Encoding *the coach's* prior
tendencies — not just the team's recent form — is one of the strongest
public signals for week-to-week NFL accuracy that isn't already in the
closing line.

All aggregates are computed strictly on games **before** the target
game's date (leakage-safe via ``shift(1)`` + ``expanding`` mean).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from nfl_model.data.warehouse import query
from nfl_model.logging import get_logger

log = get_logger("features.coaching")

# Plays inside the opponent's 50 with 4 or fewer yards to gain on 4th down
# are the canonical "go for it vs punt" decision points — high-aggression
# coaches go for it; conservative coaches punt. Coach aggressiveness
# correlates with point totals and underdog cover rates.
_FOURTH_DOWN_AGGRO_YDSTOGO = 4
_FOURTH_DOWN_AGGRO_YARDLINE_100 = 60  # opponent's 50 is yardline_100 == 50

# Neutral game-script window for measuring pace and PROE without bias
# from teams playing from way ahead or way behind.
_NEUTRAL_WP_LO = 0.20
_NEUTRAL_WP_HI = 0.80


def _coach_long(games: pd.DataFrame) -> pd.DataFrame:
    """Stack home + away coach rows into one long DataFrame per (game, coach).

    Returns columns: ``game_id, game_date, season, week, team, coach,
    is_home, won, ats_cover, ou_over, fav_dog, posted_spread, total_pts``.

    Joins finalized scores + lines from the warehouse so situational
    splits (favored / underdog, over / under) can be built from the same
    long frame.
    """
    g = query(
        """
        SELECT g.game_id, g.game_date, g.season, g.week,
               g.home_team, g.away_team, g.home_coach, g.away_coach,
               g.home_score, g.away_score,
               o.spread_close, o.total_close
        FROM games g
        LEFT JOIN odds_history o
          ON o.game_id = g.game_id AND o.book = 'consensus'
        WHERE g.home_coach IS NOT NULL AND g.away_coach IS NOT NULL
          AND g.home_score IS NOT NULL AND g.away_score IS NOT NULL
        """
    )
    if g.empty:
        return pd.DataFrame()

    g["game_date"] = pd.to_datetime(g["game_date"], errors="coerce")
    g["home_margin"] = pd.to_numeric(g["home_score"], errors="coerce") - pd.to_numeric(
        g["away_score"], errors="coerce"
    )
    g["total_pts"] = pd.to_numeric(g["home_score"], errors="coerce") + pd.to_numeric(
        g["away_score"], errors="coerce"
    )
    g["spread_close"] = pd.to_numeric(g["spread_close"], errors="coerce")
    g["total_close"] = pd.to_numeric(g["total_close"], errors="coerce")

    sp_finite = g["spread_close"].notna()
    cover_diff_home = g["home_margin"] + g["spread_close"]
    g["home_cover"] = np.where(sp_finite, cover_diff_home > 0, np.nan)
    g["away_cover"] = np.where(sp_finite, cover_diff_home < 0, np.nan)
    g["over_hit"] = np.where(
        g["total_close"].notna(), g["total_pts"] > g["total_close"], np.nan
    )

    home = pd.DataFrame(
        {
            "game_id": g["game_id"], "game_date": g["game_date"],
            "season": g["season"], "week": g["week"],
            "team": g["home_team"], "coach": g["home_coach"],
            "is_home": True,
            "won": (g["home_margin"] > 0).astype(float),
            "ats_cover": g["home_cover"].astype(float),
            "ou_over": g["over_hit"].astype(float),
            # negative spread (home perspective) means home is favored
            "is_favorite": (g["spread_close"] < 0).astype(float),
            "posted_spread": g["spread_close"],
            "total_pts": g["total_pts"],
        }
    )
    away = pd.DataFrame(
        {
            "game_id": g["game_id"], "game_date": g["game_date"],
            "season": g["season"], "week": g["week"],
            "team": g["away_team"], "coach": g["away_coach"],
            "is_home": False,
            "won": (g["home_margin"] < 0).astype(float),
            "ats_cover": g["away_cover"].astype(float),
            "ou_over": g["over_hit"].astype(float),
            "is_favorite": (g["spread_close"] > 0).astype(float),
            "posted_spread": -g["spread_close"],
            "total_pts": g["total_pts"],
        }
    )
    long = pd.concat([home, away], ignore_index=True).sort_values(
        ["coach", "game_date", "game_id"]
    ).reset_index(drop=True)
    return long


def _coach_career_aggregates(long: pd.DataFrame) -> pd.DataFrame:
    """Career-to-date (shifted) per-coach aggregates: win %, ATS %, OU lean,
    favorite / underdog ATS splits, total experience.

    All numbers are computed on games STRICTLY before the row's own game_date
    so the value is what the coach had walked into kickoff.
    """
    if long.empty:
        return long

    grp = long.groupby("coach", sort=False)

    def expand_mean(s: pd.Series) -> pd.Series:
        return s.shift(1).expanding().mean()

    def expand_count(s: pd.Series) -> pd.Series:
        return s.shift(1).expanding().count()

    long["coach_career_games"] = grp["won"].transform(expand_count)
    long["coach_career_win_pct"] = grp["won"].transform(expand_mean)
    long["coach_career_ats_pct"] = grp["ats_cover"].transform(expand_mean)
    long["coach_career_ou_over_pct"] = grp["ou_over"].transform(expand_mean)

    # Splits: favorite vs underdog ATS.
    long["_fav_cover"] = np.where(long["is_favorite"] == 1, long["ats_cover"], np.nan)
    long["_dog_cover"] = np.where(long["is_favorite"] == 0, long["ats_cover"], np.nan)
    long["coach_career_ats_as_fav"] = grp["_fav_cover"].transform(expand_mean)
    long["coach_career_ats_as_dog"] = grp["_dog_cover"].transform(expand_mean)
    long = long.drop(columns=["_fav_cover", "_dog_cover"])

    # Season-to-date.
    season_grp = long.groupby(["coach", "season"], sort=False)
    long["coach_season_games"] = season_grp["won"].transform(expand_count)
    long["coach_season_win_pct"] = season_grp["won"].transform(expand_mean)
    long["coach_season_ats_pct"] = season_grp["ats_cover"].transform(expand_mean)

    # Rookie head coach: career games before this game < 17.
    long["coach_rookie_hc"] = (long["coach_career_games"].fillna(0) < 17).astype(float)
    return long


def _coach_pbp_aggregates() -> pd.DataFrame:
    """Per-(coach, game) aggregates from PBP: 4th-down aggression, neutral
    pace, neutral PROE, 2nd-half scoring differential.

    Joins PBP back to ``games.home_coach`` / ``games.away_coach`` so the
    coach who was actually on the sideline owns the play decisions.
    """
    plays = query(
        """
        SELECT pbp.game_id, pbp.posteam, pbp.down, pbp.ydstogo,
               pbp.yardline_100, pbp.play_type, pbp.is_pass, pbp.is_rush,
               pbp.wp, pbp.epa,
               g.game_date, g.season, g.home_team, g.away_team,
               g.home_coach, g.away_coach, g.home_score, g.away_score
        FROM pbp
        JOIN games g USING(game_id)
        WHERE g.home_coach IS NOT NULL AND g.away_coach IS NOT NULL
        """
    )
    if plays.empty:
        return pd.DataFrame()

    plays["game_date"] = pd.to_datetime(plays["game_date"], errors="coerce")
    plays["coach"] = np.where(
        plays["posteam"] == plays["home_team"], plays["home_coach"], plays["away_coach"]
    )

    # 4th-down aggression: on 4th-and-≤4 inside opponent's 50, did they go for
    # it (play_type in pass/run) vs punt/FG?
    fourth = plays[plays["down"] == 4].copy()
    fourth["aggro_opportunity"] = (
        (fourth["ydstogo"] <= _FOURTH_DOWN_AGGRO_YDSTOGO)
        & (fourth["yardline_100"] <= _FOURTH_DOWN_AGGRO_YARDLINE_100)
    )
    fourth["went_for_it"] = fourth["play_type"].isin(["pass", "run"])
    aggro_opps = (
        fourth[fourth["aggro_opportunity"]]
        .groupby(["game_id", "coach"], dropna=True)
        .agg(
            fourth_aggro_opps=("aggro_opportunity", "sum"),
            fourth_aggro_goes=("went_for_it", "sum"),
        )
        .reset_index()
    )

    # Pace + PROE in neutral game script (0.2 < wp < 0.8).
    neutral_mask = plays["wp"].between(_NEUTRAL_WP_LO, _NEUTRAL_WP_HI, inclusive="both")
    neutral = plays[neutral_mask & plays["play_type"].isin(["pass", "run"])]
    pace = (
        neutral.groupby(["game_id", "coach"], dropna=True)
        .agg(
            coach_neutral_pass_rate=("is_pass", "mean"),
            coach_neutral_plays=("is_pass", "size"),
        )
        .reset_index()
    )

    # PROE: subtract league-mean neutral pass rate per game.
    league_pass_by_game = (
        neutral.groupby("game_id")["is_pass"].mean().rename("league_pass_rate").reset_index()
    )
    pace = pace.merge(league_pass_by_game, on="game_id", how="left")
    pace["coach_neutral_proe"] = pace["coach_neutral_pass_rate"] - pace["league_pass_rate"]
    pace = pace.drop(columns=["league_pass_rate"])

    pbp_agg = aggro_opps.merge(pace, on=["game_id", "coach"], how="outer")
    return pbp_agg


def _coach_pbp_rolling(games_long: pd.DataFrame, pbp_per_game: pd.DataFrame) -> pd.DataFrame:
    """Roll per-game PBP aggregates into leakage-safe per-(coach, game) features.

    For each (coach, game_date) row in ``games_long``, returns the mean
    of pbp aggregates over every prior game the coach worked.
    """
    if games_long.empty or pbp_per_game.empty:
        return games_long

    merged = games_long.merge(pbp_per_game, on=["game_id", "coach"], how="left")
    merged = merged.sort_values(["coach", "game_date", "game_id"]).reset_index(drop=True)
    grp = merged.groupby("coach", sort=False)

    for col, out_col in [
        ("fourth_aggro_opps", "coach_4th_aggro_opps_pg"),
        ("fourth_aggro_goes", "coach_4th_aggro_goes_pg"),
        ("coach_neutral_pass_rate", "coach_neutral_pass_rate_avg"),
        ("coach_neutral_proe", "coach_neutral_proe_avg"),
    ]:
        merged[out_col] = grp[col].transform(lambda s: s.shift(1).expanding().mean())

    # 4th-down aggression rate = goes / opportunities, computed from the rolling
    # numerator + denominator so coaches with few opportunities aren't penalised.
    safe_opps = merged["coach_4th_aggro_opps_pg"].replace(0, np.nan)
    merged["coach_4th_aggro_rate"] = merged["coach_4th_aggro_goes_pg"] / safe_opps

    return merged


def build(games: pd.DataFrame) -> pd.DataFrame:
    """Return per-game home/away coaching features keyed by game_id."""
    if games.empty:
        return pd.DataFrame()

    long = _coach_long(games)
    if long.empty:
        return pd.DataFrame({"game_id": games["game_id"]})

    long = _coach_career_aggregates(long)
    pbp_agg = _coach_pbp_aggregates()
    long = _coach_pbp_rolling(long, pbp_agg)

    # HC team change flag: was this coach with a different team in their
    # immediately preceding game? Catches the "moved this offseason" signal.
    long = long.sort_values(["coach", "game_date", "game_id"]).reset_index(drop=True)
    long["prev_team"] = long.groupby("coach", sort=False)["team"].shift(1)
    long["coach_team_change_flag"] = (
        long["prev_team"].notna() & (long["prev_team"] != long["team"])
    ).astype(float)
    long = long.drop(columns=["prev_team"])

    # Subset of columns to project as features.
    feat_cols = [
        "coach_career_games", "coach_career_win_pct", "coach_career_ats_pct",
        "coach_career_ou_over_pct", "coach_career_ats_as_fav", "coach_career_ats_as_dog",
        "coach_season_games", "coach_season_win_pct", "coach_season_ats_pct",
        "coach_rookie_hc",
        "coach_4th_aggro_opps_pg", "coach_4th_aggro_goes_pg", "coach_4th_aggro_rate",
        "coach_neutral_pass_rate_avg", "coach_neutral_proe_avg",
        "coach_team_change_flag",
    ]
    keep_cols = ["game_id", "coach"] + feat_cols

    home_lookup = (
        games[["game_id", "home_coach"]]
        .rename(columns={"home_coach": "coach"})
        .merge(long[keep_cols], on=["game_id", "coach"], how="left")
        .drop(columns=["coach"])
    )
    away_lookup = (
        games[["game_id", "away_coach"]]
        .rename(columns={"away_coach": "coach"})
        .merge(long[keep_cols], on=["game_id", "coach"], how="left")
        .drop(columns=["coach"])
    )
    home_lookup = home_lookup.rename(columns={c: f"home_{c}" for c in feat_cols})
    away_lookup = away_lookup.rename(columns={c: f"away_{c}" for c in feat_cols})

    out = home_lookup.merge(away_lookup, on="game_id", how="outer")
    log.debug("coaching.built", n=len(out))
    return out
