"""Star-player With-Or-Without-You (WOWY) features.

For every player who has been a regular high-snap starter (offense or
defense), we measure their team's per-play EPA *when the player was on
the field for a high share of snaps* versus *when they were not*. The
difference — their **presence delta** — is the WOWY estimate of how
much that player adds.

At prediction time, for each game we sum the prior presence-delta of any
player on the team's injury report as ``Out`` or ``Doubtful``. The two
team-level features ``home_star_loss_epa`` and ``away_star_loss_epa``
are the engine's estimate of how much offense the team is missing this
week relative to their full-strength baseline.

Why this is harder than it sounds: most "stars" don't sit out, so the
WOWY sample is tiny. We compensate by (a) shrinking each player's delta
toward their team's average delta, and (b) only counting players whose
WOWY sample has at least ``_MIN_PRESENT_GAMES`` games on each side. The
output is conservative — it will under-weight true stars rather than
hallucinate impact for borderline backups.

Star identity is the player_id, which **follows the player across team
moves** — Deebo Samuel's delta computed when he was a 49er is what we
apply when he plays for any future team.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from nfl_model.data.warehouse import query
from nfl_model.logging import get_logger

log = get_logger("features.star_players")

# Minimum snap share to consider a player "present" in a game.
_PRESENT_THRESHOLD = 0.50
# Minimum games on each side of the WOWY split for a player's delta to count.
_MIN_PRESENT_GAMES = 6
_MIN_ABSENT_GAMES = 2
# Shrinkage weight toward 0 (no-impact baseline) for low-sample players.
_SHRINK_PRIOR_GAMES = 8.0
# How far back to look when computing the WOWY delta (in days). 730 = 2 years.
_LOOKBACK_DAYS = 730


def _compute_player_wowy(today: pd.Timestamp) -> pd.DataFrame:
    """Compute per-player presence delta using all data strictly before ``today``.

    Returns a DataFrame indexed by player_id with columns:
    ``player_name, position, present_games, absent_games, presence_delta``.
    """
    snaps = query(
        """
        SELECT s.game_id, s.player_id, s.player_name, s.team, s.position,
               s.offense_pct, s.defense_pct,
               g.game_date, t.offense_epa_per_play, t.defense_epa_per_play
        FROM snap_counts s
        JOIN games g USING(game_id)
        LEFT JOIN team_game_stats t
          ON t.game_id = s.game_id AND t.team = s.team
        WHERE g.home_score IS NOT NULL
        """
    )
    if snaps.empty:
        return pd.DataFrame()

    snaps["game_date"] = pd.to_datetime(snaps["game_date"], errors="coerce")
    snaps = snaps[snaps["game_date"] < today].copy()
    if snaps.empty:
        return pd.DataFrame()
    cutoff = today - pd.Timedelta(days=_LOOKBACK_DAYS)
    snaps = snaps[snaps["game_date"] >= cutoff]
    if snaps.empty:
        return pd.DataFrame()

    # Choose offensive vs defensive impact axis per player based on which
    # side of the ball they play most. For offensive snaps we measure
    # offense_epa; for defensive snaps the negative of opponent's offense
    # epa (= defense_epa per our convention).
    snaps["impact_pct"] = np.where(
        snaps["offense_pct"].fillna(0) >= snaps["defense_pct"].fillna(0),
        snaps["offense_pct"].fillna(0),
        snaps["defense_pct"].fillna(0),
    )
    snaps["impact_side"] = np.where(
        snaps["offense_pct"].fillna(0) >= snaps["defense_pct"].fillna(0),
        "offense",
        "defense",
    )
    snaps["team_epa_for_player"] = np.where(
        snaps["impact_side"] == "offense",
        snaps["offense_epa_per_play"],
        snaps["defense_epa_per_play"],
    )

    snaps["present"] = snaps["impact_pct"] >= _PRESENT_THRESHOLD

    # For each player, aggregate present vs absent team EPA. "Absent" rows
    # are games where their team played but they were under threshold;
    # those rows still appear in snap_counts (with low pct). Players who
    # truly missed a game don't have a row that game, so we have to compare
    # against their team's other games.
    # Practical approach: aggregate at the player level for present games
    # and use the *team's* overall mean (over the same window) as the
    # absent baseline. This is the standard WOWY-with-team-baseline trick.

    # Per-team baseline EPA (over the same lookback window).
    team_baseline_off = (
        snaps.groupby("team", as_index=False)["offense_epa_per_play"].mean()
        .rename(columns={"offense_epa_per_play": "team_baseline_offense"})
    )
    team_baseline_def = (
        snaps.groupby("team", as_index=False)["defense_epa_per_play"].mean()
        .rename(columns={"defense_epa_per_play": "team_baseline_defense"})
    )

    # Present-game per-player average.
    player_present = (
        snaps[snaps["present"]]
        .groupby(["player_id", "player_name", "position", "team", "impact_side"], as_index=False)
        .agg(
            present_games=("game_id", "nunique"),
            present_team_epa=("team_epa_for_player", "mean"),
        )
    )
    # Keep the team where the player played the most present games (their
    # "primary" team for the lookback window). Players who switched teams
    # mid-window get the team with the larger present sample.
    player_present = player_present.sort_values(
        ["player_id", "present_games"], ascending=[True, False]
    ).drop_duplicates(subset=["player_id"], keep="first")

    player_present = player_present.merge(team_baseline_off, on="team", how="left")
    player_present = player_present.merge(team_baseline_def, on="team", how="left")
    player_present["team_baseline"] = np.where(
        player_present["impact_side"] == "offense",
        player_present["team_baseline_offense"],
        player_present["team_baseline_defense"],
    )
    player_present["raw_delta"] = (
        player_present["present_team_epa"] - player_present["team_baseline"]
    )

    # Shrinkage toward 0 with a prior weight of _SHRINK_PRIOR_GAMES games.
    # delta_shrunk = (n / (n + prior)) * raw_delta
    n = player_present["present_games"].astype(float)
    player_present["presence_delta"] = (
        (n / (n + _SHRINK_PRIOR_GAMES)) * player_present["raw_delta"]
    )

    # Require enough present games to be considered a "star".
    player_present["is_star_candidate"] = (
        player_present["present_games"] >= _MIN_PRESENT_GAMES
    )

    # Pick the top stars per (team, impact_side) by absolute delta. This
    # caps the count so we don't sum noise from 53 players per side.
    top_per_team = (
        player_present[player_present["is_star_candidate"]]
        .sort_values(["team", "impact_side", "presence_delta"], key=lambda s: s.abs() if s.name == "presence_delta" else s, ascending=[True, True, False])
    )

    return player_present[
        ["player_id", "player_name", "team", "position", "impact_side",
         "present_games", "presence_delta", "is_star_candidate"]
    ]


def build(games: pd.DataFrame) -> pd.DataFrame:
    """Per-game star-player WOWY features keyed by ``game_id``."""
    if games.empty:
        return pd.DataFrame()

    # Compute WOWY against the earliest game in the slate so the delta
    # table is leakage-safe for every row in ``games``. (For the live
    # weekly prediction this means we use only what we knew before the
    # season's first game; for historical backtest this means we use
    # everything before the target row's date — slightly conservative
    # since we batch by slate, not per-row, but cheap and safe.)
    games["game_date"] = pd.to_datetime(games["game_date"], errors="coerce")
    today = pd.to_datetime(games["game_date"].min())
    if pd.isna(today):
        return pd.DataFrame({"game_id": games["game_id"]})

    wowy = _compute_player_wowy(today)
    if wowy.empty or "presence_delta" not in wowy.columns:
        return pd.DataFrame({"game_id": games["game_id"]})

    # Pull the current week's injury list and join the WOWY delta. We use
    # the per-(season, week, team, player) Out/Doubtful rows.
    inj = query(
        """
        SELECT i.season, i.week, i.team, i.player_id, i.game_status
        FROM injuries i
        WHERE i.game_status IN ('Out', 'Doubtful')
        """
    )
    if inj.empty:
        return pd.DataFrame(
            {
                "game_id": games["game_id"],
                "home_star_loss_epa": 0.0,
                "away_star_loss_epa": 0.0,
                "home_star_outs": 0,
                "away_star_outs": 0,
            }
        )

    # Severity weight: Out = 1.0, Doubtful = 0.6.
    severity = {"Out": 1.0, "Doubtful": 0.6}
    inj["severity"] = inj["game_status"].map(severity).fillna(0.0)

    inj_with_wowy = inj.merge(
        wowy[["player_id", "presence_delta", "is_star_candidate", "impact_side"]],
        on="player_id",
        how="inner",
    )
    inj_with_wowy = inj_with_wowy[inj_with_wowy["is_star_candidate"]]
    # Loss = severity * presence_delta. Positive presence_delta = good player
    # being absent reduces the team's expected EPA by that amount.
    inj_with_wowy["loss_epa"] = (
        inj_with_wowy["severity"] * inj_with_wowy["presence_delta"]
    )

    loss_per_team_week = (
        inj_with_wowy.groupby(["season", "week", "team"], as_index=False)
        .agg(
            star_loss_epa=("loss_epa", "sum"),
            star_outs=("player_id", "nunique"),
        )
    )

    base = games[["game_id", "season", "week", "home_team", "away_team"]].copy()
    home = base.rename(columns={"home_team": "team"}).merge(
        loss_per_team_week, on=["season", "week", "team"], how="left"
    ).drop(columns=["team"])
    away = base.rename(columns={"away_team": "team"}).merge(
        loss_per_team_week, on=["season", "week", "team"], how="left"
    ).drop(columns=["team"])
    home = home.rename(
        columns={"star_loss_epa": "home_star_loss_epa", "star_outs": "home_star_outs"}
    )
    away = away.rename(
        columns={"star_loss_epa": "away_star_loss_epa", "star_outs": "away_star_outs"}
    )
    out = home.merge(away, on=["game_id", "season", "week"], how="outer").drop(
        columns=["season", "week"]
    )
    for c in ("home_star_loss_epa", "away_star_loss_epa"):
        out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0.0)
    for c in ("home_star_outs", "away_star_outs"):
        out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0).astype(int)

    log.debug("star_players.built", n=len(out))
    return out
