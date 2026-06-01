"""Feature-ablation harness.

Run the walk-forward backtest while dropping one feature group at a time
to measure each group's marginal contribution to engine-bets ATS,
top-10% ATS, and CLV. Surfaces the actual key indicators.

Usage:
    nfl-model ablate --start 2018 --end 2024 --output logs/ablation.csv

Each ablation rerun is a full backtest, so this is heavy. By default we
run the baseline + every group in ``FEATURE_GROUPS``. Pass
``--groups market,team_epa`` to scope to a subset while iterating.
"""

from __future__ import annotations

from typing import Iterable

import pandas as pd

from nfl_model.logging import get_logger

log = get_logger("backtest.ablate")

# Feature group -> column-name prefix(es) the assembler emits. A column is
# considered "in the group" if any of its prefixes matches. The matching is
# *startswith*, so "home_qb_" catches every column produced by qb_form.py.
FEATURE_GROUPS: dict[str, list[str]] = {
    "team_epa": [
        "home_offense_epa", "away_offense_epa",
        "home_defense_epa", "away_defense_epa",
        "home_pass_epa", "away_pass_epa",
        "home_rush_epa", "away_rush_epa",
        "home_success_rate", "away_success_rate",
        "home_explosive_rate", "away_explosive_rate",
    ],
    "qb_form": ["home_qb_", "away_qb_"],
    "injuries": ["home_inj_", "away_inj_"],
    "pace_proe": ["home_pace_", "away_pace_"],
    "schedule_context": [
        "home_sc_", "away_sc_",
        "thursday_game", "monday_game", "sunday_late", "primetime", "divisional",
    ],
    "surface_weather": ["sw_"],
    "market": ["mkt_"],
    "coaching": ["home_coach_", "away_coach_"],
    "officiating": ["ref_"],
    "kicking": ["home_kicker_", "away_kicker_"],
    "situational": ["home_post_bye", "away_post_bye", "div_rematch"],
    "success_explosive": ["home_eff_", "away_eff_"],
    # v2 additions
    "drive_eff": ["home_drv_", "away_drv_"],
    "ol_continuity": ["home_ol_", "away_ol_"],
    "qb_tier": ["home_qb_tier", "away_qb_tier"],
    "star_players": ["home_star_", "away_star_"],
    # v3 additions
    "coach_matchup": ["home_coach_h2h_"],
    # The travel-direction extension hides inside schedule_context's
    # home_sc_/away_sc_ prefix, so it's already covered there.
    # v4 additions
    "lookahead": [
        "home_prev_opp_strength", "home_next_opp_strength", "home_sandwich_game",
        "away_prev_opp_strength", "away_next_opp_strength", "away_sandwich_game",
    ],
}


def _columns_for_group(group: str, all_cols: list[str]) -> list[str]:
    """Resolve the actual feature columns belonging to ``group``."""
    prefixes = FEATURE_GROUPS.get(group, [])
    return [c for c in all_cols if any(c.startswith(p) for p in prefixes)]


def run_ablation(
    start_season: int,
    end_season: int,
    *,
    groups: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Baseline + per-group ablations. Returns a DataFrame ranked by delta.

    For each named group, runs the walk-forward backtest with that group's
    columns excluded from the feature matrix. Compares to the baseline run
    (all features). Negative ``delta_top10`` means dropping the group *hurt*
    accuracy → the group is contributing. Positive ``delta_top10`` means
    dropping the group *helped* → the group is hurting / noisy.
    """
    from nfl_model.backtest.walkforward import run_walkforward_with_drop_groups

    targets = list(groups) if groups else list(FEATURE_GROUPS.keys())

    log.info("ablate.start", start=start_season, end=end_season, groups=targets)
    base = run_walkforward_with_drop_groups(start_season, end_season, drop_groups=[])
    if base.empty:
        return pd.DataFrame()
    base_top10 = float(base["ats_top10_acc"].mean())
    base_eng = float(base["ats_eng_acc"].mean())
    base_ou_eng = float(base["ou_eng_acc"].mean())
    base_clv = float(base["clv_spread"].mean())

    rows: list[dict] = [{
        "feature_group": "(baseline: all features)",
        "ablated_eng_acc": base_eng, "delta_eng": 0.0,
        "ablated_top10_acc": base_top10, "delta_top10": 0.0,
        "ablated_ou_eng_acc": base_ou_eng, "delta_ou_eng": 0.0,
        "ablated_clv": base_clv, "delta_clv": 0.0,
    }]

    for group in targets:
        if group not in FEATURE_GROUPS:
            log.warning("ablate.unknown_group", group=group)
            continue
        log.info("ablate.run", group=group)
        sub = run_walkforward_with_drop_groups(
            start_season, end_season, drop_groups=[group]
        )
        if sub.empty:
            continue
        ablated_top10 = float(sub["ats_top10_acc"].mean())
        ablated_eng = float(sub["ats_eng_acc"].mean())
        ablated_ou_eng = float(sub["ou_eng_acc"].mean())
        ablated_clv = float(sub["clv_spread"].mean())
        rows.append({
            "feature_group": group,
            "ablated_eng_acc": ablated_eng,
            "delta_eng": ablated_eng - base_eng,
            "ablated_top10_acc": ablated_top10,
            "delta_top10": ablated_top10 - base_top10,
            "ablated_ou_eng_acc": ablated_ou_eng,
            "delta_ou_eng": ablated_ou_eng - base_ou_eng,
            "ablated_clv": ablated_clv,
            "delta_clv": ablated_clv - base_clv,
        })

    # Sort by delta_top10 ascending: the *most negative* deltas (i.e. dropping
    # this group hurt the most) come first — those are the most valuable groups.
    df = pd.DataFrame(rows).sort_values("delta_top10", ascending=True, na_position="last")
    log.info("ablate.complete", n_groups=len(targets))
    return df
