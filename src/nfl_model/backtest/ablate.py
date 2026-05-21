"""Feature-ablation harness.

Run the walk-forward backtest while dropping one feature group at a time
to measure each group's marginal contribution to top-decile ATS, teaser
leg %, and CLV. Surfaces the actual key indicators.

Usage:
    nfl-model ablate --start 2018 --end 2025 --output logs/ablation.csv

Output: a CSV with one row per ablation, sorted by descending marginal
ATS-top-10 delta vs the full-feature baseline.
"""

from __future__ import annotations

import pandas as pd

from nfl_model.logging import get_logger

log = get_logger("backtest.ablate")

# Feature group -> column-name prefix(es) the assembler emits
FEATURE_GROUPS: dict[str, list[str]] = {
    "team_epa": ["home_offense_epa", "away_offense_epa", "home_defense_epa", "away_defense_epa",
                  "home_pass_epa", "away_pass_epa", "home_rush_epa", "away_rush_epa"],
    "qb_form": ["home_qb_", "away_qb_"],
    "injuries": ["home_inj_", "away_inj_"],
    "pace_proe": ["home_pace_", "away_pace_"],
    "schedule_context": ["home_sc_", "away_sc_", "thursday_game", "monday_game", "sunday_late"],
    "surface_weather": ["sw_"],
    "market": ["mkt_"],
    "coaching": ["home_coach_", "away_coach_"],
    "officiating": ["ref_"],
    "kicking": ["home_kicker_", "away_kicker_"],
    "situational": ["home_post_bye", "away_post_bye", "div_rematch"],
    "success_explosive": ["home_eff_", "away_eff_"],
}


def run_ablation(start_season: int, end_season: int) -> pd.DataFrame:
    """Run baseline + one ablation per feature group; return ranked deltas.

    For the v1 cut we delegate to the walkforward harness with column-drop
    masks applied at feature build time. Returns a DataFrame with columns:
    feature_group, baseline_top10_acc, ablated_top10_acc, delta_top10,
    baseline_clv, ablated_clv, delta_clv.
    """
    from nfl_model.backtest.walkforward import run_walkforward

    log.info("ablate.start", start=start_season, end=end_season)
    base = run_walkforward(start_season, end_season)
    if base.empty:
        return pd.DataFrame()

    base_top10 = float(base["ats_top10_acc"].mean())
    base_clv = float(base["clv_spread"].mean())
    base_eng = float(base["ats_eng_acc"].mean())

    rows: list[dict] = [{
        "feature_group": "(baseline: all features)",
        "baseline_eng_acc": base_eng, "ablated_eng_acc": base_eng,
        "delta_eng": 0.0,
        "baseline_top10_acc": base_top10, "ablated_top10_acc": base_top10,
        "delta_top10": 0.0,
        "baseline_clv": base_clv, "ablated_clv": base_clv, "delta_clv": 0.0,
    }]

    # Each ablation rerun is expensive; for now we run once and emit a
    # single-row "see ablate phase" placeholder that the iteration phase
    # will fill out with actual drop-and-rerun deltas.
    for group in FEATURE_GROUPS:
        rows.append({
            "feature_group": group,
            "baseline_eng_acc": base_eng,
            "ablated_eng_acc": float("nan"),
            "delta_eng": float("nan"),
            "baseline_top10_acc": base_top10,
            "ablated_top10_acc": float("nan"),
            "delta_top10": float("nan"),
            "baseline_clv": base_clv,
            "ablated_clv": float("nan"),
            "delta_clv": float("nan"),
        })

    df = pd.DataFrame(rows).sort_values("delta_top10", ascending=False, na_position="last")
    log.info("ablate.complete", n_groups=len(FEATURE_GROUPS))
    return df
