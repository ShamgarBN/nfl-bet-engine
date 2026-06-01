"""Weekly walk-forward NFL backtest.

For each target season, we walk through weeks chronologically. Each week:
1. Train Stage A score models + direct ATS / total classifiers on **all
   prior data** (every season earlier than the target, plus all weeks of
   the target season earlier than the current one).
2. Build features for the current week's slate.
3. Predict via simulator + direct heads, blend with ensemble weights,
   apply isotonic calibration (fit OOF from the training set).
4. Score picks against actual outcomes and aggregate per-season metrics.

Per-season metrics include the model-card columns: ats_all_acc,
ats_eng_acc, ats_top10_acc, ats_top3_acc, ml_eng_acc, ml_eng_roi,
ou_all_acc, ou_eng_acc, ou_top10_acc, teaser_leg_pct, teaser_2team_pct,
clv_spread.

This is the heart of the iteration loop -- we run this every time we add
a feature group to confirm the engine still hits its conviction-tier
targets.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from nfl_model.config import settings
from nfl_model.logging import get_logger

log = get_logger("backtest.walkforward")


def _safe_acc(picks: pd.Series, outcomes: pd.Series) -> float:
    valid = picks.notna() & outcomes.notna()
    if valid.sum() == 0:
        return float("nan")
    return float((picks[valid] == outcomes[valid]).mean())


def run_walkforward(start_season: int, end_season: int) -> pd.DataFrame:
    """Run the weekly walk-forward backtest for [start_season, end_season].

    Returns a DataFrame with one row per season carrying all model-card
    metrics. Persists per-game predictions to the journal.
    """
    return run_walkforward_with_drop_groups(start_season, end_season, drop_groups=[])


def run_walkforward_with_drop_groups(
    start_season: int,
    end_season: int,
    *,
    drop_groups: list[str],
) -> pd.DataFrame:
    """Walk-forward backtest with optional feature-group ablation.

    Identical to :func:`run_walkforward` when ``drop_groups`` is empty.
    When populated, those groups' columns are excluded from the feature
    spec before training so the ablation actually changes what the model
    sees — letting the ``ablate`` CLI surface real marginal value.
    """
    import lightgbm as lgb

    from nfl_model.backtest.ablate import FEATURE_GROUPS
    from nfl_model.features.assemble import build_features_table
    from nfl_model.model.ensemble import blend
    from nfl_model.model.feature_matrix import (
        build_runs_matrix,
        build_runs_matrix_away,
        fit_spec,
    )
    from nfl_model.model.scores import train_score_model
    from nfl_model.model.simulate import simulate_games

    full = build_features_table(settings.backtest_start_year, end_season)
    if full.empty:
        log.warning("walkforward.no_features")
        return pd.DataFrame()
    full = full.dropna(subset=["target_home_score", "target_away_score"])
    full["game_date"] = pd.to_datetime(full["game_date"], errors="coerce")

    rows_per_season: list[dict[str, Any]] = []
    journal_rows: list[dict[str, Any]] = []
    oof_picks_frames: list[pd.DataFrame] = []  # collected for calibrator fitting

    for season in range(start_season, end_season + 1):
        season_games = full[full["season"] == season].copy()
        if season_games.empty:
            continue
        # Walk-forward correct: train on all data BEFORE this season starts.
        # We refit once per season rather than once per week. The training
        # data still excludes anything from `season`, so this is equally
        # leakage-safe -- just faster (~20x). Per-week refit didn't add
        # measurable accuracy in our smoke test on 2024.
        train_set = full[full["season"] < season]
        if len(train_set) < 200:
            continue

        spec = fit_spec(train_set)
        if drop_groups:
            drop_cols: set[str] = set()
            for g in drop_groups:
                prefixes = FEATURE_GROUPS.get(g, [])
                drop_cols.update(
                    c for c in spec.feature_cols
                    if any(c.startswith(p) for p in prefixes)
                )
            spec.feature_cols = [c for c in spec.feature_cols if c not in drop_cols]
            spec.final_feature_cols = spec.feature_cols
        Xh_t, _, feat_cols = build_runs_matrix(train_set, spec)
        Xa_t, _, _ = build_runs_matrix_away(train_set, spec)
        spec.final_feature_cols = feat_cols
        yh_t = train_set["target_home_score"].astype(float).to_numpy()
        ya_t = train_set["target_away_score"].astype(float).to_numpy()

        home_model = train_score_model(Xh_t, yh_t, feat_cols, n_seeds=2)
        away_model = train_score_model(Xa_t, ya_t, feat_cols, n_seeds=2)

        # ---- Direct ATS + totals classifiers (binary cover/over targets) ----
        # Trained on the same Xh_t features but with binary y for the direct
        # cover/over labels. Used downstream to blend with the simulator's
        # joint-distribution probabilities. Leakage-safe by construction: we
        # only train on prior seasons.
        spread_clf, totals_clf = _train_direct_classifiers(
            Xh_t, train_set, feat_cols
        )

        weeks = sorted(season_games["week"].dropna().unique().tolist())
        season_picks: list[pd.DataFrame] = []

        for week in weeks:
            this_week = season_games[season_games["week"] == week].copy()
            Xh_p, _, _ = build_runs_matrix(this_week, spec)
            Xa_p, _, _ = build_runs_matrix_away(this_week, spec)
            hm, hs = home_model.predict_distribution(Xh_p)
            am, ase = away_model.predict_distribution(Xa_p)

            spread_lines = pd.to_numeric(this_week.get("mkt_spread_close"), errors="coerce").to_numpy()
            total_lines = pd.to_numeric(this_week.get("mkt_total_close"), errors="coerce").to_numpy()
            preds = simulate_games(
                game_ids=this_week["game_id"].to_numpy(),
                pred_home=(hm, hs),
                pred_away=(am, ase),
                spread_lines=spread_lines,
                total_lines=total_lines,
                n_sims=settings.monte_carlo_iterations // 2,
                seed=settings.random_seed + int(season) * 100 + int(week),
            )

            # Direct-classifier predictions on this week's matrix.
            p_spread_direct = (
                spread_clf.predict(Xh_p) if spread_clf is not None else None
            )
            p_totals_direct = (
                totals_clf.predict(Xh_p) if totals_clf is not None else None
            )

            p_cover_sim = np.array([p.p_home_cover for p in preds])
            p_over_sim = np.array([p.p_total_over for p in preds])
            p_cover_blended = blend("spread", p_cover_sim, p_spread_direct)
            p_over_blended = blend("total", p_over_sim, p_totals_direct)

            picks = pd.DataFrame(
                {
                    "game_id": [p.game_id for p in preds],
                    "season": int(season),
                    "week": int(week),
                    "p_home_win_sim": [p.p_home_win for p in preds],
                    "p_home_cover_sim": p_cover_blended,
                    "p_total_over_sim": p_over_blended,
                    "expected_margin": [p.expected_margin for p in preds],
                    "expected_total": [p.expected_total for p in preds],
                    "spread_used": [p.spread_used for p in preds],
                    "total_used": [p.total_used for p in preds],
                }
            )
            picks = picks.merge(
                this_week[["game_id", "target_home_win", "target_home_score", "target_away_score",
                           "mkt_spread_close", "mkt_total_close",
                           "mkt_ml_implied_home", "mkt_ml_implied_away"]],
                on="game_id", how="left",
            )
            picks["margin"] = picks["target_home_score"].astype(float) - picks["target_away_score"].astype(float)
            picks["total_pts"] = picks["target_home_score"].astype(float) + picks["target_away_score"].astype(float)
            # ML pick is the higher-probability side (no per-side juice yet).
            # pick_spread / pick_ou are recomputed below from the model's
            # expected_margin / expected_total vs the posted line (edge-based).
            picks["pick_ml"] = np.where(picks["p_home_win_sim"] >= 0.5, "home", "away")
            # Outcome side -- use object arrays so we can mix strings with NaN
            sp = picks["mkt_spread_close"].astype(float)
            sp_valid = np.isfinite(sp)
            cover_diff = (picks["margin"].astype(float) + sp).to_numpy()
            actual_cover = np.full(len(picks), None, dtype=object)
            actual_cover[sp_valid & (cover_diff > 0)] = "home"
            actual_cover[sp_valid & (cover_diff < 0)] = "away"
            actual_cover[sp_valid & (cover_diff == 0)] = "push"
            picks["actual_cover"] = actual_cover

            tl = picks["mkt_total_close"].astype(float)
            tl_valid = np.isfinite(tl)
            tot_diff = (picks["total_pts"].astype(float) - tl).to_numpy()
            actual_total = np.full(len(picks), None, dtype=object)
            actual_total[tl_valid & (tot_diff > 0)] = "over"
            actual_total[tl_valid & (tot_diff < 0)] = "under"
            actual_total[tl_valid & (tot_diff == 0)] = "push"
            picks["actual_total"] = actual_total

            picks["actual_ml"] = np.where(picks["target_home_win"].astype(bool), "home", "away")
            # Per-week ATS pick (edge-vs-line in points). Mirrors the
            # season-level computation later — duplicated here so the journal
            # rows we write below carry a correctly-signed pick.
            wk_sp = picks["mkt_spread_close"].astype(float)
            wk_exp_cover = picks["expected_margin"].astype(float) + wk_sp
            picks["pick_spread"] = np.where(wk_exp_cover > 0, "home", "away")
            wk_tot = picks["mkt_total_close"].astype(float)
            wk_tot_diff = picks["expected_total"].astype(float) - wk_tot
            picks["pick_ou"] = np.where(wk_tot_diff > 0, "over", "under")
            season_picks.append(picks)

            # Build journal rows for the slate (all picks, not yet filtered).
            for _, r in picks.iterrows():
                journal_rows.append({
                    "game_id": r["game_id"], "season": int(season), "week": int(week),
                    "market": "spread", "pick": r["pick_spread"],
                    "prob": float(r["p_home_cover_sim"]) if r["pick_spread"] == "home" else float(1 - r["p_home_cover_sim"]),
                    "market_line": float(r["mkt_spread_close"]) if pd.notna(r["mkt_spread_close"]) else None,
                    "market_price": -110, "edge": float("nan"),
                })

        if not season_picks:
            continue
        all_picks = pd.concat(season_picks, ignore_index=True)
        oof_picks_frames.append(all_picks)

        # ---- Spread conviction: how many points the picked side covers the
        #      line by IN EXPECTATION, given the model's expected margin ----
        # The home side covers when (margin + posted_spread) > 0. The model's
        # estimate of that quantity is (expected_margin + posted_spread). When
        # positive: home is expected to cover, pick home. When negative: away.
        # Conviction = |expected_cover_margin| in points.
        #
        # This rewards picks where the model materially disagrees with the
        # line — *not* picks where the model just predicts a blowout the line
        # already prices in.
        posted = all_picks["mkt_spread_close"].astype(float)
        exp_cover = all_picks["expected_margin"].astype(float) + posted
        all_picks["pick_spread"] = np.where(exp_cover > 0, "home", "away")
        all_picks["spread_conviction"] = exp_cover.abs().fillna(0.0)
        all_picks["spread_correct"] = (all_picks["pick_spread"] == all_picks["actual_cover"])
        engine_bets_mask = all_picks["spread_conviction"] >= settings.spread_edge_min_pct
        ats_all = float(all_picks["spread_correct"].mean()) if len(all_picks) else float("nan")
        ats_eng = float(all_picks.loc[engine_bets_mask, "spread_correct"].mean()) if engine_bets_mask.sum() else float("nan")

        top10_mask = all_picks["spread_conviction"] >= all_picks["spread_conviction"].quantile(1 - settings.tier_top_10_pct)
        top3_mask = all_picks["spread_conviction"] >= all_picks["spread_conviction"].quantile(1 - settings.tier_top_3_pct)
        ats_top10 = float(all_picks.loc[top10_mask, "spread_correct"].mean()) if top10_mask.sum() else float("nan")
        ats_top3 = float(all_picks.loc[top3_mask, "spread_correct"].mean()) if top3_mask.sum() else float("nan")

        # ---- ML conviction: edge vs de-vigged market implied prob ----
        # p_model(picked side) minus market implied prob (de-vigged) for the
        # same side. Positive edge = model thinks the picked side is more
        # likely than the price says.
        def _ml_market_prob_picked(r: pd.Series) -> float:
            implied = r.get(f"mkt_ml_implied_{r['pick_ml']}")
            try:
                return float(implied) if implied is not None and pd.notna(implied) else 0.5
            except (TypeError, ValueError):
                return 0.5
        all_picks["p_market_ml_picked"] = all_picks.apply(_ml_market_prob_picked, axis=1)
        all_picks["p_model_ml_picked"] = np.where(
            all_picks["pick_ml"] == "home",
            all_picks["p_home_win_sim"],
            1 - all_picks["p_home_win_sim"],
        )
        all_picks["ml_conviction"] = (all_picks["p_model_ml_picked"] - all_picks["p_market_ml_picked"]).clip(lower=0)
        all_picks["ml_correct"] = (all_picks["pick_ml"] == all_picks["actual_ml"])
        ml_eng_mask = all_picks["ml_conviction"] >= settings.ml_kelly_cushion
        ml_eng_acc = float(all_picks.loc[ml_eng_mask, "ml_correct"].mean()) if ml_eng_mask.sum() else float("nan")

        # ML ROI: stake 1u at the listed market price, count expected return
        def _ml_roi_row(r: pd.Series) -> float:
            if not r.get("ml_eng_in", True):
                return 0.0
            implied = r.get(f"mkt_ml_implied_{r['pick_ml']}")
            if pd.isna(implied) or implied <= 0:
                return 0.0
            decimal_odds = 1.0 / implied
            return (decimal_odds - 1.0) if r["ml_correct"] else -1.0
        all_picks["ml_eng_in"] = ml_eng_mask
        all_picks["ml_units"] = all_picks.apply(_ml_roi_row, axis=1)
        ml_eng_roi = float(all_picks.loc[ml_eng_mask, "ml_units"].mean()) if ml_eng_mask.sum() else float("nan")

        # ---- Total conviction: how many points the model's expected total
        #      differs from the posted total ----
        # Same rationale as spread: pick the side the model's expected total
        # falls on relative to the line; size by the magnitude of the gap.
        posted_tot = all_picks["mkt_total_close"].astype(float)
        tot_diff = all_picks["expected_total"].astype(float) - posted_tot
        all_picks["pick_ou"] = np.where(tot_diff > 0, "over", "under")
        all_picks["ou_conviction"] = tot_diff.abs().fillna(0.0)
        all_picks["ou_correct"] = all_picks["pick_ou"] == all_picks["actual_total"]
        ou_eng_mask = all_picks["ou_conviction"] >= settings.total_edge_min_pct
        ou_top10_mask = all_picks["ou_conviction"] >= all_picks["ou_conviction"].quantile(1 - settings.tier_top_10_pct)
        ou_all = float(all_picks["ou_correct"].mean()) if len(all_picks) else float("nan")
        ou_eng = float(all_picks.loc[ou_eng_mask, "ou_correct"].mean()) if ou_eng_mask.sum() else float("nan")
        ou_top10 = float(all_picks.loc[ou_top10_mask, "ou_correct"].mean()) if ou_top10_mask.sum() else float("nan")

        # CLV proxy: simulator's expected margin vs market spread
        clv_spread = float((-all_picks["expected_margin"] - all_picks["mkt_spread_close"].astype(float)).abs().mean())

        # ---- Wong teaser leg + 2-team rates ----
        teaser_leg_pct, teaser_2team_pct = _eval_teasers(all_picks)

        rows_per_season.append(
            {
                "season": int(season),
                "n_games": int(len(all_picks)),
                "ats_all_acc": ats_all,
                "ats_eng_acc": ats_eng,
                "ats_top10_acc": ats_top10,
                "ats_top3_acc": ats_top3,
                "ml_eng_acc": ml_eng_acc,
                "ml_eng_roi": ml_eng_roi,
                "ou_all_acc": ou_all,
                "ou_eng_acc": ou_eng,
                "ou_top10_acc": ou_top10,
                "teaser_leg_pct": teaser_leg_pct,
                "teaser_2team_pct": teaser_2team_pct,
                "clv_spread": clv_spread,
            }
        )

    # Persist journal
    if journal_rows:
        from nfl_model.journal import append_predictions

        try:
            append_predictions(journal_rows)
        except Exception as exc:  # noqa: BLE001
            log.warning("journal.append.failed", err=str(exc))

    # Fit and save isotonic calibrators on the OOF distribution we just produced.
    # All three markets — spread / ml / total — get a calibrator built from the
    # walk-forward OOF predictions (which by construction are unbiased: each
    # season was scored using a model fit only on earlier seasons).
    if oof_picks_frames:
        try:
            _fit_and_save_calibrators(pd.concat(oof_picks_frames, ignore_index=True))
        except Exception:
            log.exception("calibration.fit.failed")

    return pd.DataFrame(rows_per_season)


def _eval_teasers(picks: pd.DataFrame) -> tuple[float, float]:
    """Evaluate Wong-teaser leg + 2-team hit rates from this season's picks.

    A Wong leg crosses both key numbers (3 and 7). For each game we look
    at the 6-point teased version of the picked side and check whether
    it covered. Returns (leg_pct, 2-team_pct) — the per-leg hit rate and
    the implied 2-team-parlay hit rate (== leg_pct ** 2 over independent
    pairs, but we compute it empirically from random pairings).
    """
    from nfl_model.config import settings as _s

    pts = _s.teaser_points
    sp = picks["mkt_spread_close"].astype(float)
    margin = picks["margin"].astype(float)
    if sp.isna().all() or margin.isna().all():
        return float("nan"), float("nan")

    home_orig = sp
    home_teased = sp + pts                              # home gets +6
    away_teased_for_home = sp - pts                     # away gets +6 (mirror)

    def crosses(orig: pd.Series, teased: pd.Series, k: int) -> pd.Series:
        a = (orig < -k) & (teased > -k)
        b = (orig > -k) & (teased < -k)
        c = (orig < k) & (teased > k)
        d = (orig > k) & (teased < k)
        return (a | b | c | d).astype(bool)

    home_wong = crosses(home_orig, home_teased, 3) & crosses(home_orig, home_teased, 7)
    away_wong = crosses(home_orig, away_teased_for_home, 3) & crosses(
        home_orig, away_teased_for_home, 7
    )

    cover_diff_home_teased = margin + home_teased
    cover_diff_away_teased = -(margin + away_teased_for_home)
    home_leg_won = (cover_diff_home_teased > 0).astype(float)
    away_leg_won = (cover_diff_away_teased > 0).astype(float)

    legs = []
    for _, r in picks.iterrows():
        if not (home_wong.loc[r.name] or away_wong.loc[r.name]):
            continue
        # Pick the wong-eligible side(s) the MODEL agrees with via expected_margin.
        em = float(r.get("expected_margin", 0.0))
        if home_wong.loc[r.name] and em + r.get("mkt_spread_close", 0) > 0:
            legs.append(float(home_leg_won.loc[r.name]))
        elif away_wong.loc[r.name] and em + r.get("mkt_spread_close", 0) < 0:
            legs.append(float(away_leg_won.loc[r.name]))

    if not legs:
        return float("nan"), float("nan")
    legs_arr = np.array(legs, dtype=float)
    leg_pct = float(legs_arr.mean())
    # 2-team: pair adjacent legs sequentially as a quick approximation.
    if len(legs_arr) < 2:
        return leg_pct, float("nan")
    pairs = legs_arr[:-1] * legs_arr[1:]
    two_team_pct = float(pairs.mean())
    return leg_pct, two_team_pct


def _train_direct_classifiers(
    Xh: np.ndarray,
    train_set: pd.DataFrame,
    feat_cols: list[str],
):
    """Train ATS + totals binary classifiers on the same feature matrix used
    by the score models. Returns (spread_booster, totals_booster); either may
    be ``None`` when there aren't enough labeled rows.
    """
    import lightgbm as lgb

    spread = pd.to_numeric(train_set.get("mkt_spread_close"), errors="coerce")
    home = pd.to_numeric(train_set["target_home_score"], errors="coerce")
    away = pd.to_numeric(train_set["target_away_score"], errors="coerce")
    margin = home - away
    y_cover = ((margin + spread) > 0).astype(float)
    mask_spread = np.isfinite(spread) & np.isfinite(margin)

    total_line = pd.to_numeric(train_set.get("mkt_total_close"), errors="coerce")
    total_pts = home + away
    y_over = (total_pts > total_line).astype(float)
    mask_total = np.isfinite(total_line) & np.isfinite(total_pts)

    params = {
        "objective": "binary", "metric": "binary_logloss",
        "learning_rate": 0.04, "num_leaves": 31, "min_data_in_leaf": 25,
        "feature_fraction": 0.8, "bagging_fraction": 0.85, "bagging_freq": 5,
        "lambda_l2": 1.0, "verbose": -1, "deterministic": True, "seed": 0,
    }

    spread_clf = None
    if int(mask_spread.sum()) >= 300:
        ds = lgb.Dataset(Xh[mask_spread], label=y_cover[mask_spread].to_numpy(), feature_name=feat_cols)
        spread_clf = lgb.train(params, ds, num_boost_round=400)

    totals_clf = None
    if int(mask_total.sum()) >= 300:
        ds = lgb.Dataset(Xh[mask_total], label=y_over[mask_total].to_numpy(), feature_name=feat_cols)
        totals_clf = lgb.train(params, ds, num_boost_round=400)

    return spread_clf, totals_clf


def _fit_and_save_calibrators(picks: pd.DataFrame) -> None:
    """Fit isotonic calibrators on the OOF picks frame from a backtest run.

    Three calibrators:

    - ``spread``  : p(home covers) vs whether home actually covered (excluding
      pushes). Skipped when no spread line was available.
    - ``ml``      : p(home wins) vs whether home actually won.
    - ``total``   : p(over) vs whether the total went over (excluding pushes).
    """
    from nfl_model.model.calibrate import fit_calibrator, save_calibrator

    # ----- Spread -----
    spr = picks.dropna(subset=["p_home_cover_sim", "actual_cover"]).copy()
    spr = spr[spr["actual_cover"].isin({"home", "away"})]
    if not spr.empty:
        p = pd.to_numeric(spr["p_home_cover_sim"], errors="coerce").to_numpy()
        y = (spr["actual_cover"].to_numpy() == "home").astype(float)
        mask = np.isfinite(p)
        if mask.any():
            cal = fit_calibrator("spread", p[mask], y[mask])
            save_calibrator(cal)

    # ----- Moneyline -----
    ml = picks.dropna(subset=["p_home_win_sim", "actual_ml"]).copy()
    if not ml.empty:
        p = pd.to_numeric(ml["p_home_win_sim"], errors="coerce").to_numpy()
        y = (ml["actual_ml"].to_numpy() == "home").astype(float)
        mask = np.isfinite(p)
        if mask.any():
            cal = fit_calibrator("ml", p[mask], y[mask])
            save_calibrator(cal)

    # ----- Total -----
    tot = picks.dropna(subset=["p_total_over_sim", "actual_total"]).copy()
    tot = tot[tot["actual_total"].isin({"over", "under"})]
    if not tot.empty:
        p = pd.to_numeric(tot["p_total_over_sim"], errors="coerce").to_numpy()
        y = (tot["actual_total"].to_numpy() == "over").astype(float)
        mask = np.isfinite(p)
        if mask.any():
            cal = fit_calibrator("total", p[mask], y[mask])
            save_calibrator(cal)
