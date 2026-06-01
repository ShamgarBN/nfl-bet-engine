"""Live weekly prediction pipeline.

Used by ``nfl-model predict`` and the desktop app's ``/`` page. Pulls the
upcoming slate, refreshes injuries / weather / lines, runs the production
models + simulator + ensemble + calibrators, and writes a per-week picks
CSV plus journal rows.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from nfl_model.config import settings
from nfl_model.data.warehouse import query
from nfl_model.logging import get_logger

log = get_logger("predict.weekly")


def _resolve_week(week_arg: str) -> tuple[int, int]:
    """Resolve 'current' or 'YYYY-W' into (season, week)."""
    if "-" in week_arg:
        s, w = week_arg.split("-")
        return int(s), int(w)
    today = date.today()
    season = today.year - 1 if today.month <= 2 else today.year
    df = query(
        "SELECT week, MIN(game_date) AS d FROM games "
        "WHERE season = ? AND home_score IS NULL GROUP BY week ORDER BY d",
        (season,),
    )
    if df.empty:
        # Fall back to most-recent finalized week if season not started yet
        df2 = query(
            "SELECT MAX(season) AS s, MAX(week) AS w FROM games WHERE home_score IS NOT NULL"
        )
        if df2.empty or pd.isna(df2.iloc[0]["s"]) or pd.isna(df2.iloc[0]["w"]):
            return season, 1
        return int(df2.iloc[0]["s"]), int(df2.iloc[0]["w"])
    return season, int(df.iloc[0]["week"])


def predict_for_week(week_arg: str, *, refresh_data: bool = False) -> pd.DataFrame:
    """Predict the upcoming slate for ``week_arg``.

    Returns a wide DataFrame with one row per game and calibrated
    probabilities + edge per market. Rows are also appended to the
    prediction journal so the season page can grade them later.
    """
    from nfl_model.features.assemble import build_features_table
    from nfl_model.journal import append_predictions
    from nfl_model.model.calibrate import load_calibrator
    from nfl_model.model.ensemble import blend
    from nfl_model.model.feature_matrix import (
        build_runs_matrix,
        build_runs_matrix_away,
    )
    from nfl_model.model.scores import load_score_model
    from nfl_model.model.simulate import simulate_games
    from nfl_model.model.spread_direct import load_spread_classifier
    from nfl_model.model.totals import load_totals_classifier

    season, week = _resolve_week(week_arg)
    log.info("predict.week", season=season, week=week)

    # Pull schedule + market lines for the slate
    slate = query(
        """
        SELECT g.game_id, g.season, g.week, g.game_date, g.kickoff_ts,
               g.home_team, g.away_team, g.home_qb, g.away_qb,
               g.roof, g.surface, g.weekday,
               o.spread_close, o.total_close,
               o.ml_close_home, o.ml_close_away
        FROM games g
        LEFT JOIN odds_history o ON o.game_id = g.game_id AND o.book = 'consensus'
        WHERE g.season = ? AND g.week = ?
        ORDER BY g.game_date, g.kickoff_ts
        """,
        (season, week),
    )
    if slate.empty:
        return pd.DataFrame()

    spec_path = settings.model_dir / "feature_spec.joblib"
    if not spec_path.exists():
        log.warning("predict.no_models", reason="train models first via `nfl-model train`")
        # Surface the slate with placeholder probs so the UI has something
        out = slate.copy()
        out["p_home_win"] = 0.5
        out["p_home_cover"] = 0.5
        out["p_total_over"] = 0.5
        out["edge_spread"] = 0.0
        out["edge_total"] = 0.0
        out["edge_ml"] = 0.0
        out["confidence_tier"] = "n/a"
        return out

    spec = joblib.load(spec_path)
    home_model = load_score_model("scores_home")
    away_model = load_score_model("scores_away")

    # Build features for these games (must include older seasons in the rolling
    # context because each game's features are leakage-safe rolling aggregates
    # over prior games for both teams).
    full_feats = build_features_table(season - 5, season)
    target = full_feats[(full_feats["season"] == season) & (full_feats["week"] == week)]
    if target.empty:
        return pd.DataFrame()

    Xh, _, _ = build_runs_matrix(target, spec)
    Xa, _, _ = build_runs_matrix_away(target, spec)
    hm, hs = home_model.predict_distribution(Xh)
    am, ase = away_model.predict_distribution(Xa)

    spreads = pd.to_numeric(target.get("mkt_spread_close"), errors="coerce").to_numpy()
    totals = pd.to_numeric(target.get("mkt_total_close"), errors="coerce").to_numpy()
    preds = simulate_games(
        game_ids=target["game_id"].to_numpy(),
        pred_home=(hm, hs),
        pred_away=(am, ase),
        spread_lines=spreads,
        total_lines=totals,
        n_sims=settings.monte_carlo_iterations_inference,
        seed=settings.random_seed + season * 100 + week,
    )

    p_home_win = np.array([p.p_home_win for p in preds])
    p_home_cover = np.array([p.p_home_cover for p in preds])
    p_total_over = np.array([p.p_total_over for p in preds])

    # ---- Direct-classifier blend ----
    # The direct ATS / total classifiers are trained on the binary
    # cover/over labels directly; the simulator is trained on score MSE.
    # Blending the two via model.ensemble.blend gives a better-calibrated
    # final probability than either head alone.
    try:
        spread_clf = load_spread_classifier()
        p_spread_direct = spread_clf.predict_proba_home_cover(Xh)
        p_home_cover = blend("spread", p_home_cover, p_spread_direct)
    except FileNotFoundError:
        log.debug("predict.no_spread_direct", reason="train spread_direct first")
    try:
        totals_clf = load_totals_classifier()
        p_totals_direct = totals_clf.predict_proba_over(Xh)
        p_total_over = blend("total", p_total_over, p_totals_direct)
    except FileNotFoundError:
        log.debug("predict.no_totals_direct", reason="train totals first")

    # Apply calibrators if present
    cal_spread = load_calibrator("spread")
    if cal_spread is not None:
        p_home_cover = cal_spread.transform(p_home_cover)
    cal_total = load_calibrator("total")
    if cal_total is not None:
        p_total_over = cal_total.transform(p_total_over)
    cal_ml = load_calibrator("ml")
    if cal_ml is not None:
        p_home_win = cal_ml.transform(p_home_win)

    out = slate.copy().reset_index(drop=True)
    out["p_home_win"] = p_home_win
    out["p_home_cover"] = p_home_cover
    out["p_total_over"] = p_total_over
    out["expected_margin"] = [p.expected_margin for p in preds]
    out["expected_total"] = [p.expected_total for p in preds]

    # ---- Picks driven by edge vs market (not raw p_home_cover) ----
    # Spread: pick the side the model's expected_margin beats the line on,
    # size by the magnitude of the implied cover margin. Same for totals.
    # ML keeps the higher-probability side (no per-game vig adjustment yet).
    posted_sp = pd.to_numeric(out["spread_close"], errors="coerce").to_numpy()
    posted_tot = pd.to_numeric(out["total_close"], errors="coerce").to_numpy()
    exp_margin = np.array([p.expected_margin for p in preds], dtype=float)
    exp_total = np.array([p.expected_total for p in preds], dtype=float)
    exp_cover = exp_margin + posted_sp
    tot_diff = exp_total - posted_tot

    out["pick_spread"] = np.where(exp_cover > 0, "home", "away")
    out["pick_ml"] = np.where(p_home_win >= 0.5, "home", "away")
    out["pick_ou"] = np.where(tot_diff > 0, "over", "under")

    # Conviction = points of disagreement with the market.
    out["edge_spread"] = np.abs(exp_cover)
    out["edge_total"] = np.abs(tot_diff)
    out["edge_ml"] = np.abs(p_home_win - 0.5)

    # Tier by the larger of spread / total point edge (in points). We use
    # absolute thresholds (3 pts / 5 pts) rather than within-slate quantiles
    # because a 16-game NFL slate is too small for quantile cutoffs to
    # meaningfully separate "best 3% of picks". A 3-point spread or total
    # disagreement with a sharp closing line is already meaningful; >=5 is
    # the kind of edge that historically correlates with above-market hit
    # rates. The weekly app surfaces the raw edge numbers so the user can
    # apply their own threshold if they want.
    conv = np.maximum(out["edge_spread"].to_numpy(), out["edge_total"].fillna(0.0).to_numpy())
    cutoff_top10 = 3.0
    cutoff_top3 = 5.0

    def _tier(c: float) -> str:
        if not np.isfinite(c):
            return "n/a"
        if c >= cutoff_top3:
            return "top-3%"
        if c >= cutoff_top10:
            return "top-10%"
        # spread_edge_min_pct is a probability cushion; for the point-edge
        # tier we floor at 1 point (basically any meaningful disagreement).
        if c >= 1.0:
            return "engine"
        return "n/a"

    out["confidence_tier"] = [_tier(c) for c in conv]

    # Journal: write a row per (game x market) for later grading
    journal = []
    for _, r in out.iterrows():
        for market, pick_col, prob_col, line_col in (
            ("spread", "pick_spread", "p_home_cover", "spread_close"),
            ("ml", "pick_ml", "p_home_win", "ml_close_home"),
            ("total", "pick_ou", "p_total_over", "total_close"),
        ):
            pick = r[pick_col]
            base_prob = float(r[prob_col])
            prob = base_prob if pick in {"home", "over"} else 1.0 - base_prob
            journal.append({
                "game_id": r["game_id"],
                "season": int(r["season"]),
                "week": int(r["week"]),
                "game_date": str(r["game_date"]),
                "home_team": r["home_team"],
                "away_team": r["away_team"],
                "market": market,
                "pick": pick,
                "prob": prob,
                "edge": float(r[f"edge_{market.replace('total', 'total').replace('ml', 'ml').replace('spread', 'spread')}"]),
                "market_line": float(r[line_col]) if pd.notna(r[line_col]) else None,
                "market_price": -110,
                "tier": r["confidence_tier"],
            })
    if journal:
        append_predictions(journal)

    return out


def write_picks_csv(picks: pd.DataFrame, week_arg: str) -> Path:
    """Persist this week's picks for later journaling and review."""
    settings.cache_dir.mkdir(parents=True, exist_ok=True)
    safe = week_arg.replace("/", "-")
    path = settings.cache_dir / f"picks_{safe}.csv"
    picks.to_csv(path, index=False)
    return path
