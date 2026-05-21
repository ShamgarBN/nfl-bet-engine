"""Append-only prediction journal + auto-grading + slice metrics.

Every time the model produces predictions for a slate, a row per
(game_id x market) is appended to ``data/journal/predictions.parquet``.
This survives morning-sync invalidations and model retrains, so the
season page can show an honest record across the whole year.

Auto-grading joins finalized scores from the warehouse. Pushes (NFL has
real spreads of integers like -3, -7, -10 that frequently land exactly
on the number) are tracked separately from wins/losses.

The :func:`calibration_bins` / :func:`rolling_accuracy` / :func:`slice_breakdown`
helpers feed the /season page's charts and tier-by-tier table.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date as date_cls, datetime, timezone
from typing import Any

import numpy as np
import pandas as pd

from nfl_model.config import settings
from nfl_model.logging import get_logger

log = get_logger("journal")

JOURNAL_PATH = settings.journal_dir / "predictions.parquet"

# ROI assuming a flat -110 vig, used as the default when the journal
# row didn't carry an explicit American price.
_ROI_BY_OUTCOME = {"win": 0.909, "loss": -1.0, "push": 0.0}


@dataclass(frozen=True)
class SeasonMarketSummary:
    """Per-season per-market summary used by /season page and CLI."""

    season: int
    market: str
    n: int                  # graded picks (win + loss + push)
    pending: int            # picks awaiting finals
    wins: int
    losses: int
    pushes: int
    win_rate: float | None  # wins / (wins + losses); pushes excluded
    roi_units: float        # cumulative profit/loss across all picks
    brier: float            # mean squared error vs realized outcome
    log_loss: float         # mean log-loss vs realized outcome
    # CLV proxy. We don't track our actual ticket entry price, so the
    # next best signal is the model-vs-market edge at posting time.
    # ``mean_edge_pp`` is averaged over all graded rows with a known
    # market_prob; ``clv_n`` is the sample size that average is built from.
    mean_edge_pp: float
    clv_n: int

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def append_predictions(rows: list[dict[str, Any]]) -> int:
    """Append a batch of prediction rows to the journal.

    Each row should at minimum carry: ``game_id``, ``season``, ``week``,
    ``game_date``, ``home_team``, ``away_team``, ``market`` (spread/ml/total),
    ``pick``, ``prob``, ``edge``, ``market_line``, ``market_price``, ``tier``.
    Any of ``model_prob`` / ``market_prob`` / ``edge_pp`` / ``recorded_at``
    that aren't passed in are derived from the basics so the metrics
    helpers below have a stable schema.
    """
    if not rows:
        return 0
    settings.journal_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    now = _now_utc_iso()
    df["written_at"] = now
    if "recorded_at" not in df.columns:
        df["recorded_at"] = now

    # Backfill the richer columns that the /season metrics expect.
    if "model_prob" not in df.columns and "prob" in df.columns:
        df["model_prob"] = df["prob"]
    if "edge_pp" not in df.columns and "edge" in df.columns:
        df["edge_pp"] = pd.to_numeric(df["edge"], errors="coerce") * 100.0

    if JOURNAL_PATH.exists():
        prior = pd.read_parquet(JOURNAL_PATH)
        df = pd.concat([prior, df], ignore_index=True)
    df.to_parquet(JOURNAL_PATH, index=False)
    log.info("journal.append", rows=len(rows), path=str(JOURNAL_PATH))
    return len(df)


def read_journal() -> pd.DataFrame:
    """Return the full journal as a DataFrame (empty if no journal)."""
    if not JOURNAL_PATH.exists():
        return pd.DataFrame()
    return pd.read_parquet(JOURNAL_PATH)


def grade_journal(
    *,
    journal: pd.DataFrame | None = None,
    season: int | None = None,
    through: date_cls | None = None,
) -> pd.DataFrame:
    """Load the journal, keep the *latest* snapshot per (game, market, pick),
    and join finalized game outcomes.

    Returns:
        DataFrame with grading columns (``outcome``, ``push``, ``units``,
        ``status``). Empty if there is no journal at all. Pending rows
        carry ``outcome == 'pending'`` so the /season page can split
        graded vs pending counts.
    """
    if journal is None:
        if not JOURNAL_PATH.exists():
            return pd.DataFrame()
        journal = pd.read_parquet(JOURNAL_PATH)
    if journal is None or journal.empty:
        return pd.DataFrame()

    df = journal.copy()
    df["game_date"] = pd.to_datetime(df["game_date"]).dt.date
    if season is not None:
        df = df[df["season"] == int(season)]
    if through is not None:
        df = df[df["game_date"] <= through]
    if df.empty:
        return df

    # Keep only the latest snapshot per (game, market, pick) — the
    # model's final word before kickoff.
    if "recorded_at" in df.columns:
        df["recorded_at"] = pd.to_datetime(df["recorded_at"], errors="coerce", utc=True)
        df = df.sort_values("recorded_at").drop_duplicates(
            subset=["game_id", "market", "pick"], keep="last",
        )

    from nfl_model.data.warehouse import query

    games = query(
        "SELECT game_id, home_score, away_score, home_win FROM games"
    )
    if games.empty:
        df["outcome"] = "pending"
        df["push"] = False
        df["units"] = float("nan")
        df["status"] = "pending"
        return df

    merged = df.merge(games, on="game_id", how="left")
    merged["status"] = np.where(merged["home_score"].notna(), "Final", "pending")
    merged["margin"] = pd.to_numeric(merged["home_score"], errors="coerce") - pd.to_numeric(
        merged["away_score"], errors="coerce"
    )
    merged["total_pts"] = pd.to_numeric(merged["home_score"], errors="coerce") + pd.to_numeric(
        merged["away_score"], errors="coerce"
    )

    grades = merged.apply(_grade_row, axis=1)
    merged[["outcome", "push", "units"]] = pd.DataFrame(list(grades), index=merged.index)
    return merged


def _grade_row(row: pd.Series) -> dict[str, Any]:
    """Return outcome / push / units for a single journal row."""
    if row.get("status") != "Final":
        return {"outcome": "pending", "push": False, "units": float("nan")}

    market = str(row.get("market", "")).lower()
    pick = str(row.get("pick", "")).lower()
    line = row.get("market_line", float("nan"))
    price = row.get("market_price")
    margin = row.get("margin", float("nan"))
    total_pts = row.get("total_pts", float("nan"))
    home_win = bool(row.get("home_win", False))

    outcome = "loss"
    push = False

    if market == "ml":
        won = home_win if pick == "home" else (not home_win)
        outcome = "win" if won else "loss"
    elif market == "spread":
        if pd.isna(line) or pd.isna(margin):
            return {"outcome": "pending", "push": False, "units": float("nan")}
        # spread is home-perspective; home covers when margin + line > 0
        if pick == "home":
            adjusted = margin + line
        else:
            adjusted = -margin - line
        if adjusted == 0:
            outcome = "push"
            push = True
        elif adjusted > 0:
            outcome = "win"
        else:
            outcome = "loss"
    elif market == "total":
        if pd.isna(line) or pd.isna(total_pts):
            return {"outcome": "pending", "push": False, "units": float("nan")}
        if total_pts == line:
            outcome = "push"
            push = True
        elif (total_pts > line and pick == "over") or (total_pts < line and pick == "under"):
            outcome = "win"
        else:
            outcome = "loss"
    elif market == "teaser":
        # Teaser legs reuse the spread grading against the teased line
        # (already encoded in market_line).
        if pd.isna(line) or pd.isna(margin):
            return {"outcome": "pending", "push": False, "units": float("nan")}
        if pick == "home":
            adjusted = margin + line
        else:
            adjusted = -margin - line
        outcome = "win" if adjusted > 0 else ("push" if adjusted == 0 else "loss")
        push = adjusted == 0

    if outcome == "push":
        units = 0.0
    elif pd.notna(price):
        try:
            p = float(price)
        except (TypeError, ValueError):
            p = -110.0
        if outcome == "win":
            units = p / 100.0 if p >= 100 else 100.0 / abs(p)
        else:
            units = -1.0
    else:
        units = _ROI_BY_OUTCOME.get(outcome, float("nan"))

    return {"outcome": outcome, "push": push, "units": units}


def season_summary(graded: pd.DataFrame) -> list[SeasonMarketSummary]:
    """Per-season per-market summaries from a graded journal."""
    if graded is None or graded.empty:
        return []

    out: list[SeasonMarketSummary] = []
    for (season, market), grp in graded.groupby(["season", "market"], dropna=True):
        wins = int((grp["outcome"] == "win").sum())
        losses = int((grp["outcome"] == "loss").sum())
        pushes = int((grp["outcome"] == "push").sum())
        pending = int((grp["outcome"] == "pending").sum())
        decided = wins + losses
        win_rate = (wins / decided) if decided > 0 else None
        roi_units = float(pd.to_numeric(grp["units"], errors="coerce").fillna(0.0).sum())

        finalized = grp[grp["outcome"].isin({"win", "loss"})]
        if not finalized.empty and "model_prob" in finalized.columns:
            y = (finalized["outcome"] == "win").astype(float).to_numpy()
            p = pd.to_numeric(finalized["model_prob"], errors="coerce").fillna(0.5).to_numpy()
            p = np.clip(p, 1e-6, 1 - 1e-6)
            brier = float(np.mean((p - y) ** 2))
            log_loss = float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))
        else:
            brier = float("nan")
            log_loss = float("nan")

        if "edge_pp" in finalized.columns:
            edge_series = pd.to_numeric(finalized["edge_pp"], errors="coerce").dropna()
        else:
            edge_series = pd.Series([], dtype=float)
        if not edge_series.empty:
            mean_edge_pp = float(edge_series.mean())
            clv_n = int(edge_series.shape[0])
        else:
            mean_edge_pp = float("nan")
            clv_n = 0

        out.append(
            SeasonMarketSummary(
                season=int(season), market=str(market),
                n=int(wins + losses + pushes), pending=pending,
                wins=wins, losses=losses, pushes=pushes,
                win_rate=win_rate, roi_units=roi_units,
                brier=brier, log_loss=log_loss,
                mean_edge_pp=mean_edge_pp, clv_n=clv_n,
            )
        )
    out.sort(key=lambda s: (s.season, s.market))
    return out


def rolling_accuracy(
    graded: pd.DataFrame,
    *,
    window_days: int = 30,
    market: str | None = None,
) -> pd.DataFrame:
    """Daily series of rolling win rate over the last ``window_days``.

    Returns DataFrame with columns ``[game_date, n, win_rate, roi]``.
    """
    cols = ["game_date", "n", "win_rate", "roi"]
    if graded is None or graded.empty:
        return pd.DataFrame(columns=cols)

    sub = graded
    if market is not None:
        sub = sub[sub["market"] == market]
    sub = sub[sub["outcome"].isin({"win", "loss", "push"})].copy()
    if sub.empty:
        return pd.DataFrame(columns=cols)

    sub["game_date"] = pd.to_datetime(sub["game_date"])
    sub["is_win"] = (sub["outcome"] == "win").astype(float)
    sub["is_loss"] = (sub["outcome"] == "loss").astype(float)
    sub["is_push"] = (sub["outcome"] == "push").astype(float)
    sub["roi_value"] = pd.to_numeric(sub["units"], errors="coerce").fillna(0.0)

    daily = (
        sub.groupby(sub["game_date"].dt.date)
        .agg(wins=("is_win", "sum"), losses=("is_loss", "sum"),
             pushes=("is_push", "sum"), roi=("roi_value", "sum"))
        .reset_index()
    )
    daily["game_date"] = pd.to_datetime(daily["game_date"])
    daily = daily.sort_values("game_date").set_index("game_date")

    rolled = daily.rolling(window=f"{window_days}D").sum()
    denom = (rolled["wins"] + rolled["losses"]).replace(0, np.nan)
    rolled["win_rate"] = rolled["wins"] / denom
    rolled["n"] = rolled["wins"] + rolled["losses"] + rolled["pushes"]
    rolled = rolled.reset_index()
    rolled["game_date"] = rolled["game_date"].dt.date
    return rolled[cols]


def calibration_bins(
    graded: pd.DataFrame,
    *,
    n_bins: int = 10,
    market: str | None = None,
) -> pd.DataFrame:
    """Forecast vs realized win rate, bucketed by model probability.

    A well-calibrated model has ``forecast_mean ≈ realized_rate`` across
    every bucket.
    """
    cols = ["bucket_lo", "bucket_hi", "mid", "n", "forecast_mean", "realized_rate", "diff_pp"]
    if graded is None or graded.empty:
        return pd.DataFrame(columns=cols)

    sub = graded
    if market is not None:
        sub = sub[sub["market"] == market]
    sub = sub[sub["outcome"].isin({"win", "loss"})].copy()
    if sub.empty or "model_prob" not in sub.columns:
        return pd.DataFrame(columns=cols)

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    sub["bucket"] = pd.cut(
        pd.to_numeric(sub["model_prob"], errors="coerce").astype(float),
        bins=edges, include_lowest=True,
    )
    sub["is_win"] = (sub["outcome"] == "win").astype(float)

    g = (
        sub.groupby("bucket", observed=True)
        .agg(n=("is_win", "size"),
             forecast_mean=("model_prob", "mean"),
             realized_rate=("is_win", "mean"))
        .reset_index()
    )
    intervals = list(g["bucket"])
    g["bucket_lo"] = [float(b.left) for b in intervals]
    g["bucket_hi"] = [float(b.right) for b in intervals]
    g["mid"] = (np.asarray(g["bucket_lo"]) + np.asarray(g["bucket_hi"])) / 2.0
    g["diff_pp"] = (g["forecast_mean"].astype(float) - g["realized_rate"].astype(float)) * 100
    return g[cols]


def slice_breakdown(graded: pd.DataFrame, *, by: str = "tier") -> pd.DataFrame:
    """Win rate by an arbitrary categorical slice (tier, market, team, ...)."""
    cols = [by, "n", "win_rate", "roi", "wins", "losses", "pushes"]
    if graded is None or graded.empty or by not in graded.columns:
        return pd.DataFrame(columns=cols)

    sub = graded[graded["outcome"].isin({"win", "loss", "push"})].copy()
    if sub.empty:
        return pd.DataFrame(columns=cols)

    sub["is_win"] = (sub["outcome"] == "win").astype(float)
    sub["is_loss"] = (sub["outcome"] == "loss").astype(float)
    sub["is_push"] = (sub["outcome"] == "push").astype(float)
    sub["roi_value"] = pd.to_numeric(sub["units"], errors="coerce").fillna(0.0)

    g = (
        sub.groupby(by, dropna=False)
        .agg(wins=("is_win", "sum"),
             losses=("is_loss", "sum"),
             pushes=("is_push", "sum"),
             roi=("roi_value", "sum"))
        .reset_index()
    )
    denom = g["wins"] + g["losses"]
    g["win_rate"] = np.where(denom > 0, g["wins"] / denom.replace(0, np.nan), np.nan)
    g["n"] = g["wins"] + g["losses"] + g["pushes"]
    return g.sort_values("n", ascending=False)[cols]
