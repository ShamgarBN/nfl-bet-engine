"""End-of-season sweep for the NFL Bet Engine.

When a season's regular season finishes, this module:

1. Verifies completeness (refuses to run on a half-pulled season).
2. Snapshots the prediction journal.
3. Runs the full walk-forward backtest through this season.
4. Grades the journal per market and per tier.
5. Identifies worst slices (where the model bled the most).
6. Writes a structured report under ``reports/end_of_season_<YEAR>/``.
7. Archives the model-of-record so we can compare next year.

Designed to be run by the weekly retrain job once the season ends -- the
human never has to remember to trigger it.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from nfl_model.config import settings
from nfl_model.journal import grade_journal, season_summary
from nfl_model.journal.core import JOURNAL_PATH
from nfl_model.logging import get_logger
from nfl_model.season.rollover import is_regular_season_over

log = get_logger("season.end_of_season")

REPORTS_ROOT: Path = settings.project_root / "reports"
_MIN_SLICE_N = 10
_TOP_K_WORST = 8


@dataclass
class EndOfSeasonReport:
    """In-memory representation of the report we just wrote to disk."""

    season: int
    completed_at: str
    output_dir: str
    journal_snapshot_path: str | None
    backtest_csv_path: str | None
    summary_json_path: str
    report_md_path: str
    slices_csv_path: str | None
    summary_by_market: list[dict[str, Any]]
    worst_slices: list[dict[str, Any]]
    recommendations: list[str]
    model_snapshot_dir: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def run_end_of_season_sweep(
    season: int,
    *,
    train_start: int | None = None,
    force: bool = False,
    skip_backtest: bool = False,
) -> EndOfSeasonReport:
    """Execute the full sweep for ``season``."""
    if not force and not is_regular_season_over(season):
        raise RuntimeError(
            f"Refusing to run end-of-season sweep for {season}: "
            f"regular season does not appear to be complete. "
            f"Pass force=True if you really mean it."
        )

    train_start = train_start or max(2014, season - 6)
    output_dir = REPORTS_ROOT / f"end_of_season_{season}"
    output_dir.mkdir(parents=True, exist_ok=True)
    log.info("end_of_season.start", season=season, output_dir=str(output_dir))

    completed_at = datetime.now().isoformat(timespec="seconds")

    journal_snapshot_path: str | None = None
    if JOURNAL_PATH.exists():
        snap = output_dir / "predictions_journal_snapshot.parquet"
        shutil.copy2(JOURNAL_PATH, snap)
        journal_snapshot_path = str(snap)
        log.info("end_of_season.journal.snapshot", path=str(snap))

    graded = grade_journal(season=season)
    summaries = [
        s.as_dict() for s in (season_summary(graded) if graded is not None else [])
    ]
    log.info("end_of_season.journal.graded", n_summaries=len(summaries))

    backtest_csv_path: str | None = None
    backtest_df: pd.DataFrame | None = None
    if not skip_backtest:
        try:
            from nfl_model.backtest.walkforward import run_walkforward

            log.info("end_of_season.backtest.start", start=train_start + 1, end=season)
            backtest_df = run_walkforward(train_start + 1, season)
            if not backtest_df.empty:
                backtest_csv_path = str(output_dir / "backtest.csv")
                backtest_df.to_csv(backtest_csv_path, index=False)
        except Exception:  # noqa: BLE001
            log.exception("end_of_season.backtest.failed")

    slices_df = _identify_worst_slices(graded)
    slices_csv_path: str | None = None
    if not slices_df.empty:
        slices_csv_path = str(output_dir / "slices.csv")
        slices_df.to_csv(slices_csv_path, index=False)

    model_snapshot_dir = _archive_model_of_record(season, output_dir)

    recommendations = _generate_recommendations(
        summaries=summaries, slices=slices_df, backtest=backtest_df,
    )

    summary_json = {
        "season": season,
        "completed_at": completed_at,
        "train_start": train_start,
        "summary_by_market": summaries,
        "worst_slices": slices_df.to_dict("records") if not slices_df.empty else [],
        "recommendations": recommendations,
        "backtest_csv": backtest_csv_path,
        "journal_snapshot": journal_snapshot_path,
        "model_snapshot_dir": str(model_snapshot_dir) if model_snapshot_dir else None,
    }
    summary_json_path = output_dir / "summary.json"
    summary_json_path.write_text(
        json.dumps(summary_json, indent=2, default=str), encoding="utf-8",
    )

    report_md_path = output_dir / "report.md"
    report_md_path.write_text(
        _render_report(
            season=season, completed_at=completed_at,
            summaries=summaries, slices=slices_df,
            backtest=backtest_df, recommendations=recommendations,
        ),
        encoding="utf-8",
    )

    report = EndOfSeasonReport(
        season=season,
        completed_at=completed_at,
        output_dir=str(output_dir),
        journal_snapshot_path=journal_snapshot_path,
        backtest_csv_path=backtest_csv_path,
        summary_json_path=str(summary_json_path),
        report_md_path=str(report_md_path),
        slices_csv_path=slices_csv_path,
        summary_by_market=summaries,
        worst_slices=slices_df.to_dict("records") if not slices_df.empty else [],
        recommendations=recommendations,
        model_snapshot_dir=str(model_snapshot_dir) if model_snapshot_dir else None,
        metadata={"train_start": train_start},
    )
    log.info(
        "end_of_season.done", season=season,
        report_md=str(report_md_path), backtest_csv=backtest_csv_path,
    )
    return report


def _identify_worst_slices(graded: pd.DataFrame | None) -> pd.DataFrame:
    """Find the slate segments where the model lost the most units."""
    if graded is None or graded.empty:
        return pd.DataFrame()

    finalized = graded[graded["outcome"].isin({"win", "loss", "push"})].copy()
    if finalized.empty:
        return pd.DataFrame()

    finalized["is_win"] = (finalized["outcome"] == "win").astype(float)
    finalized["is_loss"] = (finalized["outcome"] == "loss").astype(float)
    finalized["units"] = pd.to_numeric(finalized["units"], errors="coerce").fillna(0.0)

    dimensions = [
        ("market", "market"),
        ("tier", "tier"),
        ("home_team", "home_team"),
        ("away_team", "away_team"),
    ]

    rows: list[pd.DataFrame] = []
    for col, label in dimensions:
        if col not in finalized.columns:
            continue
        grouped = (
            finalized.dropna(subset=[col])
            .groupby(col, dropna=True)
            .agg(
                n=("is_win", "size"),
                wins=("is_win", "sum"),
                losses=("is_loss", "sum"),
                roi=("units", "sum"),
            )
            .reset_index()
        )
        grouped = grouped[grouped["n"] >= _MIN_SLICE_N]
        if grouped.empty:
            continue
        denom = (grouped["wins"] + grouped["losses"]).replace(0, pd.NA)
        grouped["win_rate"] = (grouped["wins"] / denom).astype(float)
        grouped["dimension"] = label
        grouped["value"] = grouped[col].astype(str)
        grouped = grouped.drop(columns=[col])
        rows.append(grouped[["dimension", "value", "n", "wins", "losses", "win_rate", "roi"]])

    if not rows:
        return pd.DataFrame()

    out = pd.concat(rows, ignore_index=True)
    out = out.sort_values("roi").head(_TOP_K_WORST).reset_index(drop=True)
    return out


def _archive_model_of_record(season: int, output_dir: Path) -> Path | None:
    """Copy the currently-trained model files into the report folder."""
    model_dir = settings.model_dir
    if not model_dir.exists():
        return None
    artefacts = list(model_dir.glob("*.joblib"))
    if not artefacts:
        return None

    dest = output_dir / f"model_of_record_{season}"
    dest.mkdir(parents=True, exist_ok=True)
    for p in artefacts:
        shutil.copy2(p, dest / p.name)
    log.info(
        "end_of_season.model_archive",
        season=season, dest=str(dest), n_files=len(artefacts),
    )
    return dest


def _generate_recommendations(
    *,
    summaries: list[dict[str, Any]],
    slices: pd.DataFrame,
    backtest: pd.DataFrame | None,
) -> list[str]:
    """Produce a short list of off-season investigation prompts."""
    recs: list[str] = []
    by_market: dict[str, dict[str, Any]] = {row["market"]: row for row in summaries}

    if "spread" in by_market:
        sp = by_market["spread"]
        wr = sp.get("win_rate")
        if wr is not None and wr == wr and wr < 0.520:
            recs.append(
                f"Spread picks bled at {wr * 100:.1f}% over {sp['n']} graded picks "
                "(target: 57%+ on engine bets). Re-examine market-signal features "
                "(reverse-line moves, key-number crosses) and QB-form rolling windows."
            )
        elif wr is not None and wr == wr and wr >= 0.580:
            recs.append(
                f"Spread engine hit {wr * 100:.1f}% over {sp['n']} picks -- ahead "
                "of the 57% target. Document the current feature set as a baseline "
                "before tweaking anything off-season."
            )

    if "ml" in by_market:
        ml = by_market["ml"]
        wr = ml.get("win_rate")
        if wr is not None and wr == wr and wr < 0.640:
            recs.append(
                f"Moneyline win rate of {wr * 100:.1f}% over {ml['n']} picks fell "
                "short of the 72% engine target. Check the ML calibrator and "
                "the QB-injury severity grading."
            )

    if "total" in by_market:
        ou = by_market["total"]
        wr = ou.get("win_rate")
        if wr is not None and wr == wr and wr < 0.520:
            recs.append(
                f"Totals (O/U) bled at {wr * 100:.1f}% over {ou['n']} picks. The "
                "weather-on-pass-axis and pace/PROE features likely need revisiting."
            )

    if not slices.empty:
        worst = slices.iloc[0]
        recs.append(
            f"Biggest single drag was {worst['dimension']} = {worst['value']!r} "
            f"({int(worst['n'])} picks, {worst['win_rate'] * 100:.1f}% win rate, "
            f"{worst['roi']:+.2f}u P/L). Investigate feature coverage in this segment."
        )

    if not recs:
        recs.append(
            "No structural concerns detected. Continue weekly retraining on a "
            "rolling 6-season window."
        )
    return recs


def _render_report(
    *,
    season: int,
    completed_at: str,
    summaries: list[dict[str, Any]],
    slices: pd.DataFrame,
    backtest: pd.DataFrame | None,
    recommendations: list[str],
) -> str:
    """Render the human-readable markdown report."""
    lines: list[str] = []
    lines.append(f"# {season} season post-mortem")
    lines.append("")
    lines.append(f"_Generated {completed_at}._")
    lines.append("")
    lines.append("This report grades every prediction the model emitted live during")
    lines.append("the season and surfaces where next year's iteration should focus.")
    lines.append("")

    lines.append("## Per-market performance")
    lines.append("")
    if summaries:
        lines.append("| Market | N | W-L-P | Win % | P/L (u) | Brier | Log-loss |")
        lines.append("|---|---:|:---:|---:|---:|---:|---:|")
        for row in summaries:
            wr = row.get("win_rate")
            wr_s = f"{wr * 100:.1f}%" if wr is not None and wr == wr else "—"
            brier = row.get("brier")
            brier_s = f"{brier:.3f}" if brier is not None and brier == brier else "—"
            ll = row.get("log_loss")
            ll_s = f"{ll:.3f}" if ll is not None and ll == ll else "—"
            wlp = f"{row['wins']}-{row['losses']}"
            if row.get("pushes"):
                wlp += f"-{row['pushes']}"
            lines.append(
                f"| {row['market']} | {row['n']} | {wlp} | {wr_s} | "
                f"{row['roi_units']:+.2f} | {brier_s} | {ll_s} |"
            )
    else:
        lines.append("_No graded predictions in the journal._")
    lines.append("")

    lines.append("## Worst-performing slices")
    lines.append("")
    if slices is not None and not slices.empty:
        lines.append(
            f"Sorted by P/L drag. Slices with fewer than {_MIN_SLICE_N} graded picks excluded."
        )
        lines.append("")
        lines.append("| Dimension | Value | N | Win % | P/L (u) |")
        lines.append("|---|---|---:|---:|---:|")
        for _, row in slices.iterrows():
            wr = row.get("win_rate")
            wr_s = f"{wr * 100:.1f}%" if pd.notna(wr) else "—"
            lines.append(
                f"| {row['dimension']} | {row['value']} | {int(row['n'])} | "
                f"{wr_s} | {row['roi']:+.2f} |"
            )
    else:
        lines.append(f"_Not enough data -- need at least {_MIN_SLICE_N} graded picks per slice._")
    lines.append("")

    lines.append("## Walk-forward backtest")
    lines.append("")
    if backtest is not None and not backtest.empty:
        lines.append("| Season | N | ATS eng | ATS top-10% | ATS top-3% | ML acc | O/U eng |")
        lines.append("|---:|---:|---:|---:|---:|---:|---:|")
        for _, r in backtest.iterrows():
            lines.append(
                f"| {int(r['season'])} | {int(r['n_games'])} | "
                f"{r.get('ats_eng_acc', float('nan')) * 100:.1f}% | "
                f"{r.get('ats_top10_acc', float('nan')) * 100:.1f}% | "
                f"{r.get('ats_top3_acc', float('nan')) * 100:.1f}% | "
                f"{r.get('ml_eng_acc', float('nan')) * 100:.1f}% | "
                f"{r.get('ou_eng_acc', float('nan')) * 100:.1f}% |"
            )
    else:
        lines.append("_Walk-forward backtest skipped or failed -- see logs._")
    lines.append("")

    lines.append("## Off-season recommendations")
    lines.append("")
    for i, rec in enumerate(recommendations, 1):
        lines.append(f"{i}. {rec}")
    lines.append("")
    return "\n".join(lines)
