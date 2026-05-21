"""Weekly self-improvement: retrain models on the freshest NFL data.

Once a week we:

1. Pull any completed games since the last training run.
2. Recompute features and re-fit Stage A (score models) + Stage B
   classifiers (spread / total) + isotonic calibrators.
3. Archive previous ``models/*.joblib`` to ``models/archive/<ts>/`` so
   we can roll back if the new model regresses.
4. Run a small validation pass against the past 21 days and refuse to
   promote the new model if its spread accuracy drops by more than
   2 percentage points.
5. Write ``data/cache/last_weekly_train.txt`` so the desktop app can
   tell when retraining last completed.
6. If the regular season just wrapped, also run the end-of-season sweep.

The CLI ``nfl-model weekly-train`` invokes this; the desktop app runs
``needs_run`` lazily on launch (Tuesday only) so the user doesn't have
to install a cron job.
"""

from __future__ import annotations

import shutil
from datetime import date, datetime
from pathlib import Path
from typing import Any

from nfl_model.config import settings
from nfl_model.logging import get_logger

log = get_logger("automation.weekly_train")

MARKER_PATH = settings.cache_dir / "last_weekly_train.txt"
ARCHIVE_ROOT = settings.model_dir / "archive"
KEEP_ARCHIVES = 5
ACCURACY_REGRESSION_TOLERANCE_PP = 2.0
TRAINING_WINDOW_YEARS = 6
# NFL: most regular-season finals settle on Sun/Mon. We retrain Tuesday.
RETRAIN_WEEKDAY = 1  # 0=Mon, 1=Tue, ... 6=Sun


def last_run_date() -> date | None:
    if not MARKER_PATH.exists():
        return None
    try:
        return datetime.fromisoformat(MARKER_PATH.read_text(encoding="utf-8").strip()).date()
    except (ValueError, OSError):
        return None


def needs_run(today: date | None = None, *, min_days: int = 7) -> bool:
    """True on Tuesdays when last train was more than ``min_days`` ago."""
    today = today or date.today()
    if today.weekday() != RETRAIN_WEEKDAY:
        return False
    last = last_run_date()
    return last is None or (today - last).days >= min_days


def _record_success() -> None:
    MARKER_PATH.parent.mkdir(parents=True, exist_ok=True)
    MARKER_PATH.write_text(datetime.now().isoformat(timespec="seconds"), encoding="utf-8")


def _archive_current_models() -> Path | None:
    """Move existing trained artefacts into a timestamped archive folder."""
    if not settings.model_dir.exists():
        return None
    artefacts = list(settings.model_dir.glob("*.joblib"))
    if not artefacts:
        return None

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dest = ARCHIVE_ROOT / stamp
    dest.mkdir(parents=True, exist_ok=True)
    for p in artefacts:
        shutil.copy2(p, dest / p.name)
    log.info("weekly_train.archive.created", path=str(dest), files=len(artefacts))

    existing = sorted(
        [p for p in ARCHIVE_ROOT.iterdir() if p.is_dir()],
        key=lambda p: p.name,
        reverse=True,
    )
    for stale in existing[KEEP_ARCHIVES:]:
        shutil.rmtree(stale, ignore_errors=True)
        log.info("weekly_train.archive.pruned", path=str(stale))
    return dest


def _restore_from_archive(archive_dir: Path) -> None:
    for p in archive_dir.glob("*.joblib"):
        shutil.copy2(p, settings.model_dir / p.name)
    log.warning("weekly_train.archive.restored", path=str(archive_dir))


def _recent_spread_accuracy(weeks: int = 3) -> float | None:
    """Score the *current* models against the last ``weeks`` of finalized games.

    Returns spread accuracy on engine picks, or None if we don't have
    enough finalized games to evaluate.
    """
    from nfl_model.data.warehouse import query

    df = query(
        """
        SELECT season, week
        FROM games WHERE home_score IS NOT NULL
        GROUP BY season, week ORDER BY season DESC, week DESC LIMIT ?
        """,
        (weeks,),
    )
    if df.empty:
        return None

    hits = 0
    total = 0
    for _, row in df.iterrows():
        season, week = int(row["season"]), int(row["week"])
        try:
            from nfl_model.predict.weekly import predict_for_week

            preds = predict_for_week(f"{season}-{week}", refresh_data=False)
        except Exception:  # noqa: BLE001
            continue
        if preds.empty:
            continue
        outcomes = query(
            "SELECT game_id, home_score, away_score FROM games "
            "WHERE season = ? AND week = ? AND home_score IS NOT NULL",
            (season, week),
        )
        if outcomes.empty:
            continue
        merged = preds.merge(outcomes, on="game_id", how="inner")
        if merged.empty or "spread_close" not in merged.columns:
            continue
        merged["margin"] = merged["home_score"] - merged["away_score"]
        merged["covered"] = merged["margin"] + merged["spread_close"] > 0
        merged["pick_home"] = merged["p_home_cover"] >= 0.5
        merged["correct"] = merged["pick_home"] == merged["covered"]
        hits += int(merged["correct"].sum())
        total += int(merged.shape[0])
    if total < 30:
        return None
    return hits / total


def run_weekly_train(through_season: int | None = None) -> dict[str, Any]:
    """Refit models if we have new data; refuse to promote if it regresses."""
    today = date.today()
    through_season = through_season or (
        today.year - 1 if today.month <= 2 else today.year
    )

    log.info("weekly_train.start", through_season=through_season)

    baseline_acc = _recent_spread_accuracy(weeks=3)
    log.info("weekly_train.baseline", recent_ats=baseline_acc)

    archive = _archive_current_models()

    import subprocess
    import sys

    train_window_start = max(2014, through_season - TRAINING_WINDOW_YEARS)
    try:
        subprocess.run(
            [
                sys.executable, "-m", "nfl_model",
                "train",
                "--through-season", str(through_season),
                "--train-start", str(train_window_start),
            ],
            check=True,
        )
    except subprocess.CalledProcessError:
        log.exception("weekly_train.refit.failed")
        if archive is not None:
            _restore_from_archive(archive)
        raise

    new_acc = _recent_spread_accuracy(weeks=3)
    log.info("weekly_train.validation", new_ats=new_acc, baseline=baseline_acc)

    if (
        baseline_acc is not None and new_acc is not None
        and (baseline_acc - new_acc) * 100.0 > ACCURACY_REGRESSION_TOLERANCE_PP
    ):
        log.warning(
            "weekly_train.regression.detected",
            baseline=baseline_acc, new=new_acc,
            drop_pp=(baseline_acc - new_acc) * 100.0,
        )
        if archive is not None:
            _restore_from_archive(archive)
        return {
            "promoted": False,
            "reason": "regression",
            "baseline_ats": baseline_acc,
            "new_ats": new_acc,
        }

    # Invalidate cached pick CSVs so the dashboard re-runs with the new model.
    for f in settings.cache_dir.glob("picks_*.csv"):
        f.unlink(missing_ok=True)

    _record_success()
    log.info("weekly_train.done")

    end_of_season_result: dict[str, Any] | None = None
    try:
        end_of_season_result = _maybe_trigger_end_of_season(through_season)
    except Exception:  # noqa: BLE001
        log.exception("weekly_train.end_of_season.failed")

    return {
        "promoted": True,
        "baseline_ats": baseline_acc,
        "new_ats": new_acc,
        "archive": str(archive) if archive else None,
        "end_of_season": end_of_season_result,
    }


def _maybe_trigger_end_of_season(through_season: int) -> dict[str, Any] | None:
    """Run end-of-season sweep if appropriate (and not already complete)."""
    from nfl_model.season import is_regular_season_over, run_end_of_season_sweep

    if not is_regular_season_over(through_season):
        return None

    report_dir = settings.project_root / "reports" / f"end_of_season_{through_season}"
    marker = report_dir / "summary.json"
    if marker.exists():
        log.info(
            "weekly_train.end_of_season.already_done",
            season=through_season, marker=str(marker),
        )
        return {"skipped": True, "reason": "already_complete", "season": through_season}

    log.info("weekly_train.end_of_season.start", season=through_season)
    report = run_end_of_season_sweep(season=through_season)
    log.info(
        "weekly_train.end_of_season.done",
        season=through_season, output_dir=report.output_dir,
    )
    return {
        "season": through_season,
        "output_dir": report.output_dir,
        "report_md": report.report_md_path,
    }
