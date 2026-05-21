"""Morning data refresh for the current NFL slate.

Pulls the most-recent finalized games (so the journal can be graded)
and refreshes injuries / weather / lines for the upcoming week so the
day's predictions are based on the latest available data.

Designed to be idempotent and safe to run repeatedly: every upsert
checks primary keys before writing, and ``last_morning_sync.txt`` makes
"already done today" easy to detect.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Any

from nfl_model.config import settings
from nfl_model.logging import get_logger

log = get_logger("automation.morning_sync")

MARKER_PATH = settings.cache_dir / "last_morning_sync.txt"


def last_run_date() -> date | None:
    """Return the date of the last successful morning sync, or None."""
    if not MARKER_PATH.exists():
        return None
    try:
        return datetime.fromisoformat(MARKER_PATH.read_text(encoding="utf-8").strip()).date()
    except (ValueError, OSError):
        return None


def needs_run(today: date | None = None) -> bool:
    """True if we have not synced yet today."""
    today = today or date.today()
    last = last_run_date()
    return last is None or last < today


def _record_success(when: datetime | None = None) -> None:
    when = when or datetime.now()
    MARKER_PATH.parent.mkdir(parents=True, exist_ok=True)
    MARKER_PATH.write_text(when.isoformat(timespec="seconds"), encoding="utf-8")


def _maybe_notify_premium(today: date) -> None:
    """Emit a single macOS notification if today/this week has top-3% picks."""
    import platform
    import shutil
    import subprocess

    marker = settings.cache_dir / f"notify_premium_{today.isoformat()}.flag"
    if marker.exists():
        return

    try:
        from nfl_model.predict.weekly import predict_for_week

        preds = predict_for_week("current", refresh_data=False)
    except Exception:  # noqa: BLE001
        log.warning("morning_sync.notify.predict_failed")
        return
    if preds is None or preds.empty:
        return
    if "confidence_tier" not in preds.columns:
        return
    has_prem = (preds["confidence_tier"] == "top-3%").any()
    if not has_prem:
        return

    if platform.system() != "Darwin" or shutil.which("osascript") is None:
        log.debug("morning_sync.notify.unsupported_platform")
        return

    title = "NFL Forecast — top-3% picks this week"
    body = "High-conviction picks are available. Open the app to review."
    try:
        subprocess.run(
            [
                "osascript", "-e",
                f'display notification "{body}" with title "{title}" sound name "Glass"',
            ],
            check=False,
            timeout=5,
        )
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text(datetime.now().isoformat(timespec="seconds"))
        log.info("morning_sync.notify.sent", date=today.isoformat())
    except Exception:  # noqa: BLE001
        log.exception("morning_sync.notify.osascript_failed")


def run_morning_sync(today: date | None = None) -> dict[str, Any]:
    """Refresh the current week's slate. Returns row-count summary."""
    today = today or date.today()

    log.info("morning_sync.start", today=today.isoformat())

    counts: dict[str, Any] = {"today": today.isoformat()}

    # NFL season runs Sep -> Feb. Compute the relevant season.
    season = today.year - 1 if today.month <= 2 else today.year

    # 1) Refresh schedule + finals + injuries + lines for the current season.
    try:
        from nfl_model.data.pipeline import pull_season

        counts["season_pull"] = pull_season(
            season,
            with_weather=True,
            with_lines=True,
        )
    except Exception:  # noqa: BLE001
        log.exception("morning_sync.season.failed")
        counts["season_pull"] = {"error": True}

    # 2) Invalidate cached prediction artefacts for the current week.
    cache_glob = list(settings.cache_dir.glob("picks_*.csv"))
    for f in cache_glob:
        try:
            f.unlink(missing_ok=True)
        except OSError:
            pass
    counts["prediction_cache_cleared"] = len(cache_glob)

    # 3) Best-effort macOS notification when high-conviction picks exist.
    try:
        _maybe_notify_premium(today)
    except Exception:  # noqa: BLE001
        log.exception("morning_sync.notify.failed")

    critical_failed = (
        isinstance(counts.get("season_pull"), dict) and counts["season_pull"].get("error")
    )
    if critical_failed:
        counts["status"] = "failed"
        counts["recorded"] = False
        log.warning("morning_sync.degraded", today=today.isoformat())
    else:
        _record_success()
        counts["status"] = "ok"
        counts["recorded"] = True
    log.info("morning_sync.done", **{k: v for k, v in counts.items() if not isinstance(v, dict)})
    return counts
