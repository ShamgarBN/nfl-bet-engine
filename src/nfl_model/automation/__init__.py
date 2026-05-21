"""Background sync + retrain jobs."""

from __future__ import annotations

from nfl_model.automation.morning_sync import (
    needs_run as morning_needs_run,
    run_morning_sync,
)
from nfl_model.automation.scheduler import install as install_scheduler
from nfl_model.automation.scheduler import uninstall as uninstall_scheduler
from nfl_model.automation.weekly_train import (
    needs_run as weekly_needs_run,
    run_weekly_train,
)

__all__ = [
    "install_scheduler",
    "morning_needs_run",
    "run_morning_sync",
    "run_weekly_train",
    "uninstall_scheduler",
    "weekly_needs_run",
]
