"""Season state / rollover predicates."""

from __future__ import annotations

from datetime import date

import pandas as pd


def test_last_completed_season_returns_none_for_empty_warehouse(
    tmp_path, monkeypatch
) -> None:
    from nfl_model import config as cfg
    from nfl_model.data import warehouse as wh
    from nfl_model.season import last_completed_season

    monkeypatch.setattr(cfg.settings, "warehouse_path", tmp_path / "warehouse.duckdb")
    wh._SCHEMA_READY = False  # noqa: SLF001
    wh.init_schema(force=True)

    assert last_completed_season(today=date(2026, 3, 1)) is None


def test_is_regular_season_over_requires_finalized_games(
    tmp_path, monkeypatch
) -> None:
    """``is_regular_season_over`` should be False until the warehouse has
    enough finalized games.
    """
    from nfl_model import config as cfg
    from nfl_model.data import warehouse as wh
    from nfl_model.season import is_regular_season_over

    monkeypatch.setattr(cfg.settings, "warehouse_path", tmp_path / "warehouse.duckdb")
    wh._SCHEMA_READY = False  # noqa: SLF001
    wh.init_schema(force=True)

    # Empty warehouse -> not over.
    assert not is_regular_season_over(2024, today=date(2026, 1, 15))

    # Insert a single game; still not enough.
    wh.upsert_dataframe(
        pd.DataFrame([{
            "game_id": "2024_01_BAL_KC", "season": 2024, "week": 1,
            "game_type": "REG", "game_date": "2024-09-05",
            "kickoff_ts": None, "weekday": "Thu",
            "home_team": "KC", "away_team": "BAL",
            "home_qb": None, "away_qb": None,
            "home_coach": None, "away_coach": None,
            "stadium_id": "KC", "stadium": "Arrowhead",
            "roof": "outdoors", "surface": "grass",
            "home_score": 27, "away_score": 24, "home_win": True,
            "overtime": False, "primetime": True, "divisional": False,
            "referee": None,
        }]),
        "games", key_columns=["game_id"],
    )
    assert not is_regular_season_over(2024, today=date(2026, 1, 15))


def test_detect_season_state_handles_empty_warehouse(tmp_path, monkeypatch) -> None:
    from nfl_model import config as cfg
    from nfl_model.data import warehouse as wh
    from nfl_model.season import detect_season_state

    monkeypatch.setattr(cfg.settings, "warehouse_path", tmp_path / "warehouse.duckdb")
    wh._SCHEMA_READY = False  # noqa: SLF001
    wh.init_schema(force=True)

    status = detect_season_state(today=date(2026, 3, 1))
    assert status.state in {"pre_season", "off_season", "in_progress"}
    assert status.finalized_games == 0
