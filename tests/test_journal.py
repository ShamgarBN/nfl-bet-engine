"""Journal: append-only, idempotent grading, push handling, market summary."""

from __future__ import annotations

import pandas as pd


def test_journal_appends_and_reads(tmp_path, monkeypatch) -> None:
    from nfl_model import config as cfg
    from nfl_model.journal import append_predictions
    from nfl_model.journal import core as journal_core
    from nfl_model.journal.core import read_journal

    monkeypatch.setattr(cfg.settings, "journal_dir", tmp_path)
    monkeypatch.setattr(journal_core, "JOURNAL_PATH", tmp_path / "predictions.parquet")

    n = append_predictions([
        {
            "game_id": "2024_01_BAL_KC",
            "season": 2024, "week": 1, "game_date": "2024-09-05",
            "home_team": "KC", "away_team": "BAL",
            "market": "spread", "pick": "home", "prob": 0.62, "edge": 0.12,
            "market_line": -3.0, "market_price": -110, "tier": "engine",
        },
        {
            "game_id": "2024_01_BAL_KC",
            "season": 2024, "week": 1, "game_date": "2024-09-05",
            "home_team": "KC", "away_team": "BAL",
            "market": "ml", "pick": "home", "prob": 0.68, "edge": 0.18,
            "market_line": None, "market_price": -200, "tier": "top-10%",
        },
    ])
    assert n == 2
    df = read_journal()
    assert len(df) == 2
    assert set(df["market"].unique()) == {"spread", "ml"}


def test_grade_journal_handles_pushes(tmp_path, monkeypatch) -> None:
    """A spread that lands exactly on the line should be graded as a push."""
    from nfl_model import config as cfg
    from nfl_model.data.warehouse import init_schema, upsert_dataframe
    from nfl_model.journal import append_predictions, grade_journal
    from nfl_model.journal import core as journal_core

    monkeypatch.setattr(cfg.settings, "warehouse_path", tmp_path / "warehouse.duckdb")
    monkeypatch.setattr(cfg.settings, "journal_dir", tmp_path)
    monkeypatch.setattr(journal_core, "JOURNAL_PATH", tmp_path / "predictions.parquet")

    from nfl_model.data import warehouse as wh

    wh._SCHEMA_READY = False  # noqa: SLF001
    init_schema(force=True)

    upsert_dataframe(
        pd.DataFrame([{
            "game_id": "2024_01_BAL_KC",
            "season": 2024, "week": 1, "game_type": "REG",
            "game_date": "2024-09-05", "kickoff_ts": None, "weekday": "Thu",
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

    append_predictions([
        {
            "game_id": "2024_01_BAL_KC",
            "season": 2024, "week": 1, "game_date": "2024-09-05",
            "home_team": "KC", "away_team": "BAL",
            "market": "spread", "pick": "home", "prob": 0.55, "edge": 0.05,
            "market_line": -3.0, "market_price": -110, "tier": "engine",
        },
    ])

    graded = grade_journal()
    assert graded is not None and not graded.empty
    spread_rows = graded[graded["market"] == "spread"]
    assert (spread_rows["outcome"] == "push").all()
    assert (spread_rows["units"] == 0.0).all()


def test_season_summary_aggregates(tmp_path, monkeypatch) -> None:
    """``season_summary`` rolls (season, market) into wins/losses/units."""
    from nfl_model.journal import season_summary

    graded = pd.DataFrame([
        {"season": 2024, "market": "ml", "outcome": "win", "units": 0.91, "prob": 0.65},
        {"season": 2024, "market": "ml", "outcome": "loss", "units": -1.0, "prob": 0.55},
        {"season": 2024, "market": "ml", "outcome": "win", "units": 0.91, "prob": 0.70},
        {"season": 2024, "market": "spread", "outcome": "push", "units": 0.0, "prob": 0.51},
    ])
    summaries = season_summary(graded)
    by = {s.market: s for s in summaries}
    assert by["ml"].wins == 2
    assert by["ml"].losses == 1
    assert by["ml"].pushes == 0
    assert by["spread"].pushes == 1
    assert by["ml"].n == 3
