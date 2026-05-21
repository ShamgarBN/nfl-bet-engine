"""Smoke tests: package imports and basic config sanity."""

from __future__ import annotations


def test_package_imports() -> None:
    import nfl_model

    assert nfl_model.__version__ == "0.1.0"


def test_config_loads() -> None:
    from nfl_model.config import settings

    assert settings.backtest_start_year == 2014
    assert settings.backtest_end_year >= 2025
    assert 3 in settings.key_numbers_spread
    assert 7 in settings.key_numbers_spread
    assert settings.teaser_points == 6
    assert settings.teaser_min_leg_prob >= 0.7


def test_warehouse_schema_initializes(tmp_path, monkeypatch) -> None:
    """Warehouse should initialize against a temporary DuckDB file."""
    from nfl_model import config as cfg
    from nfl_model.data import warehouse as wh

    monkeypatch.setattr(cfg.settings, "warehouse_path", tmp_path / "warehouse.duckdb")
    wh._SCHEMA_READY = False  # noqa: SLF001 -- reset cached flag for the test
    wh.init_schema(force=True)
    tables = wh.list_tables()
    assert "games" in tables
    assert "team_game_stats" in tables
    assert "qb_game_stats" in tables
    assert "odds_history" in tables
    assert "features" in tables
