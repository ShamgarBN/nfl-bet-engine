"""Regression test for the nflverse spread_line sign-convention gotcha.

nflverse encodes ``spread_line`` as "points the home team is favored by"
(positive when home is the favorite). The rest of the codebase — the
simulator, backtest cover math, the app's pick formatting — uses the
standard sportsbook convention where a *negative* spread means the home
team is laying points (KC -3.5).

:mod:`nfl_model.data.sources.nflverse.pull_odds_from_schedules` is the
choke point: it negates ``spread_line`` at ingestion time so every
consumer downstream can rely on the familiar convention. If anybody ever
"fixes" that flip by removing the negation, this test fails loudly
*before* the bug silently inflates backtest ATS numbers to 80%+.
"""

from __future__ import annotations

import pandas as pd


def test_spread_sign_matches_sportsbook_convention(tmp_path, monkeypatch) -> None:
    """A home-favorite ML game should write a NEGATIVE spread to odds_history."""
    from nfl_model import config as cfg
    from nfl_model.data import warehouse as wh
    from nfl_model.data.sources.nflverse import pull_odds_from_schedules

    monkeypatch.setattr(cfg.settings, "warehouse_path", tmp_path / "wh.duckdb")
    wh._SCHEMA_READY = False  # noqa: SLF001
    wh.init_schema(force=True)

    # Stub nflreadpy.load_schedules to return a single row where home is the
    # ML favorite — so spread_line is positive per nflverse convention.
    import nflreadpy
    import polars as pl

    schedule_row = pl.DataFrame(
        [
            {
                "game_id": "2099_01_AAA_BBB",
                "season": 2099,
                "week": 1,
                # In the real feed: home (BBB) favored by 4.5 -> spread_line=+4.5
                "spread_line": 4.5,
                "home_moneyline": -200,
                "away_moneyline": 170,
                "total_line": 47.5,
                "home_spread_odds": -110,
                "away_spread_odds": -110,
                "over_odds": -110,
                "under_odds": -110,
            }
        ]
    )
    monkeypatch.setattr(nflreadpy, "load_schedules", lambda seasons=None: schedule_row)

    pull_odds_from_schedules([2099])

    df = wh.query("SELECT spread_close, spread_open FROM odds_history WHERE game_id = ?",
                  ("2099_01_AAA_BBB",))
    assert not df.empty, "spread row was not written"
    # Home is favored -> sportsbook convention says spread_close is negative.
    assert float(df.iloc[0]["spread_close"]) == -4.5, (
        f"home-favorite spread should be negative; got {df.iloc[0]['spread_close']}. "
        "If this fails, the nflverse spread_line sign-flip in "
        "pull_odds_from_schedules was reverted — see file docstring."
    )
    assert float(df.iloc[0]["spread_open"]) == -4.5
