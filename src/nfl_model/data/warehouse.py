"""DuckDB-backed local warehouse for the NFL bet engine.

A single ``warehouse.duckdb`` file under ``data/`` holds every fact the
feature builders, models, and app query against. Schema is fixed at
init time and idempotent — re-running ``init_schema()`` is a no-op.

Why DuckDB? Same reasoning as the MLB engine: single-file, zero-server,
columnar reads are fast enough for 10-15 years of NFL play-by-play
(~700k rows) and parquet-friendly. All upserts go through
:func:`upsert_dataframe` which uses DuckDB's ``ON CONFLICT`` so daily
re-pulls do not duplicate rows.
"""

from __future__ import annotations

import threading
from contextlib import contextmanager
from typing import Any, Iterator

import duckdb
import pandas as pd

from nfl_model.config import settings
from nfl_model.logging import get_logger

log = get_logger("data.warehouse")

# Process-local guard so the (relatively expensive) DDL only fires once.
_SCHEMA_READY: bool = False
_SCHEMA_LOCK = threading.Lock()


# --------------------------------------------------------------------------- #
# Schema                                                                       #
# --------------------------------------------------------------------------- #

# Every column the feature/model/app code references somewhere. Keep this list
# in sync with downstream callers; the test suite asserts the headline tables
# exist (`games`, `team_game_stats`, `qb_game_stats`, `odds_history`,
# `features`). Adding a column here is cheap; renaming one is a breaking change
# for the feature builders.
_SCHEMA_DDL: list[str] = [
    # ---- Games (one row per scheduled game) -----------------------------
    """
    CREATE TABLE IF NOT EXISTS games (
        game_id      VARCHAR PRIMARY KEY,
        season       INTEGER NOT NULL,
        week         INTEGER,
        game_type    VARCHAR,             -- REG | POST | SB | PRE
        game_date    DATE,
        kickoff_ts   TIMESTAMP,
        weekday      VARCHAR,
        home_team    VARCHAR,
        away_team    VARCHAR,
        home_qb      VARCHAR,
        away_qb      VARCHAR,
        home_coach   VARCHAR,
        away_coach   VARCHAR,
        stadium_id   VARCHAR,
        stadium      VARCHAR,
        roof         VARCHAR,
        surface      VARCHAR,
        primetime    BOOLEAN,
        divisional   BOOLEAN,
        referee      VARCHAR,
        home_score   INTEGER,
        away_score   INTEGER,
        home_win     BOOLEAN,
        overtime     BOOLEAN
    )
    """,
    # ---- Team-game stats (one row per team per game) --------------------
    """
    CREATE TABLE IF NOT EXISTS team_game_stats (
        game_id                VARCHAR,
        team                   VARCHAR,
        is_home                BOOLEAN,
        offense_epa_per_play   DOUBLE,
        defense_epa_per_play   DOUBLE,
        pass_epa               DOUBLE,
        rush_epa               DOUBLE,
        success_rate           DOUBLE,
        explosive_rate         DOUBLE,
        plays_per_drive        DOUBLE,
        neutral_pass_rate      DOUBLE,
        proe                   DOUBLE,
        third_down_pct         DOUBLE,
        red_zone_td_pct        DOUBLE,
        PRIMARY KEY (game_id, team)
    )
    """,
    # ---- QB-game stats (one row per QB per game; multiple QBs possible) -
    """
    CREATE TABLE IF NOT EXISTS qb_game_stats (
        game_id        VARCHAR,
        player_id      VARCHAR,
        player_name    VARCHAR,
        team           VARCHAR,
        attempts       INTEGER,
        epa_per_play   DOUBLE,
        cpoe           DOUBLE,
        sack_rate      DOUBLE,
        is_starter     BOOLEAN,
        PRIMARY KEY (game_id, player_id)
    )
    """,
    # ---- Injury reports (weekly snapshot per player) --------------------
    """
    CREATE TABLE IF NOT EXISTS injuries (
        season           INTEGER,
        week             INTEGER,
        team             VARCHAR,
        player_id        VARCHAR,
        player_name      VARCHAR,
        position         VARCHAR,
        game_status      VARCHAR,        -- Out | Doubtful | Questionable | Probable
        practice_status  VARCHAR,
        report_date      DATE,
        PRIMARY KEY (season, week, team, player_id)
    )
    """,
    # ---- Snap counts ----------------------------------------------------
    """
    CREATE TABLE IF NOT EXISTS snap_counts (
        game_id       VARCHAR,
        player_id     VARCHAR,
        player_name   VARCHAR,
        team          VARCHAR,
        position      VARCHAR,
        offense_pct   DOUBLE,
        defense_pct   DOUBLE,
        st_pct        DOUBLE,
        PRIMARY KEY (game_id, player_id)
    )
    """,
    # ---- Weather (one row per outdoor game) -----------------------------
    """
    CREATE TABLE IF NOT EXISTS weather (
        game_id            VARCHAR PRIMARY KEY,
        temp_f             DOUBLE,
        humidity_pct       DOUBLE,
        pressure_hpa       DOUBLE,
        wind_speed_mph     DOUBLE,
        wind_dir_deg       DOUBLE,
        wind_on_pass_axis  DOUBLE,
        precipitation_mm   DOUBLE,
        condition          VARCHAR,
        is_dome            BOOLEAN
    )
    """,
    # ---- Odds (one row per game per book; 'consensus' is the aggregate) -
    """
    CREATE TABLE IF NOT EXISTS odds_history (
        game_id                     VARCHAR,
        book                        VARCHAR,
        spread_open                 DOUBLE,
        spread_close                DOUBLE,
        spread_close_home_price     INTEGER,
        spread_close_away_price     INTEGER,
        ml_open_home                INTEGER,
        ml_open_away                INTEGER,
        ml_close_home               INTEGER,
        ml_close_away               INTEGER,
        total_open                  DOUBLE,
        total_close                 DOUBLE,
        total_close_over            INTEGER,
        total_close_under           INTEGER,
        home_bets_pct               DOUBLE,
        home_handle_pct             DOUBLE,
        reverse_line_move           BOOLEAN,
        steam_move                  BOOLEAN,
        PRIMARY KEY (game_id, book)
    )
    """,
    # ---- Officiating ----------------------------------------------------
    """
    CREATE TABLE IF NOT EXISTS officiating (
        game_id          VARCHAR PRIMARY KEY,
        referee          VARCHAR,
        crew_id          VARCHAR,
        penalties_total  INTEGER,
        penalty_yards    INTEGER,
        total_points     INTEGER
    )
    """,
    # ---- Venues (static seed table) -------------------------------------
    """
    CREATE TABLE IF NOT EXISTS venues (
        stadium_id              VARCHAR PRIMARY KEY,
        stadium_name            VARCHAR,
        team                    VARCHAR,
        city                    VARCHAR,
        state                   VARCHAR,
        lat                     DOUBLE,
        lon                     DOUBLE,
        elevation_ft            INTEGER,
        roof                    VARCHAR,
        surface                 VARCHAR,
        pass_axis_bearing_deg   DOUBLE,
        timezone                VARCHAR
    )
    """,
    # ---- Play-by-play (large; one row per play, season-partitioned) -----
    # We store a thin subset; everything we currently aggregate in
    # team_game_stats / qb_game_stats lives upstream. Keeping pbp around
    # lets the ablate command re-aggregate on the fly without re-pulling.
    """
    CREATE TABLE IF NOT EXISTS pbp (
        season         INTEGER,
        week           INTEGER,
        game_id        VARCHAR,
        play_id        BIGINT,
        posteam        VARCHAR,
        defteam        VARCHAR,
        down           INTEGER,
        ydstogo        INTEGER,
        yardline_100   INTEGER,
        play_type      VARCHAR,
        passer_id      VARCHAR,
        rusher_id      VARCHAR,
        receiver_id    VARCHAR,
        epa            DOUBLE,
        cpoe           DOUBLE,
        success        BOOLEAN,
        explosive      BOOLEAN,
        air_yards      DOUBLE,
        yards_gained   INTEGER,
        sack           BOOLEAN,
        touchdown      BOOLEAN,
        field_goal     BOOLEAN,
        fg_distance    INTEGER,
        fg_made        BOOLEAN,
        is_pass        BOOLEAN,
        is_rush        BOOLEAN,
        wp             DOUBLE,            -- pre-snap win probability
        PRIMARY KEY (game_id, play_id)
    )
    """,
    # ---- Engineered features (game-keyed JSON blob; cached) -------------
    """
    CREATE TABLE IF NOT EXISTS features (
        game_id              VARCHAR PRIMARY KEY,
        game_date            DATE,
        season               INTEGER,
        week                 INTEGER,
        target_home_win      BOOLEAN,
        target_home_score    INTEGER,
        target_away_score    INTEGER,
        target_total         INTEGER,
        target_margin        INTEGER,
        feature_json         VARCHAR
    )
    """,
]


def _connect(read_only: bool = False) -> duckdb.DuckDBPyConnection:
    """Open a connection to the warehouse DuckDB file.

    Each call returns a fresh connection — DuckDB connections are not
    thread-safe to share. Cheap to open (~1 ms).
    """
    settings.warehouse_path.parent.mkdir(parents=True, exist_ok=True)
    return duckdb.connect(str(settings.warehouse_path), read_only=read_only)


@contextmanager
def connection(read_only: bool = False) -> Iterator[duckdb.DuckDBPyConnection]:
    """Context-managed warehouse connection (closes on exit)."""
    conn = _connect(read_only=read_only)
    try:
        yield conn
    finally:
        conn.close()


def init_schema(*, force: bool = False) -> None:
    """Create every warehouse table if it does not already exist.

    Idempotent and process-cached — subsequent calls in the same process
    short-circuit unless ``force=True`` (used by the tests that point at a
    fresh tmp database).
    """
    global _SCHEMA_READY
    with _SCHEMA_LOCK:
        if _SCHEMA_READY and not force:
            return
        with connection() as conn:
            for ddl in _SCHEMA_DDL:
                conn.execute(ddl)
        _SCHEMA_READY = True
    log.debug("warehouse.schema_ready", path=str(settings.warehouse_path))


def _ensure_ready() -> None:
    if not _SCHEMA_READY:
        init_schema()


# --------------------------------------------------------------------------- #
# Read helpers                                                                 #
# --------------------------------------------------------------------------- #


def list_tables() -> list[str]:
    """Return the warehouse table names."""
    _ensure_ready()
    with connection(read_only=True) as conn:
        rows = conn.execute(
            "SELECT table_name FROM information_schema.tables "
            "WHERE table_schema = 'main' ORDER BY table_name"
        ).fetchall()
    return [r[0] for r in rows]


def table_row_count(table: str) -> int:
    """Return ``COUNT(*)`` for the named table (0 if it doesn't exist)."""
    _ensure_ready()
    with connection(read_only=True) as conn:
        try:
            r = conn.execute(f"SELECT COUNT(*) AS n FROM {table}").fetchone()
        except duckdb.CatalogException:
            return 0
        return int(r[0]) if r else 0


def query(sql: str, params: tuple[Any, ...] | None = None) -> pd.DataFrame:
    """Run a SELECT and return a DataFrame.

    Returns an empty DataFrame if the query targets a table that doesn't
    exist yet — keeps feature builders happy on a pristine warehouse.
    """
    _ensure_ready()
    with connection(read_only=True) as conn:
        try:
            return conn.execute(sql, params or ()).fetchdf()
        except duckdb.CatalogException as exc:
            log.warning("warehouse.query.missing_table", sql=sql.split()[0:6], err=str(exc))
            return pd.DataFrame()


# --------------------------------------------------------------------------- #
# Write helpers                                                                #
# --------------------------------------------------------------------------- #


def _quote_ident(name: str) -> str:
    if not name.replace("_", "").isalnum():
        raise ValueError(f"unsafe identifier: {name!r}")
    return name


def upsert_dataframe(
    df: pd.DataFrame,
    table: str,
    *,
    key_columns: list[str],
) -> int:
    """Upsert rows into ``table`` keyed on ``key_columns``.

    Uses DuckDB's ``INSERT ... ON CONFLICT DO UPDATE`` so daily re-pulls
    refresh fields (e.g. ``home_score`` once a game finalizes) without
    duplicating PK rows.

    Args:
        df: Rows to write. Must contain every ``key_columns`` value.
        table: Target table name. Must exist.
        key_columns: Primary-key column list for conflict resolution.

    Returns:
        Number of rows written.
    """
    if df is None or df.empty:
        return 0
    _ensure_ready()

    safe_table = _quote_ident(table)
    for k in key_columns:
        _quote_ident(k)

    df = df.copy()
    # Make sure column names are str (parquet writes can yield Categorical etc.)
    df.columns = [str(c) for c in df.columns]

    with connection() as conn:
        # Discover the table's existing columns; insert only those that overlap.
        existing = [
            r[0]
            for r in conn.execute(
                f"SELECT column_name FROM information_schema.columns "
                f"WHERE table_schema='main' AND table_name='{safe_table}' "
                f"ORDER BY ordinal_position"
            ).fetchall()
        ]
        if not existing:
            raise ValueError(f"table {table!r} does not exist; call init_schema() first")

        write_cols = [c for c in df.columns if c in existing]
        missing_keys = [k for k in key_columns if k not in write_cols]
        if missing_keys:
            raise ValueError(
                f"upsert into {table} missing key column(s) {missing_keys}; "
                f"have {list(df.columns)}"
            )
        if not write_cols:
            return 0
        sub = df[write_cols]

        update_cols = [c for c in write_cols if c not in key_columns]
        update_clause = (
            ", ".join(f"{_quote_ident(c)} = excluded.{_quote_ident(c)}" for c in update_cols)
            if update_cols
            else None
        )

        col_list = ", ".join(_quote_ident(c) for c in write_cols)
        key_list = ", ".join(_quote_ident(c) for c in key_columns)

        # Register the DataFrame as a temp view, then INSERT...SELECT.
        conn.register("_upsert_src", sub)
        if update_clause:
            sql = (
                f"INSERT INTO {safe_table} ({col_list}) "
                f"SELECT {col_list} FROM _upsert_src "
                f"ON CONFLICT ({key_list}) DO UPDATE SET {update_clause}"
            )
        else:
            sql = (
                f"INSERT INTO {safe_table} ({col_list}) "
                f"SELECT {col_list} FROM _upsert_src "
                f"ON CONFLICT ({key_list}) DO NOTHING"
            )
        conn.execute(sql)
        conn.unregister("_upsert_src")

    log.info("warehouse.upsert", table=table, rows=len(df))
    return int(len(df))


def drop_table(table: str) -> None:
    """Drop ``table`` if it exists. Used by tests + the rare schema reset."""
    _ensure_ready()
    safe = _quote_ident(table)
    with connection() as conn:
        conn.execute(f"DROP TABLE IF EXISTS {safe}")


def execute(sql: str, params: tuple[Any, ...] | None = None) -> None:
    """Run an arbitrary statement (DDL / DELETE / etc.). Used by maintenance."""
    _ensure_ready()
    with connection() as conn:
        conn.execute(sql, params or ())
