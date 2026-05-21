"""Service layer for the NFL Forecast desktop app.

Wraps the weekly prediction pipeline with caching, lightweight result
shaping for templates, and a small picks-log persistence layer. All
business logic lives here so the route module stays thin.

Caching strategy
----------------
The weekly pipeline takes 5-15 seconds (feature build + score models +
Monte Carlo + ensemble + calibrators). To keep the UI snappy we cache
per-week prediction results on disk under
``data/cache/predictions/<season>-W<week>.parquet``. The Refresh button
deletes the cache for the target week and recomputes.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, date as date_cls, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from nfl_model.config import settings
from nfl_model.logging import get_logger

log = get_logger("app.services")


# --------------------------------------------------------------------------- #
# Predictions cache                                                            #
# --------------------------------------------------------------------------- #

PRED_CACHE_DIR = settings.cache_dir / "predictions"


def _cache_path(season: int, week: int) -> Path:
    PRED_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return PRED_CACHE_DIR / f"{season}-W{week:02d}.parquet"


def load_cached_predictions(season: int, week: int) -> pd.DataFrame | None:
    """Return cached prediction DataFrame for ``(season, week)`` if any."""
    path = _cache_path(season, week)
    if not path.exists():
        return None
    try:
        df = pd.read_parquet(path)
        df["game_date"] = pd.to_datetime(df["game_date"]).dt.date
        return df
    except Exception as exc:  # noqa: BLE001 -- defensive cache read
        log.warning("predictions.cache.read_failed", path=str(path), error=str(exc))
        return None


def save_predictions(season: int, week: int, df: pd.DataFrame) -> Path:
    """Persist a freshly computed prediction DataFrame."""
    path = _cache_path(season, week)
    out = df.copy()
    out["game_date"] = pd.to_datetime(out["game_date"]).dt.date
    out.to_parquet(path, index=False)
    return path


def resolve_week(week_arg: str | None) -> tuple[int, int]:
    """Resolve a ``YYYY-W``, ``YYYY/W``, ``current``, or ``None`` query string."""
    from nfl_model.predict.weekly import _resolve_week

    if not week_arg:
        return _resolve_week("current")
    arg = week_arg.replace("/", "-")
    if "-" in arg:
        try:
            return _resolve_week(arg)
        except Exception:  # noqa: BLE001
            return _resolve_week("current")
    if arg.isdigit():
        # Bare integer = week within the current season.
        season, _ = _resolve_week("current")
        return season, int(arg)
    return _resolve_week(arg)


def compute_predictions(season: int, week: int, *, refresh: bool) -> pd.DataFrame:
    """Run the model for ``(season, week)`` (optionally refreshing data)."""
    from nfl_model.predict.weekly import predict_for_week

    df = predict_for_week(f"{season}-{week}", refresh_data=refresh)
    if df is None or df.empty:
        return pd.DataFrame()
    save_predictions(season, week, df)
    return df


def get_predictions(
    season: int, week: int, *, refresh: bool = False,
) -> pd.DataFrame:
    """High-level entrypoint used by the routes."""
    if not refresh:
        cached = load_cached_predictions(season, week)
        if cached is not None and not cached.empty:
            return cached
    return compute_predictions(season, week, refresh=refresh)


def list_available_weeks() -> list[tuple[int, int]]:
    """Return all (season, week) pairs in the warehouse, newest first."""
    from nfl_model.data.warehouse import query

    df = query(
        "SELECT DISTINCT season, week FROM games ORDER BY season DESC, week DESC"
    )
    if df.empty:
        return []
    return [(int(r["season"]), int(r["week"])) for _, r in df.iterrows()]


def neighbor_weeks(season: int, week: int) -> tuple[tuple[int, int] | None, tuple[int, int] | None]:
    """Return ((prev_season, prev_week), (next_season, next_week)) for nav buttons."""
    available = list_available_weeks()
    if not available:
        return None, None
    sorted_asc = list(reversed(available))
    try:
        idx = sorted_asc.index((season, week))
    except ValueError:
        # Current week may not yet have games rows — fall back to placement.
        same_season = [w for s, w in sorted_asc if s == season]
        prev = (season, max([w for w in same_season if w < week], default=None))
        nxt = (season, min([w for w in same_season if w > week], default=None))
        return (prev if prev[1] is not None else None,
                nxt if nxt[1] is not None else None)
    prev = sorted_asc[idx - 1] if idx > 0 else None
    nxt = sorted_asc[idx + 1] if idx < len(sorted_asc) - 1 else None
    return prev, nxt


# --------------------------------------------------------------------------- #
# Pick shaping for the UI                                                      #
# --------------------------------------------------------------------------- #


@dataclass
class PickRow:
    """One row per game per market for the dashboard."""

    game_id: str
    season: int
    week: int
    game_date: date_cls
    away_team: str
    home_team: str
    market: str        # spread | ml | total
    pick: str          # home / away / over / under
    pick_long: str     # 'KC -3.5' / 'KC ML' / 'OVER 47.5'
    model_prob: float
    market_prob: float | None
    edge_pp: float | None
    confidence: float
    tier: str
    expected_margin: float
    expected_total: float
    p_home_win: float
    p_home_cover: float
    p_total_over: float | None
    spread_line: float | None
    total_line: float | None
    ml_close_home: int | None
    ml_close_away: int | None

    def to_dict(self) -> dict[str, Any]:
        d = self.__dict__.copy()
        d["game_date"] = self.game_date.isoformat()
        return d


def _confidence_tier(conf: float) -> str:
    """Bucket confidence into a tier label.

    Mirrors the MLB labels — premium / strong / edge / lean / pass — so
    the same UI palette and filter semantics work across both apps.
    """
    if conf >= 0.40:
        return "premium"
    if conf >= 0.25:
        return "strong"
    if conf >= 0.15:
        return "edge"
    if conf >= 0.07:
        return "lean"
    return "pass"


def _ml_market_prob(home_price: int | None, away_price: int | None) -> float | None:
    """De-vigged P(home win) implied by American moneyline prices."""
    if home_price is None or away_price is None:
        return None
    try:
        h = float(home_price)
        a = float(away_price)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(h) or not np.isfinite(a):
        return None

    def _imp(p: float) -> float:
        return -p / (-p + 100.0) if p < 0 else 100.0 / (p + 100.0)

    p_home = _imp(h)
    p_away = _imp(a)
    s = p_home + p_away
    if s <= 0:
        return None
    return float(p_home / s)


def _format_spread_pick(home_team: str, away_team: str, line: float, pick: str) -> str:
    """Return e.g. 'KC -3.5' or 'KC +3.5' for the picked side."""
    if not np.isfinite(line):
        return f"{home_team if pick == 'home' else away_team} (no line)"
    if pick == "home":
        sign = "" if line < 0 else "+"
        return f"{home_team} {sign}{line:g}"
    away_line = -line
    sign = "" if away_line < 0 else "+"
    return f"{away_team} {sign}{away_line:g}"


def shape_picks(df: pd.DataFrame) -> list[PickRow]:
    """Convert the raw weekly prediction DataFrame to UI rows.

    Each game contributes up to three rows (spread / ml / total) so the
    table can be filtered by market and sorted by confidence within a market.
    """
    if df is None or df.empty:
        return []

    rows: list[PickRow] = []
    for _, r in df.iterrows():
        game_id = str(r["game_id"])
        season = int(r["season"])
        week = int(r["week"])
        gd = r["game_date"]
        game_date = gd if isinstance(gd, date_cls) else pd.to_datetime(gd).date()
        home, away = str(r["home_team"]), str(r["away_team"])

        spread_line = float(r["spread_close"]) if pd.notna(r.get("spread_close")) else None
        total_line = float(r["total_close"]) if pd.notna(r.get("total_close")) else None
        ml_home = int(r["ml_close_home"]) if pd.notna(r.get("ml_close_home")) else None
        ml_away = int(r["ml_close_away"]) if pd.notna(r.get("ml_close_away")) else None

        p_home_win = float(r["p_home_win"]) if pd.notna(r.get("p_home_win")) else 0.5
        p_home_cover = float(r["p_home_cover"]) if pd.notna(r.get("p_home_cover")) else 0.5
        p_total_over = float(r["p_total_over"]) if pd.notna(r.get("p_total_over")) else None
        expected_margin = (
            float(r["expected_margin"]) if pd.notna(r.get("expected_margin")) else 0.0
        )
        expected_total = (
            float(r["expected_total"]) if pd.notna(r.get("expected_total")) else 0.0
        )

        # ----------------------------------------------------------- spread
        spread_pick = "home" if p_home_cover >= 0.5 else "away"
        spread_prob = p_home_cover if spread_pick == "home" else 1.0 - p_home_cover
        spread_long = (
            _format_spread_pick(home, away, spread_line, spread_pick)
            if spread_line is not None else f"{home if spread_pick == 'home' else away} (no line)"
        )
        spread_conf = abs(p_home_cover - 0.5) * 2.0
        # NFL spreads are -110/-110 → de-vigged market prob = 50%, so edge
        # is just (p_cover_picked_side - 0.5) * 100 in pp.
        spread_edge_pp = (spread_prob - 0.5) * 100.0
        rows.append(PickRow(
            game_id=game_id, season=season, week=week, game_date=game_date,
            away_team=away, home_team=home,
            market="spread", pick=spread_pick, pick_long=spread_long,
            model_prob=spread_prob,
            market_prob=0.5,
            edge_pp=spread_edge_pp,
            confidence=spread_conf,
            tier=_confidence_tier(spread_conf),
            expected_margin=expected_margin, expected_total=expected_total,
            p_home_win=p_home_win, p_home_cover=p_home_cover,
            p_total_over=p_total_over,
            spread_line=spread_line, total_line=total_line,
            ml_close_home=ml_home, ml_close_away=ml_away,
        ))

        # ----------------------------------------------------------- ml
        ml_pick = "home" if p_home_win >= 0.5 else "away"
        ml_prob = p_home_win if ml_pick == "home" else 1.0 - p_home_win
        ml_long = f"{home if ml_pick == 'home' else away} ML"
        ml_conf = abs(p_home_win - 0.5) * 2.0
        market_p_home = _ml_market_prob(ml_home, ml_away)
        ml_market_prob: float | None = None
        ml_edge_pp: float | None = None
        if market_p_home is not None:
            ml_market_prob = market_p_home if ml_pick == "home" else 1.0 - market_p_home
            ml_edge_pp = (ml_prob - ml_market_prob) * 100.0
        rows.append(PickRow(
            game_id=game_id, season=season, week=week, game_date=game_date,
            away_team=away, home_team=home,
            market="ml", pick=ml_pick, pick_long=ml_long,
            model_prob=ml_prob, market_prob=ml_market_prob,
            edge_pp=ml_edge_pp, confidence=ml_conf,
            tier=_confidence_tier(ml_conf),
            expected_margin=expected_margin, expected_total=expected_total,
            p_home_win=p_home_win, p_home_cover=p_home_cover,
            p_total_over=p_total_over,
            spread_line=spread_line, total_line=total_line,
            ml_close_home=ml_home, ml_close_away=ml_away,
        ))

        # ----------------------------------------------------------- total
        if p_total_over is not None and total_line is not None:
            ou_pick = "over" if p_total_over >= 0.5 else "under"
            ou_prob = p_total_over if ou_pick == "over" else 1.0 - p_total_over
            ou_long = f"{ou_pick.upper()} {total_line:g}"
            ou_conf = abs(p_total_over - 0.5) * 2.0
            ou_edge_pp = (ou_prob - 0.5) * 100.0  # -110/-110 baseline
            rows.append(PickRow(
                game_id=game_id, season=season, week=week, game_date=game_date,
                away_team=away, home_team=home,
                market="total", pick=ou_pick, pick_long=ou_long,
                model_prob=ou_prob, market_prob=0.5, edge_pp=ou_edge_pp,
                confidence=ou_conf, tier=_confidence_tier(ou_conf),
                expected_margin=expected_margin, expected_total=expected_total,
                p_home_win=p_home_win, p_home_cover=p_home_cover,
                p_total_over=p_total_over,
                spread_line=spread_line, total_line=total_line,
                ml_close_home=ml_home, ml_close_away=ml_away,
            ))

    rows.sort(key=lambda r: r.confidence, reverse=True)
    return rows


# --------------------------------------------------------------------------- #
# Game detail                                                                  #
# --------------------------------------------------------------------------- #


@dataclass
class GameDetail:
    game_id: str
    season: int
    week: int
    game_date: date_cls
    away_team: str
    home_team: str
    home_qb: str | None
    away_qb: str | None
    venue: str
    roof: str | None
    surface: str | None
    weekday: str | None
    spread_line: float | None
    total_line: float | None
    ml_close_home: int | None
    ml_close_away: int | None
    p_home_win: float
    p_home_cover: float
    p_total_over: float | None
    expected_margin: float
    expected_total: float
    weather: dict[str, Any]
    score_distribution: dict[str, Any]
    feature_panel: list[dict[str, Any]]


def _normal_pmf(mean: float, std: float, k_max: int = 60) -> list[float]:
    """Approximate P(score = k) via a Gaussian, k=0..k_max.

    NFL final scores are integer-valued but nowhere near Poisson — we use
    a normal-of-margin proxy to give the chart shape. The chart is only
    illustrative; the underlying simulator does the real work.
    """
    if not np.isfinite(mean) or std <= 0:
        std = max(std, 1.0)
        mean = max(mean, 0.0)
    ks = np.arange(0, k_max + 1)
    pdf = (1.0 / (std * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((ks - mean) / std) ** 2)
    pdf = np.where(pdf < 1e-9, 0.0, pdf)
    s = pdf.sum()
    if s > 0:
        pdf = pdf / s
    return [float(x) for x in pdf]


def _score_distribution(home_mean: float, away_mean: float) -> dict[str, Any]:
    """Compute approximate P(score = k) for k=0..60 for both teams."""
    home_std = max(7.0, np.sqrt(max(home_mean, 1.0)) * 2.0)
    away_std = max(7.0, np.sqrt(max(away_mean, 1.0)) * 2.0)
    return {
        "k": list(range(61)),
        "home": _normal_pmf(home_mean, home_std, 60),
        "away": _normal_pmf(away_mean, away_std, 60),
    }


def get_game_detail(season: int, week: int, game_id: str) -> GameDetail | None:
    """Build the full game-detail payload for one game."""
    from nfl_model.data.warehouse import query

    df = get_predictions(season, week)
    if df.empty:
        return None
    row = df[df["game_id"].astype(str) == str(game_id)]
    if row.empty:
        return None
    r = row.iloc[0]

    meta = query(
        """
        SELECT g.stadium, g.roof, g.surface, g.weekday, g.home_qb, g.away_qb
        FROM games g WHERE g.game_id = ?
        """,
        (str(game_id),),
    )
    if not meta.empty:
        m = meta.iloc[0]
        venue = str(m["stadium"]) if pd.notna(m["stadium"]) else "Unknown"
        roof = str(m["roof"]) if pd.notna(m["roof"]) else None
        surface = str(m["surface"]) if pd.notna(m["surface"]) else None
        weekday = str(m["weekday"]) if pd.notna(m["weekday"]) else None
        home_qb = str(m["home_qb"]) if pd.notna(m["home_qb"]) else None
        away_qb = str(m["away_qb"]) if pd.notna(m["away_qb"]) else None
    else:
        venue = "Unknown"
        roof = surface = weekday = home_qb = away_qb = None

    weather_row = query(
        """
        SELECT temp_f, humidity_pct, wind_speed_mph, wind_dir_deg,
               precipitation_mm, condition, is_dome
        FROM weather WHERE game_id = ?
        """,
        (str(game_id),),
    )
    weather: dict[str, Any] = {}
    if not weather_row.empty:
        w = weather_row.iloc[0]
        weather = {
            "temp_f": float(w["temp_f"]) if pd.notna(w["temp_f"]) else None,
            "humidity_pct": float(w["humidity_pct"]) if pd.notna(w["humidity_pct"]) else None,
            "wind_speed_mph": float(w["wind_speed_mph"]) if pd.notna(w["wind_speed_mph"]) else None,
            "wind_dir_deg": float(w["wind_dir_deg"]) if pd.notna(w["wind_dir_deg"]) else None,
            "precipitation_mm": float(w["precipitation_mm"]) if pd.notna(w["precipitation_mm"]) else None,
            "condition": str(w["condition"]) if pd.notna(w["condition"]) else None,
            "is_dome": bool(w["is_dome"]) if pd.notna(w["is_dome"]) else False,
        }

    expected_margin = float(r["expected_margin"]) if pd.notna(r.get("expected_margin")) else 0.0
    expected_total = float(r["expected_total"]) if pd.notna(r.get("expected_total")) else 0.0
    home_mean = (expected_total + expected_margin) / 2.0
    away_mean = (expected_total - expected_margin) / 2.0
    distro = _score_distribution(home_mean, away_mean)

    panel = _feature_panel(season, week, str(game_id))

    spread_line = float(r["spread_close"]) if pd.notna(r.get("spread_close")) else None
    total_line = float(r["total_close"]) if pd.notna(r.get("total_close")) else None
    ml_home = int(r["ml_close_home"]) if pd.notna(r.get("ml_close_home")) else None
    ml_away = int(r["ml_close_away"]) if pd.notna(r.get("ml_close_away")) else None

    gd = r["game_date"]
    game_date = gd if isinstance(gd, date_cls) else pd.to_datetime(gd).date()

    return GameDetail(
        game_id=str(r["game_id"]),
        season=int(r["season"]), week=int(r["week"]),
        game_date=game_date,
        away_team=str(r["away_team"]), home_team=str(r["home_team"]),
        home_qb=home_qb, away_qb=away_qb,
        venue=venue, roof=roof, surface=surface, weekday=weekday,
        spread_line=spread_line, total_line=total_line,
        ml_close_home=ml_home, ml_close_away=ml_away,
        p_home_win=float(r["p_home_win"]),
        p_home_cover=float(r["p_home_cover"]),
        p_total_over=float(r["p_total_over"]) if pd.notna(r.get("p_total_over")) else None,
        expected_margin=expected_margin, expected_total=expected_total,
        weather=weather, score_distribution=distro,
        feature_panel=panel,
    )


def _feature_panel(season: int, week: int, game_id: str) -> list[dict[str, Any]]:
    """Hand-curated panel of features for the game-detail view."""
    try:
        from nfl_model.features.assemble import build_features_table

        feats = build_features_table(season - 1, season)
    except Exception:  # noqa: BLE001
        return []
    if feats is None or feats.empty:
        return []

    row = feats[feats["game_id"].astype(str) == game_id]
    if row.empty:
        return []
    r = row.iloc[0]

    def get(col: str) -> float | None:
        if col in r.index and pd.notna(r[col]):
            return float(r[col])
        return None

    panel: list[dict[str, Any]] = []

    def add(label: str, group: str, home_col: str, away_col: str,
            unit: str = "", fmt: str = "{:.2f}") -> None:
        h, a = get(home_col), get(away_col)
        if h is None and a is None:
            return
        panel.append({
            "label": label, "group": group,
            "home": fmt.format(h) + unit if h is not None else "—",
            "away": fmt.format(a) + unit if a is not None else "—",
            "home_raw": h, "away_raw": a,
        })

    # Team form
    add("EPA/play (off), last 4g",   "Offense", "home_team_off_epa_per_play_r4g",  "away_team_off_epa_per_play_r4g", fmt="{:+.3f}")
    add("EPA/play (def), last 4g",   "Defense", "home_team_def_epa_per_play_r4g",  "away_team_def_epa_per_play_r4g", fmt="{:+.3f}")
    add("Success rate (off), 4g",    "Offense", "home_team_off_success_rate_r4g",  "away_team_off_success_rate_r4g", unit="%", fmt="{:.1%}")
    add("Explosive play %, 4g",      "Offense", "home_team_off_explosive_pct_r4g", "away_team_off_explosive_pct_r4g", unit="%", fmt="{:.1%}")
    add("Pace (sec/play), 4g",       "Offense", "home_team_pace_secs_r4g",         "away_team_pace_secs_r4g")
    add("PROE (pass over expected), 4g", "Offense", "home_team_proe_r4g",          "away_team_proe_r4g", unit="%", fmt="{:.1%}")

    # QB form
    add("QB EPA/db, last 4g",        "QB", "home_qb_epa_per_db_r4g", "away_qb_epa_per_db_r4g", fmt="{:+.3f}")
    add("QB CPOE, last 4g",          "QB", "home_qb_cpoe_r4g",       "away_qb_cpoe_r4g",        unit="%", fmt="{:.1%}")
    add("QB pressure rate allowed",  "QB", "home_qb_pressure_rate_r4g", "away_qb_pressure_rate_r4g", unit="%", fmt="{:.1%}")

    # Injuries
    add("Starters Out + Doubtful",   "Injuries", "home_inj_starters_out_or_doubtful", "away_inj_starters_out_or_doubtful", fmt="{:.0f}")
    add("Skill players Q+",          "Injuries", "home_inj_skill_questionable_or_worse", "away_inj_skill_questionable_or_worse", fmt="{:.0f}")
    add("OL starters limited",       "Injuries", "home_inj_ol_starters_limited", "away_inj_ol_starters_limited", fmt="{:.0f}")

    # Schedule + travel
    add("Days of rest",              "Context", "home_days_rest", "away_days_rest", fmt="{:.0f}")
    add("Travel miles",              "Context", "home_travel_miles", "away_travel_miles", fmt="{:.0f}")
    add("Short week (≤6 days)",      "Context", "home_short_week", "away_short_week", fmt="{:.0f}")

    return panel


# --------------------------------------------------------------------------- #
# Performance                                                                  #
# --------------------------------------------------------------------------- #


def load_backtest() -> pd.DataFrame:
    """Read the latest walk-forward backtest CSV."""
    candidates = [
        settings.logs_dir / "backtest_v4.csv",
        settings.logs_dir / "backtest_v3.csv",
        settings.logs_dir / "backtest_v2.csv",
        settings.logs_dir / "backtest_v1.csv",
    ]
    for path in candidates:
        if path.exists():
            return pd.read_csv(path)
    return pd.DataFrame()


# --------------------------------------------------------------------------- #
# Picks log (persisted)                                                        #
# --------------------------------------------------------------------------- #


PICKS_LOG_PATH = settings.cache_dir / "picks_log.parquet"


def _read_log_raw() -> pd.DataFrame:
    if not PICKS_LOG_PATH.exists():
        return pd.DataFrame()
    df = pd.read_parquet(PICKS_LOG_PATH)
    # Backfill ``pick_id`` and ``stake_units`` for rows logged earlier.
    if "pick_id" not in df.columns or df["pick_id"].isna().any():
        import uuid

        if "pick_id" not in df.columns:
            df["pick_id"] = ""
        df["pick_id"] = df["pick_id"].fillna("").astype(str)
        for idx in df.index[df["pick_id"] == ""]:
            df.at[idx, "pick_id"] = uuid.uuid4().hex
    if "stake_units" not in df.columns:
        df["stake_units"] = 1.0
    df["stake_units"] = pd.to_numeric(df["stake_units"], errors="coerce").fillna(1.0)
    return df


def append_logged_pick(record: dict[str, Any]) -> str:
    """Persist a "I'm taking this pick" record. Returns the new pick_id."""
    import uuid

    PICKS_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    record = {
        **record,
        "pick_id": record.get("pick_id") or uuid.uuid4().hex,
        "logged_at": datetime.now(UTC).isoformat(),
        "stake_units": float(record.get("stake_units", 1.0)),
    }
    existing = _read_log_raw()
    if existing.empty:
        df = pd.DataFrame([record])
    else:
        mask = (
            (existing["game_id"].astype(str) == str(record["game_id"]))
            & (existing["market"] == record["market"])
            & (existing["pick"] == record["pick"])
        )
        if mask.any():
            for col, val in record.items():
                if col in ("logged_at", "pick_id"):
                    continue
                existing.loc[mask, col] = val
            df = existing
        else:
            df = pd.concat([existing, pd.DataFrame([record])], ignore_index=True)
    df.to_parquet(PICKS_LOG_PATH, index=False)
    return str(record["pick_id"])


def delete_logged_pick(pick_id: str) -> bool:
    df = _read_log_raw()
    if df.empty:
        return False
    mask = df["pick_id"].astype(str) == str(pick_id)
    if not mask.any():
        return False
    df = df.loc[~mask].reset_index(drop=True)
    if df.empty:
        PICKS_LOG_PATH.unlink(missing_ok=True)
    else:
        df.to_parquet(PICKS_LOG_PATH, index=False)
    return True


def update_logged_pick(pick_id: str, *, stake_units: float | None = None) -> bool:
    df = _read_log_raw()
    if df.empty:
        return False
    mask = df["pick_id"].astype(str) == str(pick_id)
    if not mask.any():
        return False
    if stake_units is not None:
        df.loc[mask, "stake_units"] = max(0.0, float(stake_units))
    df.to_parquet(PICKS_LOG_PATH, index=False)
    return True


def clear_logged_picks() -> int:
    df = _read_log_raw()
    n = int(len(df))
    PICKS_LOG_PATH.unlink(missing_ok=True)
    return n


def load_picks_log() -> pd.DataFrame:
    df = _read_log_raw()
    if df.empty:
        return df
    return _grade_picks(df)


def _grade_picks(picks: pd.DataFrame) -> pd.DataFrame:
    """Join logged picks against finalized game results."""
    from nfl_model.data.warehouse import query

    if picks.empty:
        return picks
    ids = picks["game_id"].astype(str).unique().tolist()
    if not ids:
        return picks
    finals = query(
        f"""
        SELECT game_id, home_score, away_score, home_win
        FROM games WHERE game_id IN ({",".join("?" for _ in ids)})
        """,
        tuple(ids),
    )
    if finals.empty:
        picks = picks.copy()
        picks["result"] = "pending"
        picks["roi_units"] = float("nan")
        return picks
    merged = picks.merge(finals, on="game_id", how="left")

    def _outcome(row: pd.Series) -> str:
        hs = row.get("home_score")
        as_ = row.get("away_score")
        if pd.isna(hs) or pd.isna(as_):
            return "pending"
        hs = float(hs)
        as_ = float(as_)
        market = str(row.get("market", "")).lower()
        pick = str(row.get("pick", "")).lower()
        line = row.get("market_line")
        if market == "ml":
            home_won = hs > as_
            if hs == as_:
                return "push"
            won = home_won if pick == "home" else (not home_won)
            return "win" if won else "loss"
        if market == "spread":
            margin = hs - as_
            try:
                line_f = float(line)
            except (TypeError, ValueError):
                return "pending"
            adjusted = margin + line_f if pick == "home" else -margin - line_f
            if adjusted == 0:
                return "push"
            return "win" if adjusted > 0 else "loss"
        if market == "total":
            try:
                line_f = float(line)
            except (TypeError, ValueError):
                return "pending"
            total = hs + as_
            if total == line_f:
                return "push"
            went_over = total > line_f
            return "win" if (went_over and pick == "over") or (not went_over and pick == "under") else "loss"
        return "pending"

    merged["result"] = merged.apply(_outcome, axis=1)

    def _roi_per_unit(row: pd.Series) -> float:
        result = row.get("result")
        if result == "win":
            price = row.get("market_price")
            try:
                p = float(price) if price is not None else -110.0
            except (TypeError, ValueError):
                p = -110.0
            return p / 100.0 if p >= 100 else 100.0 / abs(p)
        if result == "loss":
            return -1.0
        if result == "push":
            return 0.0
        return float("nan")

    per_unit = merged.apply(_roi_per_unit, axis=1)
    stakes = pd.to_numeric(merged.get("stake_units", 1.0), errors="coerce").fillna(1.0)
    merged["roi_units"] = per_unit * stakes
    return merged


# --------------------------------------------------------------------------- #
# Season aggregation                                                           #
# --------------------------------------------------------------------------- #


def season_performance(season: int | None = None) -> dict[str, Any]:
    """Aggregate the prediction journal into everything the /season page needs."""
    from nfl_model.journal import (
        JOURNAL_PATH,
        calibration_bins,
        grade_journal,
        rolling_accuracy,
        season_summary,
        slice_breakdown,
    )

    if not JOURNAL_PATH.exists():
        return {
            "season": season,
            "available_seasons": [],
            "summary": [],
            "calibration": {},
            "rolling": {},
            "tier_breakdown": {},
            "recent_results": [],
            "journal_size": 0,
            "eos_report": None,
        }

    journal = pd.read_parquet(JOURNAL_PATH)
    available_seasons = sorted(
        int(s) for s in pd.unique(pd.to_numeric(journal["season"], errors="coerce").dropna())
    )
    if season is None and available_seasons:
        season = max(available_seasons)

    graded = grade_journal(journal=journal, season=season)
    summary = [s.as_dict() for s in season_summary(graded)]

    markets = ["spread", "ml", "total"]
    calibration: dict[str, list[dict[str, Any]]] = {}
    rolling: dict[str, list[dict[str, Any]]] = {}
    tier_breakdown: dict[str, list[dict[str, Any]]] = {}

    for market in markets:
        cb = calibration_bins(graded, market=market)
        if not cb.empty:
            calibration[market] = cb.to_dict("records")
        ra = rolling_accuracy(graded, market=market, window_days=30)
        if not ra.empty:
            recs = ra.to_dict("records")
            for rec in recs:
                rec["game_date"] = rec["game_date"].isoformat() if rec["game_date"] else None
            rolling[market] = recs
        sub = graded[graded["market"] == market] if graded is not None else pd.DataFrame()
        sb = slice_breakdown(sub, by="tier")
        if not sb.empty:
            tier_breakdown[market] = sb.to_dict("records")

    if graded is not None and not graded.empty:
        tail = (
            graded[graded["outcome"].isin({"win", "loss", "push"})]
            .sort_values("game_date", ascending=False)
            .head(50)
        )
        keep_cols = [
            c for c in [
                "game_date", "away_team", "home_team", "market",
                "pick", "model_prob", "tier", "outcome", "units",
            ] if c in tail.columns
        ]
        recent = tail[keep_cols].copy()
        recent["game_date"] = recent["game_date"].astype(str)
        recent_results = recent.to_dict("records")
    else:
        recent_results = []

    eos_links: dict[str, str] | None = None
    if season is not None:
        try:
            from nfl_model.season.end_of_season import REPORTS_ROOT

            report_dir = REPORTS_ROOT / f"end_of_season_{season}"
            md_path = report_dir / "report.md"
            json_path = report_dir / "summary.json"
            if md_path.exists() or json_path.exists():
                eos_links = {"dir": str(report_dir)}
                if md_path.exists():
                    eos_links["markdown"] = str(md_path)
                if json_path.exists():
                    eos_links["json"] = str(json_path)
        except Exception:  # noqa: BLE001
            eos_links = None

    return {
        "season": season,
        "available_seasons": available_seasons,
        "summary": summary,
        "calibration": calibration,
        "rolling": rolling,
        "tier_breakdown": tier_breakdown,
        "recent_results": recent_results,
        "journal_size": int(len(journal)),
        "eos_report": eos_links,
    }


# --------------------------------------------------------------------------- #
# Wong-teaser cards                                                            #
# --------------------------------------------------------------------------- #


def teaser_cards(season: int, week: int) -> list[dict[str, Any]]:
    """Return ranked Wong-teaser leg cards for the given week (or [] on failure)."""
    try:
        df = get_predictions(season, week)
    except Exception:  # noqa: BLE001
        return []
    if df is None or df.empty:
        return []

    try:
        from nfl_model.model.teasers import (
            construct_teaser_legs,
            filter_wong_legs,
            teaser_card_dataframe,
        )
    except Exception:  # noqa: BLE001
        return []

    spreads = pd.to_numeric(df["spread_close"], errors="coerce").to_numpy()

    def _p_home_cover_at(i: int, line: float) -> float:
        base = float(df["p_home_cover"].iloc[i])
        # Cheap inference: shift the calibrated cover prob by (line - posted)
        # using a ~10-pt-per-σ rule of thumb. See teasers module for the
        # fully-fledged variant when we re-simulate.
        shift = (line - spreads[i]) / 10.0
        return float(np.clip(base + 0.5 * shift, 0.01, 0.99))

    legs = construct_teaser_legs(
        game_ids=df["game_id"].to_list(),
        home_teams=df["home_team"].to_list(),
        away_teams=df["away_team"].to_list(),
        spreads_home=spreads,
        p_home_cover_at=_p_home_cover_at,
    )
    wong = filter_wong_legs(legs, min_prob=settings.teaser_min_leg_prob - 0.10)
    return teaser_card_dataframe(wong).to_dict("records")
