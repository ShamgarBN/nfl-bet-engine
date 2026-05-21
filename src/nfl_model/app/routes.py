"""HTTP routes for the NFL Forecast desktop app.

Page-level routes:
* ``/`` — current week's slate (filterable by market + tier)
* ``/game/<id>`` — game detail with score distribution + feature panel
* ``/season`` — graded journal: per-market summaries, calibration, rolling
* ``/performance`` — walk-forward backtest table
* ``/log`` — user-logged picks + auto-grading
* ``/teasers`` — Wong-teaser opportunities for the current week

Plus HTMX partials, JSON APIs, and CSV export endpoints.

Security posture: bound to 127.0.0.1 only. We treat every value in the
pick-log JSON as opaque strings that are later joined against trusted
``game_id`` strings, so there is no injection surface against the
warehouse.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import APIRouter, Form, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, Response
from fastapi.templating import Jinja2Templates

from nfl_model.app import services
from nfl_model.logging import get_logger

log = get_logger("app.routes")

APP_DIR = Path(__file__).resolve().parent
templates = Jinja2Templates(directory=APP_DIR / "templates")

router = APIRouter()


# --------------------------------------------------------------------------- #
# Jinja helpers                                                                #
# --------------------------------------------------------------------------- #


def _pct(x: float | None, digits: int = 1) -> str:
    if x is None or (isinstance(x, float) and (x != x)):
        return "—"
    return f"{x * 100:.{digits}f}%"


def _odds(prob: float | None) -> str:
    if prob is None or prob <= 0 or prob >= 1 or (isinstance(prob, float) and prob != prob):
        return "—"
    if prob >= 0.5:
        return f"-{round(prob / (1 - prob) * 100)}"
    return f"+{round((1 - prob) / prob * 100)}"


def _num(x: float | None, digits: int = 2) -> str:
    if x is None or (isinstance(x, float) and (x != x)):
        return "—"
    return f"{x:.{digits}f}"


def _signed_pp(x: float | None) -> str:
    if x is None or (isinstance(x, float) and (x != x)):
        return "—"
    return f"{x:+.1f}pp"


def _signed(x: float | None) -> str:
    if x is None or (isinstance(x, float) and (x != x)):
        return "—"
    return f"{x:+g}"


templates.env.filters["pct"] = _pct
templates.env.filters["odds"] = _odds
templates.env.filters["num"] = _num
templates.env.filters["signed_pp"] = _signed_pp
templates.env.filters["signed"] = _signed


# --------------------------------------------------------------------------- #
# Automation status                                                            #
# --------------------------------------------------------------------------- #


def _automation_status() -> dict[str, str]:
    from datetime import date as _date

    try:
        from nfl_model.automation import morning_sync, weekly_train
    except Exception:  # noqa: BLE001
        return {"morning_sync": "—", "morning_sync_state": "never", "weekly_train": "—"}

    def _fmt(d: Any) -> str:
        return d.isoformat() if d is not None else "never"

    last_sync = morning_sync.last_run_date() if hasattr(morning_sync, "last_run_date") else None
    today = _date.today()
    if last_sync is None:
        sync_state = "never"
    elif last_sync >= today:
        sync_state = "ok"
    elif (today - last_sync).days <= 1:
        sync_state = "yesterday"
    else:
        sync_state = "stale"

    last_train = (
        weekly_train.last_run_date() if hasattr(weekly_train, "last_run_date") else None
    )
    return {
        "morning_sync": _fmt(last_sync),
        "morning_sync_state": sync_state,
        "weekly_train": _fmt(last_train),
    }


def _render(request: Request, template: str, context: dict[str, Any]) -> HTMLResponse:
    full_context = {**context, "automation_status": _automation_status()}
    return templates.TemplateResponse(request, template, full_context)


# --------------------------------------------------------------------------- #
# Pages                                                                        #
# --------------------------------------------------------------------------- #


_TIER_THRESHOLDS = {
    "all": 0.0,
    "lean": 0.07,
    "edge": 0.15,
    "strong": 0.25,
    "premium": 0.40,
}


def _filter_picks(
    picks: list[services.PickRow],
    *,
    market: str,
    tier: str,
) -> list[services.PickRow]:
    market = market.lower()
    tier = tier.lower()
    out = [p for p in picks if p.market == market]
    threshold = _TIER_THRESHOLDS.get(tier, 0.0)
    out = [p for p in out if p.confidence >= threshold]
    out.sort(key=lambda p: p.confidence, reverse=True)
    return out


def _format_week_label(season: int, week: int) -> str:
    if week >= 19:
        names = {19: "Wild Card", 20: "Divisional", 21: "Conf. Champ.", 22: "Super Bowl"}
        return f"{season} {names.get(week, f'Postseason W{week}')}"
    return f"{season} · Week {week}"


@router.get("/", response_class=HTMLResponse)
def dashboard(
    request: Request,
    week: str | None = Query(default=None),
    market: str = Query(default="spread"),
    tier: str = Query(default="all"),
) -> HTMLResponse:
    """Current week's slate."""
    season, wk = services.resolve_week(week)
    has_cache = services.load_cached_predictions(season, wk) is not None
    error: str | None = None
    try:
        df = services.get_predictions(season, wk, refresh=False)
    except Exception as exc:  # noqa: BLE001
        log.exception("dashboard.predict_failed")
        df = pd.DataFrame()
        error = str(exc)

    picks = services.shape_picks(df) if not df.empty else []
    picks_filtered = _filter_picks(picks, market=market, tier=tier)
    prev, nxt = services.neighbor_weeks(season, wk)
    available = services.list_available_weeks()

    return _render(
        request,
        "dashboard.html",
        {
            "season": season,
            "week": wk,
            "week_label": _format_week_label(season, wk),
            "picks": picks_filtered,
            "total_picks": len(picks),
            "active_market": market,
            "active_tier": tier,
            "prev_week": prev,
            "next_week": nxt,
            "available_weeks": available,
            "has_cache": has_cache,
            "error": error,
        },
    )


@router.get("/game/{game_id}", response_class=HTMLResponse)
def game_detail(
    request: Request,
    game_id: str,
    week: str | None = Query(default=None),
) -> HTMLResponse:
    season, wk = services.resolve_week(week)
    detail = services.get_game_detail(season, wk, game_id)
    if detail is None:
        # Try locating the game's actual week if the user opened a stale URL.
        from nfl_model.data.warehouse import query

        match = query(
            "SELECT season, week FROM games WHERE game_id = ?", (str(game_id),),
        )
        if not match.empty:
            r = match.iloc[0]
            detail = services.get_game_detail(int(r["season"]), int(r["week"]), game_id)
            if detail is not None:
                season, wk = int(r["season"]), int(r["week"])
        if detail is None:
            raise HTTPException(status_code=404, detail="Game not found in predictions.")

    return _render(
        request,
        "game_detail.html",
        {
            "g": detail, "season": season, "week": wk,
            "week_label": _format_week_label(season, wk),
        },
    )


@router.get("/performance", response_class=HTMLResponse)
def performance(request: Request) -> HTMLResponse:
    df = services.load_backtest()
    rows = df.to_dict(orient="records") if not df.empty else []
    headline: dict[str, Any] = {}

    if not df.empty and "n_games" in df.columns:
        w = pd.to_numeric(df["n_games"], errors="coerce")
        for key, label_col in [
            ("spread", "ats_eng_acc"),
            ("ml", "ml_eng_acc"),
            ("total", "ou_eng_acc"),
        ]:
            if label_col not in df.columns:
                continue
            vals = pd.to_numeric(df[label_col], errors="coerce")
            mask = vals.notna() & w.notna()
            if not mask.any():
                continue
            wt = w[mask].fillna(0.0)
            total_w = float(wt.sum())
            if total_w <= 0:
                continue
            acc = float((vals[mask] * wt).sum() / total_w)
            headline[key] = {"accuracy": acc, "sample_size": int(total_w)}

    return _render(
        request,
        "performance.html",
        {"rows": rows, "headline": headline},
    )


@router.get("/season", response_class=HTMLResponse)
def season_view(
    request: Request,
    season: int | None = Query(default=None),
) -> HTMLResponse:
    data = services.season_performance(season=season)
    return _render(
        request,
        "season.html",
        {
            "season": data["season"],
            "available_seasons": data["available_seasons"],
            "summary": data["summary"],
            "calibration": data["calibration"],
            "rolling": data["rolling"],
            "tier_breakdown": data["tier_breakdown"],
            "recent_results": data["recent_results"],
            "journal_size": data["journal_size"],
            "eos_report": data.get("eos_report"),
        },
    )


@router.get("/api/season/report")
def season_report(season: int = Query(...)) -> Response:
    """Serve the end-of-season markdown report as plain text."""
    from nfl_model.season.end_of_season import REPORTS_ROOT

    path = REPORTS_ROOT / f"end_of_season_{int(season)}" / "report.md"
    if not path.exists():
        raise HTTPException(status_code=404, detail="No end-of-season report on disk for that year.")
    body = path.read_text(encoding="utf-8")
    return Response(
        content=body,
        media_type="text/markdown; charset=utf-8",
        headers={"Cache-Control": "no-store"},
    )


@router.get("/log", response_class=HTMLResponse)
def picks_log_view(request: Request) -> HTMLResponse:
    df = services.load_picks_log()
    if not df.empty and "logged_at" in df.columns:
        df = df.sort_values("logged_at", ascending=False)

    summary: dict[str, Any] = {}
    if not df.empty:
        graded = df[df["result"].isin(["win", "loss", "push"])]
        if not graded.empty:
            summary["graded"] = int(len(graded))
            summary["wins"] = int((graded["result"] == "win").sum())
            summary["losses"] = int((graded["result"] == "loss").sum())
            summary["pushes"] = int((graded["result"] == "push").sum())
            denom = (graded["result"].isin(["win", "loss"])).sum()
            summary["win_rate"] = (
                float(summary["wins"]) / float(denom) if denom else None
            )
            summary["units"] = float(pd.to_numeric(graded["roi_units"], errors="coerce").fillna(0.0).sum())
        else:
            summary["graded"] = 0
        summary["pending"] = int((df["result"] == "pending").sum())
        summary["total"] = int(len(df))

    return _render(
        request,
        "log.html",
        {
            "rows": df.to_dict(orient="records") if not df.empty else [],
            "summary": summary,
        },
    )


@router.get("/teasers", response_class=HTMLResponse)
def teasers_view(
    request: Request,
    week: str | None = Query(default=None),
) -> HTMLResponse:
    season, wk = services.resolve_week(week)
    cards = services.teaser_cards(season, wk)
    return _render(
        request,
        "teasers.html",
        {
            "season": season, "week": wk,
            "week_label": _format_week_label(season, wk),
            "cards": cards,
        },
    )


# --------------------------------------------------------------------------- #
# HTMX partials & JSON                                                         #
# --------------------------------------------------------------------------- #


@router.get("/api/picks", response_class=HTMLResponse)
def api_picks(
    request: Request,
    week: str | None = Query(default=None),
    market: str = Query(default="spread"),
    tier: str = Query(default="all"),
) -> HTMLResponse:
    season, wk = services.resolve_week(week)
    df = services.get_predictions(season, wk, refresh=False)
    picks = services.shape_picks(df)
    picks = _filter_picks(picks, market=market, tier=tier)
    return _render(
        request,
        "partials/picks_table.html",
        {"picks": picks, "season": season, "week": wk, "active_market": market},
    )


@router.post("/api/refresh", response_class=HTMLResponse)
def api_refresh(
    request: Request,
    week: str | None = Form(default=None),
    market: str = Form(default="spread"),
    tier: str = Form(default="all"),
) -> HTMLResponse:
    season, wk = services.resolve_week(week)
    log.info("ui.refresh.start", season=season, week=wk)
    try:
        df = services.compute_predictions(season, wk, refresh=True)
        picks = services.shape_picks(df)
        picks = _filter_picks(picks, market=market, tier=tier)
        return _render(
            request,
            "partials/picks_table.html",
            {"picks": picks, "season": season, "week": wk, "active_market": market},
        )
    except Exception as exc:  # noqa: BLE001
        log.exception("ui.refresh.failed", season=season, week=wk)
        return _render(
            request,
            "partials/picks_table.html",
            {
                "picks": [], "season": season, "week": wk, "active_market": market,
                "error": (
                    f"Refresh failed: {type(exc).__name__}: {exc}. "
                    "Check logs/ for details."
                ),
            },
        )


@router.get("/api/game/{game_id}/distribution")
def api_score_distribution(game_id: str, week: str | None = None) -> JSONResponse:
    season, wk = services.resolve_week(week)
    detail = services.get_game_detail(season, wk, game_id)
    if detail is None:
        raise HTTPException(status_code=404)
    return JSONResponse(detail.score_distribution)


@router.post("/api/picks-log/add")
def api_log_pick(payload: dict[str, Any]) -> JSONResponse:
    required = {"game_id", "market", "pick", "pick_long", "model_prob"}
    missing = required - set(payload.keys())
    if missing:
        raise HTTPException(status_code=422, detail=f"Missing fields: {sorted(missing)}")
    try:
        clean = {
            "game_id": str(payload["game_id"]),
            "season": int(payload.get("season") or 0),
            "week": int(payload.get("week") or 0),
            "game_date": str(payload.get("game_date") or ""),
            "away_team": str(payload.get("away_team") or ""),
            "home_team": str(payload.get("home_team") or ""),
            "market": str(payload["market"]),
            "pick": str(payload["pick"]),
            "pick_long": str(payload["pick_long"]),
            "model_prob": float(payload["model_prob"]),
            "market_prob": (
                float(payload["market_prob"])
                if payload.get("market_prob") not in (None, "") else None
            ),
            "edge_pp": (
                float(payload["edge_pp"])
                if payload.get("edge_pp") not in (None, "") else None
            ),
            "market_line": (
                float(payload["market_line"])
                if payload.get("market_line") not in (None, "") else None
            ),
            "market_price": (
                int(payload["market_price"])
                if payload.get("market_price") not in (None, "") else -110
            ),
            "tier": str(payload.get("tier") or ""),
            "stake_units": (
                float(payload["stake_units"])
                if payload.get("stake_units") not in (None, "") else 1.0
            ),
        }
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=f"Bad payload: {exc}") from exc
    pick_id = services.append_logged_pick(clean)
    return JSONResponse({"ok": True, "pick_id": pick_id})


@router.post("/api/picks-log/delete")
def api_delete_pick(payload: dict[str, Any]) -> JSONResponse:
    pick_id = str(payload.get("pick_id") or "").strip()
    if not pick_id:
        raise HTTPException(status_code=422, detail="Missing pick_id")
    ok = services.delete_logged_pick(pick_id)
    if not ok:
        raise HTTPException(status_code=404, detail="pick_id not found")
    return JSONResponse({"ok": True})


@router.post("/api/picks-log/update")
def api_update_pick(payload: dict[str, Any]) -> JSONResponse:
    pick_id = str(payload.get("pick_id") or "").strip()
    if not pick_id:
        raise HTTPException(status_code=422, detail="Missing pick_id")
    stake: float | None = None
    if payload.get("stake_units") not in (None, ""):
        try:
            stake = float(payload["stake_units"])
        except (TypeError, ValueError) as exc:
            raise HTTPException(status_code=422, detail=f"Bad stake_units: {exc}") from exc
        if stake < 0:
            raise HTTPException(status_code=422, detail="stake_units must be >= 0")
    ok = services.update_logged_pick(pick_id, stake_units=stake)
    if not ok:
        raise HTTPException(status_code=404, detail="pick_id not found")
    return JSONResponse({"ok": True})


@router.post("/api/picks-log/clear")
def api_clear_picks() -> JSONResponse:
    n = services.clear_logged_picks()
    return JSONResponse({"ok": True, "removed": n})


# --------------------------------------------------------------------------- #
# CSV exports                                                                  #
# --------------------------------------------------------------------------- #


def _csv_response(df: pd.DataFrame, filename: str) -> Response:
    body = df.to_csv(index=False).encode("utf-8")
    return Response(
        content=body,
        media_type="text/csv; charset=utf-8",
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"',
            "Cache-Control": "no-store",
        },
    )


@router.get("/api/export/picks")
def export_picks(week: str | None = Query(default=None)) -> Response:
    season, wk = services.resolve_week(week)
    df = services.compute_predictions(season, wk, refresh=False)
    return _csv_response(df, f"picks_{season}-W{wk:02d}.csv")


@router.get("/api/export/log")
def export_log() -> Response:
    df = services.load_picks_log()
    return _csv_response(df, "picks_log.csv")


@router.get("/api/export/journal")
def export_journal(season: int | None = Query(default=None)) -> Response:
    from nfl_model.journal import grade_journal

    df = grade_journal(season=season)
    if df is None:
        df = pd.DataFrame()
    name = f"journal_{season}.csv" if season else "journal_all.csv"
    return _csv_response(df, name)


@router.get("/api/export/backtest")
def export_backtest() -> Response:
    df = services.load_backtest()
    return _csv_response(df, "backtest_results.csv")


# --------------------------------------------------------------------------- #
# Health / misc                                                                #
# --------------------------------------------------------------------------- #


@router.get("/healthz")
def healthz() -> dict[str, str]:
    return {"status": "ok", "ts": datetime.now(UTC).isoformat()}


@router.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/favicon.ico")
def favicon() -> RedirectResponse:
    return RedirectResponse(url="/static/favicon.svg")
