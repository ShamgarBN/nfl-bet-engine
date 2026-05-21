"""Typer-based command-line interface.

Exposes the project's pipelines:

    uv run nfl-model data init
    uv run nfl-model data pull --season 2024
    uv run nfl-model data pull-range 2014 2025
    uv run nfl-model train --through-season 2024
    uv run nfl-model backtest --start 2018 --end 2025
    uv run nfl-model predict --week current
    uv run nfl-model ablate
    uv run nfl-model tune
    uv run nfl-model bake-off
    uv run nfl-model app
"""

from __future__ import annotations

from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from nfl_model.config import settings
from nfl_model.logging import configure_logging, get_logger

app = typer.Typer(no_args_is_help=True, help="NFL prediction & betting model CLI")
data_app = typer.Typer(no_args_is_help=True, help="Data ingestion commands")
app.add_typer(data_app, name="data")

console = Console()
log = get_logger("cli")


@data_app.command("init")
def data_init() -> None:
    """Initialize the warehouse schema and seed venue metadata."""
    configure_logging()
    from nfl_model.data.pipeline import ensure_venues
    from nfl_model.data.warehouse import init_schema, list_tables

    init_schema()
    n = ensure_venues()
    console.print(f"[green]Warehouse ready at[/green] {settings.warehouse_path}")
    console.print(f"[green]Tables:[/green] {', '.join(list_tables())}")
    console.print(f"[green]Venues seeded:[/green] {n}")


@data_app.command("pull")
def data_pull(
    season: Annotated[int, typer.Option(help="Season year to pull")] = 2024,
    no_weather: Annotated[bool, typer.Option(help="Skip weather pulls")] = False,
    no_lines: Annotated[bool, typer.Option(help="Skip betting line pulls")] = False,
) -> None:
    """Pull all sources for one season into the warehouse."""
    configure_logging()
    from nfl_model.data.pipeline import pull_season

    counts = pull_season(season, with_weather=not no_weather, with_lines=not no_lines)
    table = Table(title=f"Season {season} ingest counts")
    table.add_column("source")
    table.add_column("rows", justify="right")
    for k, v in counts.items():
        table.add_row(str(k), str(v))
    console.print(table)


@data_app.command("pull-range")
def data_pull_range(
    start: Annotated[int, typer.Argument(help="First season")],
    end: Annotated[int, typer.Argument(help="Last season (inclusive)")],
    no_weather: Annotated[bool, typer.Option(help="Skip weather pulls")] = False,
    no_lines: Annotated[bool, typer.Option(help="Skip betting line pulls")] = False,
) -> None:
    """Pull all sources across [start, end] (inclusive)."""
    configure_logging()
    from nfl_model.data.pipeline import pull_season

    for season in range(start, end + 1):
        try:
            pull_season(
                season,
                with_weather=not no_weather,
                with_lines=not no_lines,
            )
        except Exception as exc:  # noqa: BLE001
            log.error("season.failed", season=season, error=str(exc))


@data_app.command("status")
def data_status() -> None:
    """Show row counts per warehouse table."""
    configure_logging()
    from nfl_model.data.warehouse import list_tables, table_row_count

    table = Table(title="Warehouse status")
    table.add_column("table")
    table.add_column("rows", justify="right")
    for t in list_tables():
        table.add_row(t, f"{table_row_count(t):,}")
    console.print(table)


@app.command()
def features(
    start: Annotated[int, typer.Option(help="First season")] = 2014,
    end: Annotated[int, typer.Option(help="Last season (inclusive)")] = 2025,
) -> None:
    """Build the engineered feature table for [start, end]."""
    configure_logging()
    from nfl_model.features.assemble import build_features_table

    df = build_features_table(start, end)
    console.print(f"[green]Built features for {len(df):,} games[/green]")


@app.command()
def train(
    through_season: Annotated[
        int, typer.Option(help="Train on data up to and including this season")
    ] = 2024,
    train_start: Annotated[int, typer.Option(help="Earliest season to include")] = 2014,
) -> None:
    """Train production models + calibrators on [train_start, through_season]."""
    configure_logging()
    from nfl_model.model.training import train_production_models

    summary = train_production_models(train_start=train_start, through_season=through_season)
    console.print(f"[green]Training complete:[/green] {summary}")


@app.command()
def backtest(
    start: Annotated[int, typer.Option(help="First target season")] = 2018,
    end: Annotated[int, typer.Option(help="Last target season")] = 2025,
    output: Annotated[str, typer.Option(help="Output CSV path")] = "logs/backtest_v1.csv",
) -> None:
    """Run weekly walk-forward backtest and write per-season metrics to CSV."""
    configure_logging()
    from nfl_model.backtest.walkforward import run_walkforward

    results = run_walkforward(start, end)
    if results.empty:
        console.print("[red]Backtest produced no results.[/red]")
        raise typer.Exit(code=1)

    results.to_csv(output, index=False)
    console.print(f"[green]Wrote[/green] {output}")

    table = Table(title=f"Backtest {start}-{end}")
    for col in [
        "season",
        "n_games",
        "ats_all_acc",
        "ats_eng_acc",
        "ats_top10_acc",
        "ats_top3_acc",
        "ml_eng_acc",
        "ml_eng_roi",
        "ou_eng_acc",
        "teaser_leg_pct",
        "clv_spread",
    ]:
        table.add_column(col)
    for _, row in results.iterrows():
        table.add_row(
            str(int(row["season"])),
            str(int(row["n_games"])),
            f"{row.get('ats_all_acc', float('nan')):.3f}",
            f"{row.get('ats_eng_acc', float('nan')):.3f}",
            f"{row.get('ats_top10_acc', float('nan')):.3f}",
            f"{row.get('ats_top3_acc', float('nan')):.3f}",
            f"{row.get('ml_eng_acc', float('nan')):.3f}",
            f"{row.get('ml_eng_roi', float('nan')):+.3f}",
            f"{row.get('ou_eng_acc', float('nan')):.3f}",
            f"{row.get('teaser_leg_pct', float('nan')):.3f}",
            f"{row.get('clv_spread', float('nan')):+.2f}",
        )
    console.print(table)


@app.command("model-card")
def model_card(
    csv_path: Annotated[
        str, typer.Option(help="Backtest CSV path to summarize")
    ] = "logs/backtest_v1.csv",
) -> None:
    """Print an honest performance summary suitable for a model card."""
    import pandas as pd

    df = pd.read_csv(csv_path)
    console.print(f"\n[bold]Model Card -- backtest from {csv_path}[/bold]")

    metrics = [
        "ats_all_acc", "ats_eng_acc", "ats_top10_acc", "ats_top3_acc",
        "ml_eng_acc", "ml_eng_roi",
        "ou_all_acc", "ou_eng_acc", "ou_top10_acc",
        "teaser_leg_pct", "teaser_2team_pct",
        "clv_spread",
    ]
    agg = df[[c for c in metrics if c in df.columns]].mean()

    targets: dict[str, tuple[float, str]] = {
        "ats_all_acc":     (0.525, "Spread accuracy, all picks"),
        "ats_eng_acc":     (0.570, "Spread accuracy, engine bets (filtered)"),
        "ats_top10_acc":   (0.600, "Spread accuracy, top-10% confidence"),
        "ats_top3_acc":    (0.650, "Spread accuracy, top-3% conviction"),
        "ml_eng_acc":      (0.720, "Moneyline accuracy on engine bets"),
        "ml_eng_roi":      (0.030, "Moneyline ROI on engine bets"),
        "ou_eng_acc":      (0.550, "Total accuracy, engine bets"),
        "ou_top10_acc":    (0.580, "Total accuracy, top-10% confidence"),
        "teaser_leg_pct":  (0.740, "Wong teaser leg hit rate (cross 3 and 7)"),
        "teaser_2team_pct":(0.550, "2-team teaser hit rate"),
        "clv_spread":      (0.0,   "Closing-line value vs market (cents/game)"),
    }

    table = Table(title=f"Backtest aggregate across {len(df)} seasons")
    table.add_column("Metric")
    table.add_column("Description")
    table.add_column("Target", justify="right")
    table.add_column("Measured", justify="right")
    table.add_column("Status")
    for metric, (target, desc) in targets.items():
        val = float(agg.get(metric, float("nan")))
        if val != val:
            status, value_str = "[grey]n/a[/grey]", "n/a"
        else:
            value_str = (
                f"{val:+.2f}" if metric == "clv_spread"
                else f"{val:.3f}"
            )
            status = (
                "[green]met[/green]" if val >= target else "[red]below[/red]"
            )
        target_str = (
            f"{target:+.2f}" if metric == "clv_spread" else f"{target:.3f}"
        )
        table.add_row(metric, desc, target_str, value_str, status)
    console.print(table)


@app.command()
def predict(
    week: Annotated[
        str, typer.Option(help="Target week: 'current', or 'YYYY-W' (e.g. 2026-1)")
    ] = "current",
    refresh: Annotated[bool, typer.Option(help="Refresh data first")] = True,
) -> None:
    """Produce calibrated picks + teaser cards for ``week``."""
    configure_logging()
    from nfl_model.predict.weekly import predict_for_week, write_picks_csv

    picks = predict_for_week(week, refresh_data=refresh)
    if picks.empty:
        console.print("[yellow]No games to predict for that week.[/yellow]")
        return
    path = write_picks_csv(picks, week)
    console.print(f"[green]Wrote picks to[/green] {path}")


@app.command()
def ablate(
    start: Annotated[int, typer.Option(help="First target season")] = 2018,
    end: Annotated[int, typer.Option(help="Last target season")] = 2025,
    output: Annotated[str, typer.Option(help="Output CSV path")] = "logs/ablation_v1.csv",
) -> None:
    """Drop one feature group at a time; rank by marginal value."""
    configure_logging()
    from nfl_model.backtest.ablate import run_ablation

    results = run_ablation(start, end)
    results.to_csv(output, index=False)
    console.print(f"[green]Wrote ablation report to[/green] {output}")
    console.print(results.to_string(index=False))


@app.command()
def tune(
    trials: Annotated[int, typer.Option(help="Optuna trials")] = 50,
    start: Annotated[int, typer.Option(help="First target season")] = 2018,
    end: Annotated[int, typer.Option(help="Last target season")] = 2024,
) -> None:
    """Optuna walk-forward search over hyperparameters + ensemble blend."""
    configure_logging()
    from nfl_model.backtest.tune import run_tuning

    best = run_tuning(start, end, n_trials=trials)
    console.print(f"[green]Best parameters:[/green] {best}")


@app.command("bake-off")
def bake_off(
    start: Annotated[int, typer.Option(help="First target season")] = 2018,
    end: Annotated[int, typer.Option(help="Last target season")] = 2024,
) -> None:
    """Run candidate model configs in parallel; auto-promote the winner per market."""
    configure_logging()
    from nfl_model.backtest.tune import run_bakeoff

    winners = run_bakeoff(start, end)
    table = Table(title="Bake-off results")
    table.add_column("Market")
    table.add_column("Winner")
    table.add_column("Metric", justify="right")
    for market, (name, score) in winners.items():
        table.add_row(market, name, f"{score:.4f}")
    console.print(table)


@app.command(name="app")
def app_cmd(
    width: Annotated[int, typer.Option(help="Window width in pixels")] = 1280,
    height: Annotated[int, typer.Option(help="Window height in pixels")] = 860,
) -> None:
    """Launch the NFL Forecast app in a native macOS window."""
    configure_logging()
    try:
        from nfl_model.app.desktop import launch_native_window
    except ImportError as exc:
        console.print(
            "[red]pywebview is not installed.[/red] "
            "Install it with `uv sync --extra desktop` or use `nfl-model serve` instead."
        )
        raise typer.Exit(code=1) from exc

    try:
        launch_native_window(width=width, height=height)
    except RuntimeError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from exc


@app.command()
def serve(
    host: Annotated[str, typer.Option(help="Bind address (local-only by default)")] = "127.0.0.1",
    port: Annotated[int, typer.Option(help="HTTP port")] = 8766,
    open_browser: Annotated[bool, typer.Option(help="Open the app in your default browser")] = True,
    reload: Annotated[bool, typer.Option(help="Auto-reload on code changes (dev only)")] = False,
) -> None:
    """Launch the local NFL forecast app (browser tab variant)."""
    configure_logging()
    import threading
    import time
    import webbrowser

    import uvicorn

    if host not in {"127.0.0.1", "localhost"}:
        console.print(
            f"[red]Refusing to bind to {host!r}. The desktop app is local-only.[/red]"
        )
        raise typer.Exit(code=2)

    url = f"http://{host}:{port}"
    console.print(f"[green]NFL Forecast running at[/green] {url}")
    console.print("Press Ctrl+C to stop.")

    if open_browser:
        def _open() -> None:
            time.sleep(0.8)
            try:
                webbrowser.open(url, new=2)
            except Exception:  # noqa: BLE001
                pass

        threading.Thread(target=_open, daemon=True).start()

    uvicorn.run(
        "nfl_model.app.main:app",
        host=host,
        port=port,
        log_level="info",
        reload=reload,
        access_log=False,
    )


@app.command("morning-sync")
def morning_sync_cmd() -> None:
    """Refresh the current week's slate (idempotent; safe to repeat)."""
    configure_logging()
    from nfl_model.automation import run_morning_sync

    counts = run_morning_sync()
    table = Table(title="Morning sync", show_header=False)
    for k, v in counts.items():
        table.add_row(str(k), str(v) if not isinstance(v, dict) else "(see logs)")
    console.print(table)


@app.command("weekly-train")
def weekly_train_cmd(
    through_season: Annotated[
        int | None, typer.Option(help="Train through this season (default: current)")
    ] = None,
) -> None:
    """Refit production models against the freshest data (with regression guard)."""
    configure_logging()
    from nfl_model.automation import run_weekly_train

    result = run_weekly_train(through_season=through_season)
    table = Table(title="Weekly train", show_header=False)
    for k, v in result.items():
        table.add_row(str(k), str(v) if not isinstance(v, dict) else "(see logs)")
    console.print(table)


@app.command("install-schedule")
def install_schedule_cmd() -> None:
    """Install macOS LaunchAgents for daily sync + weekly retrain."""
    configure_logging()
    from nfl_model.automation import install_scheduler

    paths = install_scheduler()
    for label, path in paths.items():
        console.print(f"[green]{label}[/green] -> {path}")


@app.command("uninstall-schedule")
def uninstall_schedule_cmd() -> None:
    """Remove the LaunchAgents installed by ``install-schedule``."""
    configure_logging()
    from nfl_model.automation import uninstall_scheduler

    uninstall_scheduler()
    console.print("[green]LaunchAgents uninstalled.[/green]")


@app.command("end-of-season")
def end_of_season_cmd(
    season: Annotated[int, typer.Option(help="Season year (e.g. 2025)")],
    train_start: Annotated[
        int | None, typer.Option(help="Earliest training season")
    ] = None,
    force: Annotated[bool, typer.Option(help="Skip the 'season-over?' check")] = False,
    skip_backtest: Annotated[bool, typer.Option(help="Skip the slow walk-forward")] = False,
) -> None:
    """Generate the end-of-season post-mortem report."""
    configure_logging()
    from nfl_model.season import run_end_of_season_sweep

    report = run_end_of_season_sweep(
        season=season,
        train_start=train_start,
        force=force,
        skip_backtest=skip_backtest,
    )
    console.print(f"[green]Report written:[/green] {report.report_md_path}")
    console.print(f"[green]Output dir:[/green] {report.output_dir}")


@app.command("season-status")
def season_status_cmd() -> None:
    """Show the current season's state (in-progress / ended / off-season)."""
    configure_logging()
    from nfl_model.season import detect_season_state

    status = detect_season_state()
    table = Table(title=f"{status.season} status", show_header=False)
    table.add_row("State", status.state)
    table.add_row("Finalized games", str(status.finalized_games))
    table.add_row("Description", status.description)
    console.print(table)


@app.command("journal-grade")
def journal_grade_cmd(
    season: Annotated[
        int | None, typer.Option(help="Filter to a single season (default: most recent)")
    ] = None,
) -> None:
    """Grade the prediction journal and print per-market summaries."""
    configure_logging()
    from nfl_model.journal import grade_journal, season_summary

    graded = grade_journal(season=season)
    if graded is None or graded.empty:
        console.print("[yellow]No graded predictions yet.[/yellow]")
        return

    summaries = season_summary(graded)
    if not summaries:
        console.print("[yellow]Journal exists but no finalized outcomes joined.[/yellow]")
        return

    table = Table(title="Prediction journal summary")
    table.add_column("Season")
    table.add_column("Market")
    table.add_column("N", justify="right")
    table.add_column("W-L-P", justify="right")
    table.add_column("Win %", justify="right")
    table.add_column("Units", justify="right")
    for row in summaries:
        wr = row.win_rate
        wr_s = f"{wr*100:.1f}%" if wr is not None and wr == wr else "—"
        wlp = f"{row.wins}-{row.losses}"
        if row.pushes:
            wlp += f"-{row.pushes}"
        table.add_row(
            str(row.season), row.market, str(row.n), wlp, wr_s,
            f"{row.roi_units:+.2f}",
        )
    console.print(table)


def main() -> None:
    """Entry-point alias for [project.scripts]."""
    app()


if __name__ == "__main__":
    main()
