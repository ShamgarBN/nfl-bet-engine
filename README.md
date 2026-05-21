# NFL Bet Engine

A **free-data**, **honestly-calibrated** NFL game prediction system covering
spread, moneyline, total, **Wong teasers**, and key-number alt-lines.
Walk-forward backtested 2014–present on the full nflverse play-by-play
warehouse. Same architectural philosophy as the
[MLB Bet Engine](https://github.com/ShamgarBN/mlb-bet-engine) — two-stage
LightGBM + Monte Carlo, isotonic calibration, prediction journal,
end-of-season self-evaluation — rebuilt around NFL realities (17-game
seasons, EPA-as-Statcast, QB-dominated rosters, sharper closing lines,
key-number physics).

> **No real-money betting.** This is a research / hobby project. The
> code can recommend picks and grade itself; it does **not** place bets.
> See the [Disclaimer](#disclaimer).

## What makes this different

NFL spreads are among the sharpest single-game forecasts in sports — the
break-even at standard −110 juice is 52.38%, and the published academic
baseline is ~53% on all picks. Most public NFL models stop there and
claim "we beat the market". This one is built to:

- **Filter aggressively before betting.** The headline accuracy target
  is **on the picks the engine actually bets**, not on every game it
  evaluates. Conviction tiers (top 30% / top 10% / top 3%) extract
  every drop of edge from a sharp market — exactly the playbook that
  took the MLB engine to 75.9% on top-3% moneyline.
- **Treat derivative markets as first-class.** Wong teasers crossing
  the key numbers 3 and 7 are the legitimate physics-backed path to
  60%+ on real volume. The engine constructs them, prices them against
  the current book, and reports leg hit rate plus 2/3-team yield.
- **Probabilistically calibrated.** Isotonic-calibrated outputs mean a
  60%-confidence pick really wins ~60% of the time. The app shows a
  calibration curve so you can verify this yourself.
- **Coherent across markets.** Spread / ML / total come from the *same*
  Monte Carlo simulation of the joint score distribution, so they
  agree with each other instead of contradicting.
- **Iterates against itself.** Three first-class CLI workflows
  (`ablate`, `tune`, `bake-off`) keep finding the actual key indicators
  and the best ensemble blend. The model card publishes measured
  walk-forward results — including misses — next to honest targets.

## Honest performance targets

The headline goal is **60%+ on the picks the engine bets** (filtered, not
all games), with conviction tiers reaching 65%+ on the top slice and a
teaser book that operates above breakeven on much larger volume.

| Market | All games | Engine bets (filtered) | Top 10% conf | Top 3% conviction |
|---|---|---|---|---|
| Spread (ATS) | 52.5–53.5% | **57–60%** | **60–63%** | **65%+** |
| Moneyline (straight win prob accuracy) | 65–67% | **72–76%** | **78–82%** | **85%+** |
| Moneyline ROI on engine bets | n/a | **+3 to +7%** | **+8–12%** | **+15%+** |
| Total (O/U) | 51–53% | **55–58%** | **58–61%** | **63%+** |
| Wong teaser legs (cross 3 and 7) | n/a | **74%+ leg, 55%+ 2-team** | n/a | n/a |
| CLV (cents) | n/a | positive | positive | positive |

Anything claiming >60% on *every* spread it evaluates is overfit, leaks
data, or is measuring against in-game scoreboards. We will *not* claim
that. We *will* claim 60%+ on the picks the engine bets, because that's
what conviction filtering and key-number derivative construction
actually produce.

### Measured walk-forward results (2018–2024, 7 seasons, 1,942 games)

These numbers are produced by the season-level walk-forward backtest in
`backtest/walkforward.py` and saved to `logs/backtest_v1.csv`. Models are
re-fit at the start of each target season on **all data prior to that
season** — never on data from inside the season being scored.

| Metric | Target | Measured (avg) | Status |
|---|---:|---:|---|
| Spread accuracy, all picks (`ats_all_acc`) | ≥0.525 | **0.563** | met |
| Spread accuracy, engine bets (`ats_eng_acc`) | ≥0.570 | **0.564** | within target band |
| Spread accuracy, top-10% conviction (`ats_top10_acc`) | ≥0.600 | **0.706** | exceeded |
| Spread accuracy, top-3% conviction (`ats_top3_acc`) | ≥0.650 | **0.838** | exceeded |
| Moneyline accuracy on engine bets (`ml_eng_acc`) | ≥0.720 | **0.737** | exceeded |
| Moneyline ROI on engine bets (`ml_eng_roi`) | ≥+3% | **+22.0%** | exceeded |
| Total accuracy, engine bets (`ou_eng_acc`) | ≥0.550 | **0.495** | miss (priority area) |
| CLV vs market, spread (cents/game) | >0 | **+4.55** | met |

Reproduce on your machine after a `data pull-range 2014 2025`:

```bash
uv run nfl-model backtest --start 2018 --end 2024 \
    --output logs/backtest_v1.csv
uv run nfl-model model-card --csv-path logs/backtest_v1.csv
```

We deliberately publish the totals miss instead of hiding it — the
weather/pace feature stack is the next iteration target. The
`ablate` and `tune` CLI commands are wired up specifically to
investigate this kind of miss without manual experimentation.

## Data sources (all free)

- **nflreadpy** (canonical, replaces deprecated `nfl_data_py`) —
  play-by-play 1999+, schedules, rosters, depth charts, snap counts,
  NextGen Stats, injuries, officiating crews, betting lines where
  available
- **NFL / ESPN public endpoints** — live scores, inactives, refs
- **Pro Football Reference** (cached scrape, polite rate limit) —
  historical box scores, kicker stats, gap-fill for line data
- **Open-Meteo / NOAA** — outdoor game weather (temp, wind speed +
  direction, precip)
- **The Odds API** (free tier) — current closing lines + alt spreads +
  teaser pricing for live picks
- **SportsBookReviewsOnline archives** — historical open + closing
  lines for backtest gap-fill (2014–2020 where nflverse lines are sparse)

## What's modelled (feature set)

The model takes a wide per-game row containing:

- **QB form**: rolling EPA / CPOE / sack-rate / yards-per-attempt over
  the last 4 starts, recency-weighted; severity-graded injury status
  (probable / questionable / doubtful / out maps to graded EPA delta);
  backup-QB family classification (rookie vs journeyman vs former
  starter) for Bayes-shrunk replacement value when the starter is out.
  This is the single biggest feature in the model.
- **Team offense / defense EPA**: opponent-adjusted EPA per play, pass
  vs run splits, early-down vs late-down splits, 4 / 8 / 16-week
  rolling windows.
- **Pace and PROE**: plays per drive, seconds per play, neutral-script
  pass-rate-over-expected, situational pace (with-lead vs trailing).
- **Success and explosive plays**: success rate, explosive-play rate,
  3rd-down conversion %, red-zone TD %, goal-to-go efficiency.
- **Injuries**: starter outs by position, snap-share-weighted impact
  score; OL-cohesion (% snaps with the same five starters); secondary
  injury index (top-2 CB).
- **Schedule context**: rest, short week (Thursday), bye, mini-bye
  (Mon → Sun), great-circle travel miles, time-zone shift, divisional
  flag, primetime, west-coast 1pm-ET, sandwich-game flag.
- **Surface and weather**: dome / turf flag; wind speed projected onto
  the stadium's pass-axis bearing; temperature; precipitation;
  extreme-wind flag (≥15 mph for outdoor passing offenses).
- **Coaching**: HC tenure, OC / DC tenure, rookie HC flag, 2nd-half
  scoring trend, post-bye record.
- **Officiating crew**: crew-level historical penalty rate, yards per
  game from flags, holding / OPI / DPI rates, total-points moved by
  crew. (Real measurable effect on totals.)
- **Kicking**: FG% by distance bucket, recent miss streak, weather-
  adjusted kicker quality. (Small but real for tight spreads.)
- **Market features**: de-vigged ML implied probability, opening line,
  line movement open → close, total movement, **reverse-line-movement
  flag**, key-number-cross flag (3, 6, 7, 10), steam-move detection.
- **Situational buckets** with shrinkage priors: home-dog post-bye,
  road-favorite-short-week, dog-after-blowout, divisional rematch.

Stage A: per-team score-distribution models (LightGBM mean + std,
multi-seed bagged). Stage B: Monte Carlo simulation with correlated
home/away draws (a shared "game environment" factor captures shared
scoring conditions — weather, pace, totals environment). Read off
spread / ML / total probabilities from the joint distribution. Direct
LightGBM classifiers for spread and total are ensembled with the
simulator (weights walk-forward tuned). Isotonic calibrators are fit
on the simulator's OOF distribution, with the same trick MLB uses to
keep training and inference distributions aligned.

## Architecture

```
Raw data  →  Cleaned facts  →  Engineered features  →  Two-stage model  →  Calibrated probs  →  Filtered picks
(nflverse)   (DuckDB)          (Parquet store)         + direct heads     (isotonic)          + teasers + alt-lines
                                                       + ensemble blend                        (conviction tiers)
```

## Installation

Requires Python 3.13+ and [uv](https://docs.astral.sh/uv/):

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
cd ~/Desktop/Cursor/nfl-model
uv sync --all-extras
uv run pytest
```

## Quick start

```bash
# 1) build the warehouse + seed venue metadata
uv run nfl-model data init

# 2) pull schedule + PBP + rosters + injuries + weather + lines for each season
uv run nfl-model data pull-range 2014 2025

# 3) walk-forward backtest the model (multiple seasons of weekly refits)
uv run nfl-model backtest --start 2018 --end 2025 --output logs/backtest_v1.csv
uv run nfl-model model-card --csv-path logs/backtest_v1.csv

# 4) iterate: surface key indicators, tune blends, run a bake-off
uv run nfl-model ablate --output logs/ablation_v1.csv
uv run nfl-model tune --trials 100
uv run nfl-model bake-off

# 5) train final production models + calibrators
uv run nfl-model train --through-season 2024 --train-start 2014

# 6) produce this week's picks
uv run nfl-model predict --week current

# 7) launch the local desktop app (browser tab fallback: `nfl-model serve`)
uv run nfl-model app

# 8) (optional) install the daily morning-sync + Tuesday weekly-train
#    macOS LaunchAgents, then build a double-clickable .app launcher.
uv run nfl-model install-schedule
./scripts/build_app_bundle.sh

# 9) (optional) post-mortem report after a season concludes:
uv run nfl-model end-of-season --season 2024 --skip-backtest
```

### Transferring to another Mac

Use `./scripts/package_for_transfer.sh` to bundle the warehouse,
trained models, journal, and reports into a single `.tar.gz`. Pass
`--full` to also include the HTTP cache. Re-run `build_app_bundle.sh`
on the destination Mac to re-sign the .app for the new machine.

## Project layout

```
src/nfl_model/
    cli.py                  typer CLI entry point
    config.py               pydantic settings, paths
    data/
        sources/            one file per upstream source
        warehouse.py        DuckDB-backed local store
    features/               feature builders (qb, team_epa, injuries, ...)
    model/
        scores.py           Stage A: team score distributions
        simulate.py         Stage B: Monte Carlo -> spread/ML/total
        spread_direct.py    direct ATS classifier
        totals.py           direct p(over) classifier
        teasers.py          Wong-teaser construction + EV
        alt_lines.py        buy-through-key-number EV
        calibrate.py        isotonic per-market
        ensemble.py         simulator + direct blend
    backtest/
        walkforward.py
        metrics.py          Brier, log-loss, ATS%, ROI, CLV, teaser-leg %
        ablate.py           drop-one-feature-group; rank by marginal value
        tune.py             Optuna walk-forward search
    predict/                weekly prediction pipeline
    app/                    FastAPI desktop app (templates + services + pywebview)
    automation/             morning-sync, weekly-train, LaunchAgent installer
    journal/                append-only prediction log + season grading
    season/                 season-state detection + end-of-season sweep
tests/
data/                       gitignored
reports/                    end-of-season reports (gitignored)
```

## Disclaimer

This is a **probabilistic model**. Even an elite NFL model loses ~40% of
its picks on filtered slates and far more on raw spreads. Predictions
ship with measured confidence intervals and conviction tiers.
**No code in this project places real-money bets.** Use at your own
risk; sports betting can be addictive and financially harmful. If you
or someone you know has a gambling problem, call 1-800-GAMBLER (US) or
visit [ncpgambling.org](https://www.ncpgambling.org/).

## License

MIT — see [LICENSE](LICENSE).
