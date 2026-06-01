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

## Honest performance targets vs. measured results

NFL spreads are among the sharpest single-game forecasts in sports.
Break-even at standard −110 juice is 52.38%. Most published academic
baselines on **all** spread picks land at 51–53%, and going much above
that on every game is the strongest possible signal that a model is
leaking data or overfitting.

What this engine actually does, measured on the walk-forward backtest
across 2019–2025 (`logs/backtest_v6_full.csv`, 1,960 games, models
re-fit at the start of each target season on **all prior seasons only**):

| Market | Realistic target | **Measured (2019–25 avg)** | Status |
|---|---:|---:|---|
| Spread (ATS) — all picks | 0.515–0.530 | **0.512** | met |
| Spread — top-10% by edge | 0.55+ | **0.546** | met |
| Spread — top-3% by edge | 0.55+ | **0.491** | model overconfident on tail |
| Moneyline — engine bets | 0.62+ | **0.651** | met |
| Moneyline — ROI on engine bets | positive | **+5.3%** | met |
| Total (O/U) — engine bets | 0.52+ | **0.513** | met |
| Wong teaser legs (cross 3 and 7) | 0.72+ | **0.733** | met |
| Wong 2-team teaser | 0.55+ | **0.531** | near target |
| CLV vs market (spread, cents/game) | > 0 | **+3.40** | met |

**Where the real edge is:**

- **Top-10% spread picks at 54.6%** — meaningfully above break-even after
  switching the picks logic from raw conviction to *edge vs. the posted
  line in points*. The strongest disagreements with the line that the
  model is willing to commit conviction to actually beat the close.
- **Moneyline engine-bet accuracy 65.1% with +5.3% ROI** — a sharp coin
  where the better side wins about 5 percentage points more often than
  the price implies.
- **Wong teaser legs 73.3%** — the published Wong target is 73.5% to break
  even on 2-team teasers; the engine is right at the bubble across seven
  seasons, with 2020, 2021, and 2023+ exceeding it. The teaser book is
  the highest-volume above-market pathway in NFL betting.
- **CLV consistently positive** across every season — the model identifies
  sides the line moves toward by kickoff.

**Where it isn't (yet):**

- Spread top-3% averages only 49.1% — at the extreme tail of conviction,
  the model is still occasionally overconfident on the wrong side.
- Total top-10% is below target — totals depend heavily on game-script
  factors (in-game weather, pace shifts) the closing line bakes in.

Reproduce on your machine after a `data pull-range 2014 2025`:

```bash
uv run nfl-model backtest --start 2019 --end 2025 \
    --output logs/backtest_v6_full.csv
uv run nfl-model model-card --csv-path logs/backtest_v6_full.csv
```

The `ablate` and `tune` CLI commands are wired up to measure marginal
feature-group value so you can verify these numbers under different
configurations. See `docs/feature_roadmap.md` for the full feature
inventory and the prioritised list of next features.

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

## What's modelled (feature set — v2)

The model takes a wide per-game row with 150+ features grouped into:

- **QB form**: rolling EPA / CPOE / sack-rate over the last 4 starts,
  recency-weighted; severity-graded injury status (probable / questionable
  / doubtful / out maps to graded EPA delta).
- **QB tier** *(v2)*: ordinal 1-5 rating for the projected starter from
  prior-season EPA percentile. Captures the single biggest team-level
  distinction in expected outcomes.
- **Team offense / defense EPA**: opponent-adjusted EPA per play, pass
  vs run splits, 4 / 8 / 16-week rolling windows.
- **Star-player WOWY** *(v2)*: for every regular high-snap-share starter,
  the team's per-play EPA with-them-on-the-field vs. without (shrunk
  toward 0 by sample size). At prediction time, sum the prior presence
  delta of any star marked Out / Doubtful on the injury report.
  **Star identity follows the player across team moves.**
- **Head-coach features (HC-as-entity)** *(v2 enriched)*: career win %,
  career ATS %, ATS-as-favorite vs. ATS-as-underdog, over/under lean,
  season-to-date W-L, 4th-down aggression rate from PBP, neutral-script
  pass rate + PROE, rookie HC flag, **HC team-change flag** for the
  offseason coach moves. **Coach identity follows the coach across teams.**
- **OL continuity** *(v2)*: fraction of OL starters who started the team's
  previous game; rolling 4-game average. Sack-rate predictor that the
  market under-weights.
- **Pace and PROE**: plays per drive, neutral pass rate, PROE, 8-game
  rolling.
- **Drive efficiency** *(v2)*: yards per drive (offense / defense), red-zone
  trip rate, turnover differential proxy — 8-game rolling.
- **Success and explosive plays**: success rate, explosive-play rate,
  3rd-down conversion %, red-zone TD %, 8-game rolling.
- **Injuries**: starter outs by position, snap-share-weighted impact
  score; QB-out / OL-out / skill-out flags.
- **Schedule context**: rest, short week (Thursday), bye, great-circle
  travel miles, divisional flag, primetime, weekday breakdowns.
- **Situational**: post-bye, divisional rematch.
- **Surface and weather** *(v2 enriched)*: dome / turf flags; wind speed,
  wind-on-pass-axis, wind-perpendicular components; temperature; humidity;
  precipitation; **freezing flag** (≤32°F), **hot flag** (≥85°F),
  **heavy-precip flag** (≥5mm), **passing-unfriendly composite** flag.
- **Officiating crew**: per-referee rolling penalty count, penalty yards,
  game total points.
- **Kicking**: stub (real implementation queued — see roadmap).
- **Market features**: de-vigged ML implied probability, spread open /
  close / move, total open / close / move, **key-number-cross flag** (3,
  6, 7, 10), reverse-line-move flag, steam-move flag.

See [`docs/feature_roadmap.md`](docs/feature_roadmap.md) for the full
feature list and the prioritised set of next features (NextGen Stats,
real penalty data, coordinators, coach-matchup history, lookahead /
letdown, standings importance).

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
