# NFL Bet Engine — Feature Roadmap

What's already in the model after the v2 expansion, plus the next-best
candidates ranked by expected marginal value.

## Currently implemented

| Group | Highlights | Source |
|---|---|---|
| **Team EPA** | offense/defense EPA, pass/rush splits, success/explosive — 4/8/16-game rolling | `features/team_epa.py` |
| **QB form** | rolling EPA/CPOE/sack rate of probable starter; severity-graded injury delta | `features/qb_form.py` |
| **QB tier** *(v2)* | ordinal 1-5 from prior-season EPA percentile | `features/qb_tier.py` |
| **Injuries** | snap-share-weighted count of starters out; QB-out / OL-out / skill-out flags | `features/injuries.py` |
| **Star players WOWY** *(v2)* | per-player presence delta (team EPA with/without); summed for current outs | `features/star_players.py` |
| **OL continuity** *(v2)* | fraction of OL starters who started last week; rolling 4-game | `features/ol_continuity.py` |
| **Pace / PROE** | rolling 8-game plays per drive, neutral pass rate, PROE | `features/pace_proe.py` |
| **Drive efficiency** *(v2)* | yards per drive (off/def), turnover diff proxy, red-zone trip rate | `features/drive_eff.py` |
| **Success & explosive** | rolling success rate, explosive rate, 3rd-down %, RZ TD % | `features/success_explosive.py` |
| **Schedule context** | rest days, short week, long rest, travel miles, primetime, divisional, weekday | `features/schedule_context.py` |
| **Situational** | post-bye, divisional rematch | `features/situational.py` |
| **Coaching (HC-as-entity)** *(v2 enriched)* | career win %, ATS %, ATS-as-favorite, ATS-as-dog, OU lean, 4th-down aggression, neutral pass / PROE, rookie HC, team-change flag | `features/coaching.py` |
| **Surface & weather** | dome / turf flags, temp, humidity, wind speed, wind-on-pass-axis, wind-perpendicular, freezing / hot / heavy-precip / passing-unfriendly flags | `features/surface_weather.py` |
| **Officiating** | per-referee rolling penalty count, penalty yards, game total points | `features/officiating.py` |
| **Kicking** | stub | `features/kicking.py` |
| **Market** | de-vigged ML implied prob, spread open/close/move, total open/close/move, key-number-cross flag, reverse-line-move flag, steam-move flag | `features/market.py` |

## High-value next features (ranked)

### 1. Offensive / defensive coordinator features — `features/coordinators.py`
Same shape as the HC features but for OC + DC. Coordinator changes
mid-season or in the offseason are a strong predictor of regression
or improvement. nflverse doesn't ship coordinators directly but
nflreadpy's roster + injuries feeds plus Pro Football Reference's
coaching page can be scraped.

### 2. Real penalty data — extend `features/officiating.py`
Today we have only the head referee + game-total points; PFR ships
per-game penalty counts and yards by team. Replacing the proxy with
real numbers gives meaningful totals movement, since high-penalty
crews extend drives.

### 3. NextGen Stats — `features/nextgen.py`
nflreadpy exposes `load_nextgen_stats()` with metrics like air yards
per attempt, time-to-throw, separation, defensive pressure %. These
are the closest thing to "advanced" stats in the public data and the
market under-weights them. Highest single-feature lift for QB form.

### 4. Coach matchup history — `features/coach_matchup.py`
For each (home_coach, away_coach) pair, the head-to-head ATS and
straight-up record going into this game. Some coaches "own" others;
e.g., Belichick vs. various coordinators historically.

### 5. Turnover regression — `features/turnover_regression.py`
Teams with extreme positive turnover differential YTD tend to regress.
Compute the gap between the team's actual turnover diff and the
league-mean, weighted by sample size; flip the sign for the model so
"regression candidate" is positive.

### 6. Lookahead / letdown — `features/lookahead.py`
Did either team play a particularly hard / easy opponent last week?
Are they playing one *next* week? Both correlate with motivation /
preparation effects that are documented in the literature.

### 7. Standings importance — `features/standings.py`
Teams effectively eliminated from playoff contention play differently
in the second half. Division leaders with locked-up seeds may rest
starters. Build a "playoff implication" score from prior W-L.

### 8. Travel direction / time-zone shift — extend `features/schedule_context.py`
We have travel miles. Add: east-to-west vs. west-to-east (Pacific time
teams playing 1pm ET get up at 7am body time), and gross time-zone
delta. Documented small but real effect on ATS.

### 9. Special teams — replace `features/kicking.py` stub
Per-team rolling FG % by distance bucket, punt yards/return, kickoff
return value. Special teams contribute ~10% of variance in tight
games.

### 10. Sharp-side line movement velocity — extend `features/market.py`
Time between opening line and last-recorded line, plus the size of
movement vs. typical. Big late moves on lower handle (the "smart money"
signal) historically correlate with cover rates above the close.

### 11. Trench play differential — `features/trenches.py`
Team's PFF-style win rate at OL vs. opposing DL is one of the few
features that consistently predicts ATS in the academic literature.
Approximation: roll the team's sack rate allowed vs. league mean,
crossed with opposing team's sack rate generated vs. league mean.
Free data exists for raw sacks; the "win rate" version needs PFR
advanced-stats scrape.

### 12. Public bet vs. handle split — `features/public_split.py`
When 70% of bets are on one side but only 40% of the *money* is on
that side, sharp money is on the unpopular side. Requires a paid
data source (e.g., Action Network) — not free, so deferred.

### 13. Pace / fatigue under temperature stress — extend `features/surface_weather.py`
Cross feature: outdoor team rolling pace × temperature percentile.
Hot games slow pace late; cold games slow passing. Already partially
captured by `sw_passing_unfriendly` but a smooth interaction term
could lift totals.

## Visual design (deferred — for the final polish pass)

The app currently inherits the MLB Bet Engine's look-and-feel (FastAPI +
HTMX + Tailwind + dark theme). Before the engine ships for the
2026/2027 season the visual identity needs to be **distinctly NFL-flavoured**,
not a reskin of the baseball app. Open to direction; some options to
explore:

- **Field-as-canvas layout**: the dashboard's hero strip styled as a
  football field (yard lines, hash marks, end zones in the team's
  primary colour), with the matchup, spread, and pick rendered as
  field-position annotations.
- **Helmet / down-marker iconography**: confidence tiers shown as down
  markers (1st = top-3%, 2nd = top-10%, etc.) instead of star ratings.
- **Scoreboard chrome**: per-game cards styled like an NFL broadcast
  scorebug — team colour stripes, abbreviations in the broadcast font,
  game clock / quarter glyphs in the corner.
- **Team-colour theming**: cards/pages pick up the favourite's primary
  colour as an accent so a slate visually reads like a Sunday lineup.
- **Football-specific typography**: a condensed sans (like the broadcast
  graphics style — "FOX Sports 31" lookalikes) for headlines, a clean
  monospace for numbers.
- **Animated coin-flip / play-clock loaders** instead of generic spinners.

Constraint: keep it fast (no heavy JS frameworks, no asset blowup), keep
all data still in the table views, and keep the dark-mode default. The
look should *immediately* read as "this is the NFL tool" within a second
of the page rendering.

This is the **last** thing to nail down — chase the predictive accuracy
first, then make it look the part.

## Lower priority / experimental

- **Coach matchup record vs. defensive scheme family** — heavier infra.
- **Player injury return-from-IR flag** — useful but the injury feed already
  encodes status; the first-game-back effect is subtle.
- **Coaching staff churn rate** — proxy for organizational dysfunction.
- **Roster age / experience** — slow to move season-to-season.
- **Beat-writer sentiment scores** — needs NLP scrape; expensive to build.

## Notes on additional factors the user asked about

Both **head-coach tendencies** (v2 — implemented) and **star-player presence
deltas** (v2 — implemented) follow the entity (coach name / player_id) across
team moves. The aggregation windows are `expanding()` on a leakage-safe shift,
so when Sean Payton moved from NO to DEN his 2022 NO tendencies flow into the
2023 DEN games — exactly the cross-season carry the user asked for.

The cross-team carry assumes the coach / player retains the same style /
ability at the new team, which is true in expectation but noisy in any
single instance. The model treats them as inputs and learns the right
weighting from history.
