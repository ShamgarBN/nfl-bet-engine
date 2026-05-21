"""NFL Bet Engine.

Free-data, walk-forward backtested NFL prediction system covering spread,
moneyline, total, Wong teasers, and key-number alt-lines. Two-stage
LightGBM + Monte Carlo with isotonic calibration. Conviction-tier
filtered. Iterates against itself via ablate / tune / bake-off.

See README.md for honest performance targets and architecture details.
"""

from __future__ import annotations

__version__ = "0.1.0"
