"""Calibrator behavior: monotone, clipping at extremes, round-trip safe."""

from __future__ import annotations

import numpy as np

from nfl_model.model.calibrate import fit_calibrator, load_calibrator, save_calibrator


def test_calibrator_is_monotone() -> None:
    """A trained isotonic calibrator should be (weakly) increasing in p."""
    rng = np.random.default_rng(42)
    raw = rng.uniform(0, 1, size=2000)
    true_p = np.clip(raw**2, 0.01, 0.99)
    outcomes = (rng.uniform(0, 1, size=2000) < true_p).astype(int)

    cal = fit_calibrator("spread", raw, outcomes)
    grid = np.linspace(0.05, 0.95, 19)
    calibrated = cal.transform(grid)
    diffs = np.diff(calibrated)
    assert (diffs >= -1e-6).all(), (
        "Calibrator must be monotonically non-decreasing; "
        f"got {diffs[diffs < -1e-6]}"
    )


def test_calibrator_handles_extremes() -> None:
    """Inputs outside [0, 1] should be clipped, not raise."""
    raw = np.linspace(0.1, 0.9, 100)
    outcomes = (raw > 0.5).astype(int)
    cal = fit_calibrator("ml", raw, outcomes)
    out_low = cal.transform(np.array([-0.5, 0.0]))
    out_high = cal.transform(np.array([1.0, 1.5]))
    assert np.isfinite(out_low).all()
    assert np.isfinite(out_high).all()


def test_calibrator_round_trip(tmp_path, monkeypatch) -> None:
    """Save+load should produce a calibrator that maps inputs identically."""
    from nfl_model import config as cfg

    monkeypatch.setattr(cfg.settings, "model_dir", tmp_path)
    rng = np.random.default_rng(7)
    raw = rng.uniform(0, 1, size=500)
    outcomes = (rng.uniform(0, 1, size=500) < raw).astype(int)
    cal = fit_calibrator("total", raw, outcomes)
    grid = np.linspace(0.1, 0.9, 9)
    expected = cal.transform(grid)

    save_calibrator(cal)
    loaded = load_calibrator("total")
    assert loaded is not None
    np.testing.assert_allclose(loaded.transform(grid), expected, atol=1e-9)
