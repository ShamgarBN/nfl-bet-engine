"""FastAPI smoke tests: every page renders without an exception."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def client() -> TestClient:
    from nfl_model.app.main import app

    return TestClient(app)


def test_health_endpoint(client: TestClient) -> None:
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_healthz_endpoint(client: TestClient) -> None:
    resp = client.get("/healthz")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"


def test_dashboard_renders(client: TestClient) -> None:
    resp = client.get("/")
    assert resp.status_code == 200
    assert "NFL Forecast" in resp.text


def test_season_renders(client: TestClient) -> None:
    resp = client.get("/season")
    assert resp.status_code == 200
    assert "Season performance" in resp.text


def test_performance_renders(client: TestClient) -> None:
    resp = client.get("/performance")
    assert resp.status_code == 200
    assert "Model performance" in resp.text


def test_teasers_renders(client: TestClient) -> None:
    resp = client.get("/teasers")
    assert resp.status_code == 200
    assert "Wong-teaser" in resp.text


def test_log_renders(client: TestClient) -> None:
    resp = client.get("/log")
    assert resp.status_code == 200
    assert "My picks" in resp.text


def test_favicon_redirect(client: TestClient) -> None:
    resp = client.get("/favicon.ico", follow_redirects=False)
    assert resp.status_code in (302, 307)
    assert "/static/favicon.svg" in resp.headers.get("location", "")


def test_static_favicon(client: TestClient) -> None:
    resp = client.get("/static/favicon.svg")
    assert resp.status_code == 200
    assert "svg" in resp.headers.get("content-type", "")


def test_filter_picks_helper_handles_empty() -> None:
    from nfl_model.app.routes import _filter_picks

    assert _filter_picks([], market="spread", tier="all") == []


def test_journal_metrics_empty_inputs() -> None:
    """Calibration / rolling / slice helpers cope with no journal."""
    import pandas as pd

    from nfl_model.journal import (
        calibration_bins,
        rolling_accuracy,
        slice_breakdown,
    )

    empty = pd.DataFrame()
    assert calibration_bins(empty).empty
    assert rolling_accuracy(empty).empty
    assert slice_breakdown(empty, by="tier").empty
