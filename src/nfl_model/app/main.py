"""FastAPI factory for the NFL Forecast desktop app.

The ``nfl-model app`` CLI launches uvicorn against this factory and opens
a native pywebview window pointed at it. The app is **local-only**: it
binds to 127.0.0.1 and never accepts external traffic.

We expose ``app`` (a ready-to-serve singleton built via ``create_app()``)
and the factory itself so tests can construct an isolated instance.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from nfl_model.app.routes import router
from nfl_model.logging import configure_logging, get_logger

configure_logging()
log = get_logger("app.main")

APP_DIR = Path(__file__).resolve().parent


@asynccontextmanager
async def _lifespan(_app: FastAPI) -> AsyncIterator[None]:
    """Run startup/shutdown hooks for the FastAPI app.

    On startup we eagerly ensure the DuckDB schema exists so the first
    request after a clean install can't crash on a missing-table error.
    The ``init_schema`` call is essentially free after the first
    invocation thanks to a process-local short-circuit flag.
    """
    from nfl_model.data.warehouse import init_schema

    log.info("app.start")
    try:
        init_schema()
    except Exception:  # noqa: BLE001 -- never let DB init crash app boot
        log.exception("app.start.schema_init_failed")
    yield
    log.info("app.stop")


def create_app() -> FastAPI:
    app = FastAPI(
        title="NFL Forecast",
        description="Local desktop forecasting UI for the NFL betting model.",
        version="1.0.0",
        # Local-only: keep docs surface minimal.
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
        lifespan=_lifespan,
    )

    app.mount(
        "/static",
        StaticFiles(directory=APP_DIR / "static"),
        name="static",
    )
    app.include_router(router)
    return app


app = create_app()
