"""Native macOS window launcher (pywebview) for the NFL Forecast app.

Spins up the FastAPI server in a background thread, then opens a native
WebKit window pointed at it. Closes when the window is dismissed.
"""

from __future__ import annotations

import socket
import threading
import time

from nfl_model.logging import get_logger

log = get_logger("app.desktop")


def _free_port(default: int = 8766) -> int:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.bind(("127.0.0.1", default))
        return default
    except OSError:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
        return int(port)
    finally:
        sock.close()


def launch_native_window(*, width: int = 1280, height: int = 860) -> None:
    """Start the local FastAPI server and open a native window pointing at it."""
    try:
        import webview
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "pywebview is not installed. Install with `uv sync --extra desktop`."
        ) from exc

    import uvicorn

    from nfl_model.app.main import app as fastapi_app

    port = _free_port(8766)

    config = uvicorn.Config(
        fastapi_app, host="127.0.0.1", port=port,
        log_level="info", access_log=False,
    )
    server = uvicorn.Server(config)

    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()

    # Wait briefly for server to come up before pointing the window at it.
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline and not server.started:
        time.sleep(0.05)
    if not server.started:
        raise RuntimeError("Server failed to start in time.")

    url = f"http://127.0.0.1:{port}"
    log.info("desktop.launch", url=url, width=width, height=height)
    webview.create_window("NFL Forecast", url=url, width=width, height=height)
    webview.start()
