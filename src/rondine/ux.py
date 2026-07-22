"""Terminal UX helpers (spinner / progress)."""

from __future__ import annotations

import sys
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from typing import TextIO


@contextmanager
def spinner(
    message: str,
    *,
    stream: TextIO | None = None,
    enabled: bool | None = None,
) -> Iterator[None]:
    """Show a lightweight spinner while a blocking section runs.

    Disabled automatically when stdout/stderr is not a TTY (CI, pipes, Click runner).
    """
    out = stream or sys.stderr
    if enabled is None:
        enabled = bool(getattr(out, "isatty", lambda: False)())
    if not enabled:
        print(f"{message} ...", file=out, flush=True)
        yield
        return

    stop = threading.Event()
    frames = ("⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏")

    def _spin() -> None:
        i = 0
        while not stop.is_set():
            frame = frames[i % len(frames)]
            print(f"\r{frame} {message}", end="", file=out, flush=True)
            i += 1
            time.sleep(0.08)

    thread = threading.Thread(target=_spin, daemon=True)
    thread.start()
    try:
        yield
    finally:
        stop.set()
        thread.join(timeout=1.0)
        # Clear the spinner line, then print a done marker.
        print(f"\r{' ' * (len(message) + 4)}\r✔ {message}", file=out, flush=True)
