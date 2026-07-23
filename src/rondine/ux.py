"""Terminal UX helpers (spinner / progress)."""

from __future__ import annotations

import sys
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from typing import TextIO

import click

COLORS = {
    "heading": "bright_cyan",
    "label": "cyan",
    "success": "bright_green",
    "warning": "bright_yellow",
    "error": "bright_red",
    "accent": "bright_magenta",
    "muted": "bright_black",
    "command": "bright_green",
}


def styled(value: object, role: str, *, bold: bool = False) -> str:
    """Apply a semantic terminal color that Click strips for pipes and tests."""
    return click.style(str(value), fg=COLORS[role], bold=bold)


def echo_heading(message: str) -> None:
    click.secho(message, fg=COLORS["heading"], bold=True)


def echo_kv(
    label: str,
    value: object,
    *,
    indent: str = "",
    value_role: str | None = None,
) -> None:
    rendered = styled(value, value_role) if value_role else str(value)
    click.echo(f"{indent}{styled(label + ':', 'label', bold=True)} {rendered}")


def echo_note(message: str) -> None:
    click.echo(f"{styled('note:', 'warning', bold=True)} {message}")


def echo_warning(message: str) -> None:
    click.echo(
        f"{styled('warning:', 'warning', bold=True)} {message}",
        err=True,
    )


def echo_success(message: str) -> None:
    click.echo(styled(message, "success", bold=True))


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
            frame = click.style(frames[i % len(frames)], fg=COLORS["heading"])
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
        done = click.style("✔", fg=COLORS["success"], bold=True)
        print(f"\r{' ' * (len(message) + 4)}\r{done} {message}", file=out, flush=True)
