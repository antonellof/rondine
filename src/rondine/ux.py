"""Terminal UX helpers (spinner / progress)."""

from __future__ import annotations

import os
import sys
import threading
import time
from collections.abc import Iterator, Sequence
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
    "muted": "white",
    "command": "bright_green",
    "rule": "bright_blue",
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


def echo_rule(width: int = 72) -> None:
    click.echo(styled("─" * width, "rule"))


def is_interactive_terminal() -> bool:
    stdin = click.get_text_stream("stdin")
    stdout = click.get_text_stream("stdout")
    return bool(
        getattr(stdin, "isatty", lambda: False)()
        and getattr(stdout, "isatty", lambda: False)()
    )


def select_menu(options: Sequence[str], *, title: str = "Select a configuration") -> int | None:
    """Select an item with arrow keys, with a numbered prompt fallback."""
    if not options:
        return None
    if not is_interactive_terminal():
        choice = click.prompt(
            title,
            type=click.IntRange(1, len(options)),
            default=1,
            show_choices=True,
        )
        return int(choice) - 1

    stdin = click.get_text_stream("stdin")
    stdout = click.get_text_stream("stdout")
    import termios
    import tty

    selected = 0
    fd = stdin.fileno()
    previous = termios.tcgetattr(fd)

    def render(*, move_up: bool = False) -> None:
        if move_up:
            stdout.write(f"\x1b[{len(options)}A")
        for index, option in enumerate(options):
            stdout.write("\r\x1b[2K")
            if index == selected:
                stdout.write(
                    f"{styled('❯', 'heading', bold=True)} "
                    f"{styled(option, 'success', bold=True)}\n"
                )
            else:
                stdout.write(f"  {option}\n")
        stdout.flush()

    try:
        tty.setraw(fd)
        click.echo()
        echo_heading(title)
        click.echo(styled("↑/↓ move · enter select · q cancel", "muted"))
        render()
        while True:
            key = os.read(fd, 3)
            if key in {b"\r", b"\n"}:
                break
            if key in {b"q", b"Q", b"\x1b"}:
                return None
            if key in {b"\x1b[A", b"k", b"K"}:
                selected = (selected - 1) % len(options)
                render(move_up=True)
            elif key in {b"\x1b[B", b"j", b"J"}:
                selected = (selected + 1) % len(options)
                render(move_up=True)
            elif key == b"\x03":
                raise click.Abort()
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, previous)
    click.echo(styled(f"✓ selected {options[selected]}", "success", bold=True))
    return selected


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
