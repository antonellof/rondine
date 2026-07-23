"""Terminal interaction tests."""

from __future__ import annotations

import io
import os
import termios
import tty

import click

from rondine.ux import select_menu


class _TtyInput(io.StringIO):
    def isatty(self) -> bool:
        return True

    def fileno(self) -> int:
        return 42


class _TtyOutput(io.StringIO):
    def isatty(self) -> bool:
        return True


def test_select_menu_supports_arrow_keys(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    stdin = _TtyInput()
    stdout = _TtyOutput()
    keys = iter((b"\x1b[B", b"\r"))

    monkeypatch.setattr(
        click,
        "get_text_stream",
        lambda name: stdin if name == "stdin" else stdout,
    )
    monkeypatch.setattr(os, "read", lambda fd, size: next(keys))
    monkeypatch.setattr(termios, "tcgetattr", lambda fd: [])
    monkeypatch.setattr(termios, "tcsetattr", lambda fd, when, value: None)
    monkeypatch.setattr(tty, "setraw", lambda fd: None)

    selected = select_menu(["first", "second"])

    assert selected == 1
    assert "second" in stdout.getvalue()
    assert "\x1b[2A" in stdout.getvalue()
