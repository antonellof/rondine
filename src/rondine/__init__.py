"""Rondine — hardware-aware local LLM launcher."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("rondine")
except PackageNotFoundError:  # pragma: no cover
    __version__ = "0.1.1"

__all__ = ["__version__"]
