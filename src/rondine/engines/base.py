"""Shared engine adapter types."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class LaunchSpec:
    """Resolved argv + env for an upstream server process."""

    engine: str
    argv: list[str]
    env: dict[str, str] = field(default_factory=dict)
    cwd: str | None = None
    host: str = "127.0.0.1"
    port: int = 8080
    alias: str = ""
    model_id: str = ""
    notes: list[str] = field(default_factory=list)
    container: bool = False

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}/v1"

    def command_line(self) -> str:
        return " ".join(self.argv)


def selected_engine_args(plan: dict[str, Any]) -> dict[str, Any]:
    """Pull merged engine performance knobs from a plan/selected payload."""
    selected = plan.get("selected") or plan
    if isinstance(selected.get("engine_args"), dict):
        return dict(selected["engine_args"])
    variant = selected.get("variant") or {}
    if isinstance(variant.get("engine_args"), dict):
        return dict(variant["engine_args"])
    return {}


def append_flag(argv: list[str], flag: str, value: Any | None = None) -> None:
    if value is None:
        argv.append(flag)
        return
    if isinstance(value, bool):
        if value:
            argv.append(flag)
        return
    argv.extend([flag, str(value)])


class EngineAdapter(ABC):
    name: str

    @abstractmethod
    def setup(self, *, dry_run: bool = False) -> list[str]:
        """Return commands that were / would be run to install the engine."""

    @abstractmethod
    def pull(self, plan: dict[str, Any], *, dry_run: bool = False) -> list[str]:
        """Download model assets for the selected plan variant."""

    @abstractmethod
    def build_serve(self, plan: dict[str, Any], *, host: str, port: int) -> LaunchSpec:
        """Build an upstream serve command from a saved/selected plan."""
