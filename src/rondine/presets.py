"""Named launch presets for easy restart."""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from rondine.paths import presets_dir

_NAME_RE = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9._-]{0,63}$")


@dataclass
class Preset:
    name: str
    profile: str
    selected: dict[str, Any]
    host: str = "127.0.0.1"
    port: int = 8080
    run_name: str = "default"
    target_id: str | None = None
    engine_args: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    created_at: float = 0.0
    updated_at: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "profile": self.profile,
            "selected": self.selected,
            "host": self.host,
            "port": self.port,
            "run_name": self.run_name,
            "target_id": self.target_id,
            "engine_args": self.engine_args,
            "notes": self.notes,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Preset:
        return cls(
            name=str(data["name"]),
            profile=str(data.get("profile") or "coding"),
            selected=dict(data.get("selected") or {}),
            host=str(data.get("host") or "127.0.0.1"),
            port=int(data.get("port") or 8080),
            run_name=str(data.get("run_name") or "default"),
            target_id=data.get("target_id"),
            engine_args=dict(data.get("engine_args") or {}),
            notes=list(data.get("notes") or []),
            created_at=float(data.get("created_at") or 0.0),
            updated_at=float(data.get("updated_at") or 0.0),
        )


def validate_preset_name(name: str) -> str:
    if not _NAME_RE.match(name):
        raise ValueError(
            f"invalid preset name {name!r}; use letters/digits/._- (max 64)"
        )
    return name


def _path(name: str) -> Path:
    return presets_dir() / f"{validate_preset_name(name)}.json"


def save_preset(preset: Preset) -> Path:
    validate_preset_name(preset.name)
    now = time.time()
    if preset.created_at <= 0:
        preset.created_at = now
    preset.updated_at = now
    path = _path(preset.name)
    path.write_text(json.dumps(preset.to_dict(), indent=2), encoding="utf-8")
    return path


def load_preset(name: str) -> Preset:
    path = _path(name)
    if not path.is_file():
        raise FileNotFoundError(f"preset '{name}' not found")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"corrupt preset '{name}'")
    data.setdefault("name", name)
    return Preset.from_dict(data)


def delete_preset(name: str) -> None:
    path = _path(name)
    if not path.is_file():
        raise FileNotFoundError(f"preset '{name}' not found")
    path.unlink()


def list_presets() -> list[Preset]:
    items: list[Preset] = []
    for path in sorted(presets_dir().glob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                data.setdefault("name", path.stem)
                items.append(Preset.from_dict(data))
        except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError):
            continue
    items.sort(key=lambda p: p.updated_at or p.created_at, reverse=True)
    return items


def preset_from_selected(
    name: str,
    selected: dict[str, Any],
    *,
    profile: str,
    host: str,
    port: int,
    run_name: str = "default",
    target_id: str | None = None,
    engine_args: dict[str, Any] | None = None,
    notes: list[str] | None = None,
) -> Preset:
    return Preset(
        name=name,
        profile=profile,
        selected=selected,
        host=host,
        port=port,
        run_name=run_name,
        target_id=target_id,
        engine_args=dict(engine_args or selected.get("engine_args") or {}),
        notes=list(notes or []),
    )


def selected_with_preset_overrides(preset: Preset) -> dict[str, Any]:
    """Clone selected payload and merge preset engine_args for serve."""
    selected = dict(preset.selected)
    merged = dict(selected.get("engine_args") or {})
    merged.update(preset.engine_args)
    selected["engine_args"] = merged
    selected["profile"] = preset.profile
    return selected
