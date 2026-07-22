"""Filesystem helpers and user data locations."""

from __future__ import annotations

import os
import shutil
from pathlib import Path


def package_root() -> Path:
    """Return the installed package directory (src/rondine or site-packages)."""
    return Path(__file__).resolve().parent


def repo_root() -> Path | None:
    """Return the development checkout root when running from source."""
    here = package_root()
    # src/rondine -> repo
    candidate = here.parent.parent
    if (candidate / "catalog" / "models.toml").is_file():
        return candidate
    # hatch force-include places catalog under package
    return None


def catalog_dir() -> Path:
    env = os.environ.get("RONDINE_CATALOG_DIR")
    if env:
        return Path(env).expanduser().resolve()
    root = repo_root()
    if root is not None:
        return root / "catalog"
    pkg = package_root() / "catalog"
    if pkg.is_dir():
        return pkg
    raise FileNotFoundError("Rondine catalog not found; set RONDINE_CATALOG_DIR")


def assets_dir() -> Path:
    root = repo_root()
    if root is not None and (root / "assets").is_dir():
        return root / "assets"
    return package_root() / "assets"


def brand_logo() -> str:
    path = assets_dir() / "rondine.txt"
    if path.is_file():
        return path.read_text(encoding="utf-8").rstrip()
    return "rondine"


def data_home() -> Path:
    override = os.environ.get("RONDINE_HOME")
    if override:
        path = Path(override).expanduser().resolve()
    else:
        path = Path.home() / ".rondine"
    path.mkdir(parents=True, exist_ok=True)
    return path


def models_dir() -> Path:
    path = data_home() / "models"
    path.mkdir(parents=True, exist_ok=True)
    return path


def engines_dir() -> Path:
    path = data_home() / "engines"
    path.mkdir(parents=True, exist_ok=True)
    return path


def run_dir() -> Path:
    path = data_home() / "run"
    path.mkdir(parents=True, exist_ok=True)
    return path


def plans_dir() -> Path:
    path = data_home() / "plans"
    path.mkdir(parents=True, exist_ok=True)
    return path


def clusters_dir() -> Path:
    path = data_home() / "clusters"
    path.mkdir(parents=True, exist_ok=True)
    return path


def which(cmd: str) -> str | None:
    return shutil.which(cmd)
