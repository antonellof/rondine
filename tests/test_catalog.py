"""Catalog loading and validation."""

from __future__ import annotations

from pathlib import Path

from rondine.catalog import get_model, load_catalog, profile_settings


def test_load_catalog() -> None:
    catalog = load_catalog()
    assert catalog.version >= 1
    assert len(catalog.models) >= 5
    assert any(m.id == "qwen3.6-35b-a3b" for m in catalog.models)
    assert catalog.policy.default_port == 8080
    assert catalog.targets


def test_get_model_and_profile() -> None:
    catalog = load_catalog()
    model = get_model(catalog, "qwen3.6-27b")
    assert model.family == "qwen3.6"
    settings = profile_settings(catalog, "coding", model.family)
    assert settings["temperature"] == 0.6
    assert settings["context"] == 32768


def test_catalog_paths_exist() -> None:
    root = Path(__file__).resolve().parents[1]
    assert (root / "catalog" / "models.toml").is_file()
    assert (root / "catalog" / "hardware.toml").is_file()
