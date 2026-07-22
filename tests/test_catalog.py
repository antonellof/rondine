"""Catalog loading and validation."""

from __future__ import annotations

from pathlib import Path

from rondine.catalog import get_model, load_catalog, profile_settings


def test_load_catalog() -> None:
    catalog = load_catalog()
    assert catalog.version >= 1
    assert len(catalog.models) >= 5
    assert any(m.id == "qwen3.6-35b-a3b" for m in catalog.models)
    assert any(m.id == "qwen2.5-coder-1.5b" for m in catalog.models)
    assert any(m.id == "qwen2.5-coder-3b" for m in catalog.models)
    assert catalog.policy.default_port == 8080
    assert catalog.targets
    assert "llama.cpp" in catalog.engine_templates
    assert "defaults" in catalog.engine_templates["llama.cpp"]
    assert "cuda" in catalog.engine_templates["llama.cpp"]
    mac36 = next(t for t in catalog.targets if t.id == "mac-36")
    assert mac36.suggested_models
    assert mac36.preferred_engine == "mlx"
    cuda24 = next(t for t in catalog.targets if t.id == "cuda-24")
    assert cuda24.require_cuda
    assert cuda24.min_vram_gb == 20
    cuda8 = next(t for t in catalog.targets if t.id == "cuda-8")
    assert cuda8.suggested_models[:2] == [
        "qwen2.5-coder-3b",
        "qwen2.5-coder-1.5b",
    ]



def test_get_model_and_profile() -> None:
    catalog = load_catalog()
    model = get_model(catalog, "qwen3.6-27b")
    assert model.family == "qwen3.6"
    settings = profile_settings(catalog, "coding", model.family)
    assert settings["temperature"] == 0.6
    assert settings["context"] == 32768

    small = get_model(catalog, "qwen2.5-coder-3b")
    assert small.family == "qwen2.5"
    assert small.variants[0].weight_gb < 2.0
    assert profile_settings(catalog, "coding", small.family)["temperature"] == 0.2


def test_catalog_paths_exist() -> None:
    root = Path(__file__).resolve().parents[1]
    assert (root / "catalog" / "models.toml").is_file()
    assert (root / "catalog" / "hardware.toml").is_file()
