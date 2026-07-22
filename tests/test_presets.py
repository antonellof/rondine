"""Named preset persistence tests."""

from __future__ import annotations

from rondine.presets import (
    delete_preset,
    list_presets,
    load_preset,
    preset_from_selected,
    save_preset,
    selected_with_preset_overrides,
)


def test_preset_roundtrip(tmp_path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setenv("RONDINE_HOME", str(tmp_path))
    selected = {
        "model_id": "gemma-4-12b",
        "engine": "llama.cpp",
        "quant": "Q4_K_M",
        "repo": "bartowski/gemma-4-12B-it-GGUF",
        "engine_args": {"parallel": 1},
    }
    preset = preset_from_selected(
        "coding",
        selected,
        profile="coding",
        host="127.0.0.1",
        port=8080,
        engine_args={"batch_size": 512},
    )
    path = save_preset(preset)
    assert path.is_file()
    loaded = load_preset("coding")
    assert loaded.selected["model_id"] == "gemma-4-12b"
    merged = selected_with_preset_overrides(loaded)
    assert merged["engine_args"]["parallel"] == 1
    assert merged["engine_args"]["batch_size"] == 512
    assert any(p.name == "coding" for p in list_presets())
    delete_preset("coding")
    assert list_presets() == []
