"""Unit tests for Hub helpers (no network)."""

from __future__ import annotations

from rondine.hub import (
    HubFile,
    HubInspectResult,
    detect_quant,
    infer_engine_and_format,
)


def test_infer_engine_gguf() -> None:
    eng, fmt = infer_engine_and_format("bartowski/Qwen_Qwen3.6-35B-A3B-GGUF", ["gguf"])
    assert eng == "llama.cpp"
    assert fmt == "gguf"


def test_infer_engine_mlx() -> None:
    eng, fmt = infer_engine_and_format("mlx-community/Qwen3.6-27B-4bit", ["mlx"])
    assert eng == "mlx"
    assert fmt == "mlx"


def test_infer_engine_nvfp4() -> None:
    eng, fmt = infer_engine_and_format("unsloth/Qwen3.6-27B-NVFP4", [])
    assert eng == "vllm"
    assert fmt == "nvfp4"


def test_detect_quant() -> None:
    assert detect_quant("Qwen3.6-35B-A3B-Q4_K_M.gguf") == "Q4_K_M"
    assert detect_quant("model-UD-Q4_K_XL.gguf") == "UD-Q4_K_XL"
    assert detect_quant("weights-4bit") == "4bit"


def test_hub_inspect_to_plan_selected() -> None:
    result = HubInspectResult(
        repo_id="bartowski/Qwen_Qwen3.6-35B-A3B-GGUF",
        engine_hint="llama.cpp",
        format_hint="gguf",
        files=[
            HubFile("Qwen3.6-35B-A3B-Q4_K_M.gguf", 22.3, quant="Q4_K_M", is_gguf=True),
            HubFile("mmproj-F16.gguf", 1.0, is_mmproj=True, is_gguf=True),
        ],
        recommended_quant="Q4_K_M",
        recommended_file="Qwen3.6-35B-A3B-Q4_K_M.gguf",
        weight_gb=22.3,
        notes=["test"],
    )
    selected = result.to_plan_selected(profile="coding", context=16384)
    assert selected["engine"] == "llama.cpp"
    assert selected["repo"] == "bartowski/Qwen_Qwen3.6-35B-A3B-GGUF"
    assert selected["quant"] == "Q4_K_M"
    assert selected["variant"]["hub"] is True
    assert selected["variant"]["provider"] == "bartowski"
