"""Hardware suggestor / configurator tests."""

from __future__ import annotations

from rondine.catalog import load_catalog, resolve_engine_args
from rondine.detect import EngineStatus, HardwareInfo
from rondine.suggest import suggest_for_hardware


def _hw(*, ram: float, apple: bool = True, spark: bool = False) -> HardwareInfo:
    if spark:
        platform, arch = "linux", "aarch64"
        apple = False
    else:
        platform, arch = "darwin", "arm64"
    return HardwareInfo(
        platform=platform,
        arch=arch,
        hostname="test",
        ram_gb=ram,
        is_apple_silicon=apple,
        is_spark=spark,
        cuda_available=spark,
        cuda_capability=(12, 1) if spark else None,
        gpu_name="NVIDIA GB10" if spark else "",
        metal_available=apple,
        engines=[
            EngineStatus("llama.cpp", True, path="/usr/bin/llama-server"),
            EngineStatus("mlx", apple, detail="test"),
            EngineStatus("vllm", spark, detail="test"),
        ],
    )


def test_suggest_mac_returns_ranked_configs() -> None:
    catalog = load_catalog()
    result = suggest_for_hardware(catalog, _hw(ram=48), profile="coding", limit=3)
    assert result.target_id in {"mac-48", "mac-36"}
    assert result.suggestions
    top = result.suggestions[0]
    assert top.engine in {"mlx", "llama.cpp"}
    assert top.engine_args
    assert top.next_steps
    assert "rondine serve" in top.next_steps[-2] or "rondine serve" in " ".join(top.next_steps)


def test_suggest_spark_prefers_vllm() -> None:
    catalog = load_catalog()
    result = suggest_for_hardware(catalog, _hw(ram=128, spark=True), profile="coding")
    assert result.target_id == "spark-128"
    assert result.preferred_engine == "vllm"
    assert result.suggestions
    assert result.suggestions[0].engine == "vllm"


def test_resolve_engine_args_merges_layers() -> None:
    catalog = load_catalog()
    args = resolve_engine_args(
        catalog, "llama.cpp", profile="coding", target_template="mac"
    )
    assert args["n_gpu_layers"] == 99
    assert args["parallel"] == 1  # coding override
    assert args["cache_type_k"] == "q4_0"  # mac override
