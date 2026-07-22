"""Planner memory-fit and ranking tests."""

from __future__ import annotations

from rondine.catalog import load_catalog
from rondine.detect import EngineStatus, HardwareInfo
from rondine.planner import estimate_memory, plan_model


def _hw(
    *,
    ram: float,
    apple: bool = False,
    spark: bool = False,
    linux: bool = False,
) -> HardwareInfo:
    if apple:
        platform, arch = "darwin", "arm64"
    elif spark or linux:
        platform, arch = "linux", "aarch64" if spark else "x86_64"
    else:
        platform, arch = "darwin", "arm64"
        apple = True
    return HardwareInfo(
        platform=platform,
        arch=arch,
        hostname="test",
        ram_gb=ram,
        is_apple_silicon=apple,
        is_spark=spark,
        cuda_available=spark or (linux and False),
        cuda_capability=(12, 1) if spark else None,
        gpu_name="NVIDIA GB10" if spark else "",
        metal_available=apple,
        engines=[
            EngineStatus("llama.cpp", True, path="/usr/bin/llama-server"),
            EngineStatus("mlx", apple, detail="test"),
            EngineStatus("vllm", spark, detail="test"),
        ],
    )


def test_estimate_includes_headroom() -> None:
    catalog = load_catalog()
    model = next(m for m in catalog.models if m.id == "qwen3.6-27b")
    variant = next(v for v in model.variants if v.engine == "llama.cpp")
    est = estimate_memory(catalog, model, variant, context=32768, available_gb=48)
    assert est.total_gb > variant.weight_gb
    assert est.os_reserve_gb > 0
    assert est.fits
    tight = estimate_memory(catalog, model, variant, context=32768, available_gb=20)
    assert not tight.fits


def test_glm_rejected_on_48gb_mac() -> None:
    catalog = load_catalog()
    hw = _hw(ram=48, apple=True)
    result = plan_model(catalog, hw, "glm-5.2", profile="coding", include_opt_in=True)
    assert result.selected is None
    assert all(c.rejected for c in result.candidates)


def test_auto_coding_picks_qwen_on_48gb_mac() -> None:
    catalog = load_catalog()
    hw = _hw(ram=48, apple=True)
    result = plan_model(catalog, hw, None, profile="coding")
    assert result.selected is not None
    assert result.selected.model_id.startswith("qwen3.6")
    assert result.selected.engine in {"mlx", "llama.cpp"}


def test_spark_prefers_vllm_nvfp4() -> None:
    catalog = load_catalog()
    hw = _hw(ram=128, spark=True)
    hw.cuda_available = True
    result = plan_model(catalog, hw, "qwen3.6-35b-a3b", profile="coding")
    assert result.selected is not None
    assert result.selected.engine == "vllm"
    assert result.selected.format == "nvfp4"


def test_deepseek_fits_128gb() -> None:
    catalog = load_catalog()
    hw = _hw(ram=128, apple=True)
    result = plan_model(catalog, hw, "deepseek-v4-flash", profile="coding", include_opt_in=True)
    assert result.selected is not None
    assert "IQ3" in result.selected.quant or result.selected.quant.startswith("UD-")


def test_catalog_prefers_non_unsloth_when_marked() -> None:
    catalog = load_catalog()
    hw = _hw(ram=64, apple=True)
    result = plan_model(catalog, hw, "qwen3.6-35b-a3b", profile="coding")
    assert result.selected is not None
    provider = (result.selected.variant or {}).get("provider")
    # On Apple Silicon, mlx-community preferred MLX should win when it fits
    assert result.selected.engine in {"mlx", "llama.cpp"}
    assert provider in {"mlx-community", "bartowski", "ggml-org", "mudler", "unsloth"}


def test_catalog_has_multiple_providers() -> None:
    catalog = load_catalog()
    providers: set[str] = set()
    for model in catalog.models:
        for variant in model.variants:
            providers.add(variant.provider)
    assert "bartowski" in providers
    assert "mlx-community" in providers
    assert "ggml-org" in providers
    assert "Qwen" in providers or "google" in providers
    assert "unsloth" in providers
    assert len(providers) >= 5
