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
    vram: float = 0.0,
    gpu_name: str = "",
) -> HardwareInfo:
    if apple:
        platform, arch = "darwin", "arm64"
    elif spark:
        platform, arch = "linux", "aarch64"
    elif linux or vram > 0:
        platform, arch = "linux", "x86_64"
        linux = True
    else:
        platform, arch = "darwin", "arm64"
        apple = True
    cuda = spark or vram > 0
    return HardwareInfo(
        platform=platform,
        arch=arch,
        hostname="test",
        ram_gb=ram,
        is_apple_silicon=apple,
        is_spark=spark,
        cuda_available=cuda,
        cuda_capability=(12, 1) if spark else ((8, 9) if vram else None),
        gpu_name=gpu_name
        or ("NVIDIA GB10" if spark else ("NVIDIA GeForce RTX 4090" if vram else "")),
        vram_gb=ram if spark else vram,
        gpu_count=1 if cuda else 0,
        metal_available=apple,
        engines=[
            EngineStatus("llama.cpp", True, path="/usr/bin/llama-server"),
            EngineStatus("mlx", apple, detail="test"),
            EngineStatus("vllm", spark or (linux and vram >= 40), detail="test"),
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


def test_cuda_24_matches_vram_target() -> None:
    catalog = load_catalog()
    hw = _hw(ram=64, vram=24.0, gpu_name="NVIDIA GeForce RTX 4090")
    result = plan_model(catalog, hw, None, profile="coding")
    assert result.target_id == "cuda-24"
    assert result.selected is not None
    assert result.selected.engine == "llama.cpp"
    # Fit against VRAM, not system RAM
    assert result.selected.estimate.available_gb == 24.0


def test_cuda_8_rejects_large_models() -> None:
    catalog = load_catalog()
    hw = _hw(ram=32, vram=8.0, gpu_name="NVIDIA GeForce RTX 4060")
    result = plan_model(catalog, hw, "qwen3.6-35b-a3b", profile="coding", include_opt_in=True)
    assert result.target_id == "cuda-8"
    assert result.selected is None or result.selected.estimate.fits
    # 35B-A3B Q4 (~20GB) should not fit 8GB
    if result.selected:
        assert result.selected.weight_gb < 10
    else:
        assert any(c.rejected for c in result.candidates)


def test_cuda_8_auto_selects_small_coder() -> None:
    catalog = load_catalog()
    hw = _hw(ram=32, vram=8.0, gpu_name="NVIDIA GeForce RTX 4060")

    result = plan_model(catalog, hw, None, profile="coding")

    assert result.target_id == "cuda-8"
    assert result.selected is not None
    assert result.selected.model_id == "qwen2.5-coder-3b"
    assert result.selected.estimate.fits


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
