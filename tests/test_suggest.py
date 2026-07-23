"""Hardware suggestor / configurator tests."""

from __future__ import annotations

from rondine.catalog import load_catalog, resolve_engine_args
from rondine.detect import EngineStatus, HardwareInfo
from rondine.hub import HubFile, HubInspectResult, HubModelHit
from rondine.suggest import suggest_for_hardware


def _hw(*, ram: float, apple: bool = True, spark: bool = False, vram: float = 0.0) -> HardwareInfo:
    if spark:
        platform, arch = "linux", "aarch64"
        apple = False
    elif vram > 0:
        platform, arch = "linux", "x86_64"
        apple = False
    else:
        platform, arch = "darwin", "arm64"
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
        gpu_name="NVIDIA GB10" if spark else ("NVIDIA GeForce RTX 4090" if vram else ""),
        vram_gb=ram if spark else vram,
        gpu_count=1 if cuda else 0,
        metal_available=apple,
        engines=[
            EngineStatus("llama.cpp", True, path="/usr/bin/llama-server"),
            EngineStatus("mlx", apple, detail="test"),
            EngineStatus("vllm", spark or vram >= 40, detail="test"),
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


def test_suggest_cuda_24() -> None:
    catalog = load_catalog()
    result = suggest_for_hardware(
        catalog, _hw(ram=64, apple=False, vram=24.0), profile="coding", limit=3
    )
    assert result.target_id == "cuda-24"
    assert result.preferred_engine == "llama.cpp"
    assert result.suggestions
    assert result.suggestions[0].engine == "llama.cpp"
    assert "batch_size" in result.suggestions[0].engine_args


def test_resolve_engine_args_merges_layers() -> None:
    catalog = load_catalog()
    args = resolve_engine_args(
        catalog, "llama.cpp", profile="coding", target_template="mac"
    )
    assert args["n_gpu_layers"] == 99
    assert args["parallel"] == 1  # coding override
    assert args["cache_type_k"] == "q8_0"
    assert args["batch_size"] == 2048  # mac override
    assert args["flash_attn"] is True
    assert args.get("mlock") is True

    tight = resolve_engine_args(
        catalog, "llama.cpp", profile="coding", target_template="mac-tight"
    )
    assert tight["batch_size"] == 512
    assert tight["ubatch_size"] == 512


def test_mlx_template_enables_fast_synch() -> None:
    catalog = load_catalog()
    args = resolve_engine_args(catalog, "mlx", profile="coding", target_template="mac")
    assert args.get("metal_fast_synch") is True


def test_suggest_supplements_catalog_with_fitting_hub_result(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    catalog = load_catalog()

    def fake_search(query: str, **kwargs: object) -> list[HubModelHit]:
        assert query == "coder"
        assert kwargs["engine"] == "mlx"
        return [
            HubModelHit(
                repo_id="mlx-community/new-coder-7b-4bit",
                downloads=12_000,
                engine_hint="mlx",
                format_hint="mlx",
                score=24.0,
            )
        ]

    def fake_inspect(repo_id: str) -> HubInspectResult:
        assert repo_id == "mlx-community/new-coder-7b-4bit"
        return HubInspectResult(
            repo_id=repo_id,
            engine_hint="mlx",
            format_hint="mlx",
            files=[HubFile("weights-4bit.npz", 4.0, quant="4bit")],
            recommended_quant="4bit",
            recommended_file="weights-4bit.npz",
            weight_gb=4.0,
            notes=["engine hint: mlx / format: mlx"],
        )

    monkeypatch.setattr("rondine.suggest.search_hub", fake_search)
    monkeypatch.setattr("rondine.suggest.inspect_repo", fake_inspect)

    result = suggest_for_hardware(
        catalog,
        _hw(ram=32),
        profile="coding",
        limit=3,
        include_hub=True,
    )

    hub = next(s for s in result.suggestions if s.source == "huggingface")
    assert hub.repo == "mlx-community/new-coder-7b-4bit"
    assert hub.estimate["fits"] is True
    assert hub.selected["variant"]["hub"] is True
    assert hub.next_steps[0].startswith("rondine plan mlx-community/")
    assert len(result.suggestions) == 3


def test_suggest_falls_back_when_hub_search_fails(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    def unavailable(*args: object, **kwargs: object) -> list[HubModelHit]:
        raise RuntimeError("offline")

    monkeypatch.setattr("rondine.suggest.search_hub", unavailable)
    result = suggest_for_hardware(
        load_catalog(),
        _hw(ram=32),
        limit=3,
        include_hub=True,
    )

    assert result.suggestions
    assert all(s.source == "catalog" for s in result.suggestions)
    assert any("search unavailable" in note for note in result.notes)
