"""Load and validate declarative model / hardware catalogs."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from rondine.paths import catalog_dir


@dataclass(frozen=True)
class ModelVariant:
    engine: str
    format: str
    repo: str
    quant: str
    weight_gb: float
    include: list[str] = field(default_factory=list)
    mmproj: str | None = None
    mtp: bool = False
    spark_moe_backend: str | None = None
    recommended_profiles: list[str] = field(default_factory=list)
    min_ram_gb: float | None = None


@dataclass(frozen=True)
class ModelEntry:
    id: str
    family: str
    display_name: str
    params: str
    active_params: str
    max_context: int
    modalities: list[str]
    coding_priority: int
    notes: str = ""
    opt_in: bool = False
    variants: list[ModelVariant] = field(default_factory=list)


@dataclass(frozen=True)
class HardwareTarget:
    id: str
    label: str
    platform: str
    arch: str
    min_ram_gb: float = 0.0
    max_ram_gb: float = 1e9
    notes: str = ""
    nodes: int = 1
    cluster: bool = False
    min_ram_gb_per_node: float | None = None
    cuda_capability_major: int | None = None
    sm: str | None = None
    preferred_engine: str | None = None


@dataclass(frozen=True)
class PlannerPolicy:
    os_reserve_gb: float
    activation_margin: float
    kv_bytes_per_token_per_b: float
    min_ram_baseline_context: int
    apple_engine_order: list[str]
    spark_engine_order: list[str]
    linux_engine_order: list[str]
    default_port: int
    default_host: str


@dataclass(frozen=True)
class Catalog:
    models: list[ModelEntry]
    targets: list[HardwareTarget]
    policy: PlannerPolicy
    profiles: dict[str, Any]
    version: int


def _load_toml(path: Path) -> dict[str, Any]:
    with path.open("rb") as fh:
        return tomllib.load(fh)


def _parse_variant(raw: dict[str, Any]) -> ModelVariant:
    include = raw.get("include") or []
    if isinstance(include, str):
        include = [include]
    return ModelVariant(
        engine=str(raw["engine"]),
        format=str(raw["format"]),
        repo=str(raw["repo"]),
        quant=str(raw["quant"]),
        weight_gb=float(raw["weight_gb"]),
        include=[str(x) for x in include],
        mmproj=raw.get("mmproj"),
        mtp=bool(raw.get("mtp", False)),
        spark_moe_backend=raw.get("spark_moe_backend"),
        recommended_profiles=[str(x) for x in raw.get("recommended_profiles", [])],
        min_ram_gb=float(raw["min_ram_gb"]) if "min_ram_gb" in raw else None,
    )


def _parse_model(raw: dict[str, Any]) -> ModelEntry:
    variants = [_parse_variant(v) for v in raw.get("variants", [])]
    return ModelEntry(
        id=str(raw["id"]),
        family=str(raw["family"]),
        display_name=str(raw["display_name"]),
        params=str(raw["params"]),
        active_params=str(raw.get("active_params", raw["params"])),
        max_context=int(raw["max_context"]),
        modalities=[str(x) for x in raw.get("modalities", ["text"])],
        coding_priority=int(raw.get("coding_priority", 0)),
        notes=str(raw.get("notes", "")),
        opt_in=bool(raw.get("opt_in", False)),
        variants=variants,
    )


def _parse_target(raw: dict[str, Any]) -> HardwareTarget:
    return HardwareTarget(
        id=str(raw["id"]),
        label=str(raw["label"]),
        platform=str(raw["platform"]),
        arch=str(raw["arch"]),
        min_ram_gb=float(raw.get("min_ram_gb", 0)),
        max_ram_gb=float(raw.get("max_ram_gb", 1e9)),
        notes=str(raw.get("notes", "")),
        nodes=int(raw.get("nodes", 1)),
        cluster=bool(raw.get("cluster", False)),
        min_ram_gb_per_node=(
            float(raw["min_ram_gb_per_node"]) if "min_ram_gb_per_node" in raw else None
        ),
        cuda_capability_major=(
            int(raw["cuda_capability_major"]) if "cuda_capability_major" in raw else None
        ),
        sm=raw.get("sm"),
        preferred_engine=raw.get("preferred_engine"),
    )


def load_catalog(directory: Path | None = None) -> Catalog:
    base = directory or catalog_dir()
    models_path = base / "models.toml"
    hardware_path = base / "hardware.toml"
    if not models_path.is_file():
        raise FileNotFoundError(f"missing {models_path}")
    if not hardware_path.is_file():
        raise FileNotFoundError(f"missing {hardware_path}")

    models_raw = _load_toml(models_path)
    hardware_raw = _load_toml(hardware_path)

    models = [_parse_model(m) for m in models_raw.get("models", [])]
    if not models:
        raise ValueError("models.toml contains no models")

    targets = [_parse_target(t) for t in hardware_raw.get("targets", [])]
    policy_raw = hardware_raw.get("policy", {})
    policy = PlannerPolicy(
        os_reserve_gb=float(policy_raw.get("os_reserve_gb", 8.0)),
        activation_margin=float(policy_raw.get("activation_margin", 0.12)),
        kv_bytes_per_token_per_b=float(policy_raw.get("kv_bytes_per_token_per_b", 6e-6)),
        min_ram_baseline_context=int(policy_raw.get("min_ram_baseline_context", 4096)),
        apple_engine_order=list(policy_raw.get("apple_engine_order", ["mlx", "llama.cpp"])),
        spark_engine_order=list(policy_raw.get("spark_engine_order", ["vllm", "llama.cpp"])),
        linux_engine_order=list(policy_raw.get("linux_engine_order", ["llama.cpp", "vllm"])),
        default_port=int(policy_raw.get("default_port", 8080)),
        default_host=str(policy_raw.get("default_host", "127.0.0.1")),
    )
    profiles = models_raw.get("profiles", {})
    version = int(models_raw.get("catalog", {}).get("version", 1))
    return Catalog(
        models=models,
        targets=targets,
        policy=policy,
        profiles=profiles,
        version=version,
    )


def get_model(catalog: Catalog, model_id: str) -> ModelEntry:
    for model in catalog.models:
        if model.id == model_id:
            return model
    known = ", ".join(m.id for m in catalog.models)
    raise KeyError(f"unknown model '{model_id}'; known: {known}")


def profile_settings(catalog: Catalog, profile: str, family: str) -> dict[str, Any]:
    profiles = catalog.profiles.get(profile)
    if not isinstance(profiles, dict):
        raise KeyError(f"unknown profile '{profile}'")
    base: dict[str, Any] = {
        "context": profiles.get("context", 16384),
        "description": profiles.get("description", ""),
    }
    family_cfg = profiles.get(family)
    if isinstance(family_cfg, dict):
        base.update(family_cfg)
    return base
