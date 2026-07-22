"""Deterministic model / engine / quant recommendation."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from rondine.catalog import (
    Catalog,
    ModelEntry,
    ModelVariant,
    get_model,
    get_target,
    profile_settings,
    resolve_engine_args,
)
from rondine.detect import HardwareInfo
from rondine.paths import plans_dir


def _parse_active_b(active_params: str) -> float:
    text = active_params.strip().upper().replace(" ", "")
    match = re.match(r"([0-9.]+)\s*B", text)
    if match:
        return float(match.group(1))
    match = re.match(r"([0-9.]+)", text)
    if match:
        return float(match.group(1))
    return 7.0


@dataclass
class FitEstimate:
    weight_gb: float
    kv_gb: float
    activation_gb: float
    os_reserve_gb: float
    total_gb: float
    available_gb: float
    fits: bool
    headroom_gb: float


@dataclass
class PlanCandidate:
    model_id: str
    display_name: str
    engine: str
    format: str
    repo: str
    quant: str
    profile: str
    context: int
    weight_gb: float
    estimate: FitEstimate
    score: float
    reasons: list[str] = field(default_factory=list)
    rejected: bool = False
    reject_reason: str = ""
    variant: dict[str, Any] = field(default_factory=dict)
    sampling: dict[str, Any] = field(default_factory=dict)


@dataclass
class PlanResult:
    hardware: dict[str, Any]
    profile: str
    selected: PlanCandidate | None
    candidates: list[PlanCandidate]
    target_id: str | None = None


def available_memory_gb(catalog: Catalog, hw: HardwareInfo) -> tuple[float, float]:
    """Return (available_gb, os_or_vram_reserve) for fit estimates."""
    if hw.is_discrete_cuda:
        return hw.vram_gb, catalog.policy.vram_reserve_gb
    return hw.ram_gb, catalog.policy.os_reserve_gb


def estimate_memory(
    catalog: Catalog,
    model: ModelEntry,
    variant: ModelVariant,
    context: int,
    available_gb: float,
    *,
    reserve_gb: float | None = None,
) -> FitEstimate:
    weight = float(variant.weight_gb)
    active_b = _parse_active_b(model.active_params)
    coeff = catalog.policy.kv_bytes_per_token_per_b
    baseline = catalog.policy.min_ram_baseline_context
    # Full KV estimate for reporting.
    kv = context * active_b * coeff
    activation = weight * catalog.policy.activation_margin
    os_reserve = float(reserve_gb if reserve_gb is not None else catalog.policy.os_reserve_gb)
    base = weight + activation + os_reserve
    if variant.min_ram_gb is not None:
        # Vendor min_ram already covers a short context; only charge extra tokens.
        kv_extra = max(0, context - baseline) * active_b * coeff
        total = max(float(variant.min_ram_gb) + kv_extra, base + kv_extra)
    else:
        total = base + kv
    headroom = available_gb - total
    return FitEstimate(
        weight_gb=round(weight, 2),
        kv_gb=round(kv, 2),
        activation_gb=round(activation, 2),
        os_reserve_gb=round(os_reserve, 2),
        total_gb=round(total, 2),
        available_gb=round(available_gb, 2),
        fits=headroom >= 0,
        headroom_gb=round(headroom, 2),
    )


def engine_order(catalog: Catalog, hw: HardwareInfo) -> list[str]:
    if hw.is_apple_silicon:
        return list(catalog.policy.apple_engine_order)
    if hw.is_spark:
        return list(catalog.policy.spark_engine_order)
    if hw.is_discrete_cuda:
        return list(catalog.policy.cuda_engine_order)
    if hw.platform == "linux":
        return list(catalog.policy.linux_engine_order)
    return ["llama.cpp"]


def engine_usable(hw: HardwareInfo, engine: str) -> tuple[bool, str]:
    if engine == "mlx":
        if not hw.is_apple_silicon:
            return False, "MLX requires Apple Silicon"
        return True, ""
    if engine == "vllm":
        if hw.platform not in {"linux", "windows"}:
            return False, "vLLM path targets Linux / CUDA hosts"
        if hw.platform == "windows":
            return False, "vLLM on Windows is experimental; prefer llama.cpp"
        if not hw.cuda_available and not hw.is_spark:
            return False, "CUDA / Spark required for vLLM"
        return True, ""
    if engine == "llama.cpp":
        return True, ""
    return False, f"unknown engine {engine}"


def match_target(catalog: Catalog, hw: HardwareInfo) -> str | None:
    for target in catalog.targets:
        if target.cluster:
            continue

        # Discrete NVIDIA: match VRAM band (linux/windows x86_64).
        if target.require_cuda or target.min_vram_gb is not None:
            if hw.is_spark or hw.is_apple_silicon:
                continue
            if not hw.cuda_available:
                continue
            if target.platform not in {hw.platform, "any", "*"}:
                # Allow linux-defined CUDA targets on Windows workstations too.
                if not (target.platform == "linux" and hw.platform == "windows"):
                    continue
            if target.arch == "x86_64" and hw.arch != "x86_64":
                continue
            vram = hw.vram_gb
            lo = target.min_vram_gb if target.min_vram_gb is not None else 0.0
            hi = target.max_vram_gb if target.max_vram_gb is not None else 1e9
            if not (lo <= vram <= hi):
                continue
            if target.cuda_capability_major and hw.cuda_capability:
                if hw.cuda_capability[0] < target.cuda_capability_major:
                    continue
            return target.id

        if target.platform != hw.platform:
            continue
        # arch soft-match: arm64 vs aarch64
        if target.arch == "arm64" and hw.arch not in {"arm64", "aarch64"}:
            continue
        if target.arch == "aarch64" and hw.arch not in {"aarch64", "arm64"}:
            continue
        if target.arch == "x86_64" and hw.arch != "x86_64":
            continue
        if target.min_ram_gb <= hw.ram_gb <= target.max_ram_gb:
            if target.cuda_capability_major and hw.cuda_capability:
                if hw.cuda_capability[0] != target.cuda_capability_major:
                    continue
            return target.id
    return None


def _score_candidate(
    model: ModelEntry,
    variant: ModelVariant,
    estimate: FitEstimate,
    profile: str,
    order: list[str],
    prefer_coding: bool,
    preferred_engine: str | None = None,
) -> float:
    if not estimate.fits:
        return -1e6
    score = 0.0
    if prefer_coding:
        score += model.coding_priority * 10
    if profile in variant.recommended_profiles:
        score += 25
    try:
        eng_rank = order.index(variant.engine)
        score += max(0, 30 - eng_rank * 10)
    except ValueError:
        score -= 5
    if preferred_engine and variant.engine == preferred_engine:
        score += 18
    # Prefer more headroom but not oversized waste
    score += min(estimate.headroom_gb, 40) * 0.5
    # Prefer smaller downloads when scores are close
    score -= variant.weight_gb * 0.05
    if variant.mtp and prefer_coding:
        score += 5
    if variant.preferred:
        score += 15
    score += variant.quality_bonus
    # Mild publisher preference (official / ggml / bartowski / mlx-community)
    provider_bonus = {
        "ggml-org": 8,
        "bartowski": 7,
        "mlx-community": 7,
        "Qwen": 6,
        "google": 6,
        "deepseek-ai": 6,
        "zai-org": 5,
        "mudler": 4,
        "lmstudio-community": 3,
        "unsloth": 2,
    }
    score += provider_bonus.get(variant.provider, 0)
    return score


def plan_model(
    catalog: Catalog,
    hw: HardwareInfo,
    model_id: str | None,
    profile: str = "coding",
    include_opt_in: bool = False,
    context_override: int | None = None,
) -> PlanResult:
    prefer_coding = profile == "coding"
    order = engine_order(catalog, hw)
    target_id = match_target(catalog, hw)
    target = get_target(catalog, target_id) if target_id else None
    preferred_engine = target.preferred_engine if target else None
    template_layer = target.engine_template if target else None
    avail_gb, reserve_gb = available_memory_gb(catalog, hw)
    # Large opt-in models: gate on usable budget (VRAM or unified RAM).
    opt_in_budget = avail_gb if hw.is_discrete_cuda else hw.ram_gb

    models: list[ModelEntry]
    if model_id and model_id != "auto":
        models = [get_model(catalog, model_id)]
    else:
        models = [
            m
            for m in catalog.models
            if include_opt_in or not m.opt_in or (m.opt_in and opt_in_budget >= 200)
        ]
        if prefer_coding:
            models = sorted(models, key=lambda m: m.coding_priority, reverse=True)

    candidates: list[PlanCandidate] = []
    for model in models:
        settings = profile_settings(catalog, profile, model.family)
        context = int(context_override or settings.get("context", 16384))
        context = min(context, model.max_context)
        for variant in model.variants:
            ok, reason = engine_usable(hw, variant.engine)
            estimate = estimate_memory(
                catalog, model, variant, context, avail_gb, reserve_gb=reserve_gb
            )
            sampling = {
                k: v
                for k, v in settings.items()
                if k not in {"context", "description"}
            }
            engine_args = resolve_engine_args(
                catalog,
                variant.engine,
                profile=profile,
                target_template=template_layer,
            )
            cand = PlanCandidate(
                model_id=model.id,
                display_name=model.display_name,
                engine=variant.engine,
                format=variant.format,
                repo=variant.repo,
                quant=variant.quant,
                profile=profile,
                context=context,
                weight_gb=variant.weight_gb,
                estimate=estimate,
                score=0.0,
                variant={
                    "include": list(variant.include),
                    "mmproj": variant.mmproj,
                    "mtp": variant.mtp,
                    "spark_moe_backend": variant.spark_moe_backend,
                    "min_ram_gb": variant.min_ram_gb,
                    "provider": variant.provider,
                    "preferred": variant.preferred,
                    "hub": False,
                    "engine_args": engine_args,
                },
                sampling=sampling,
            )
            if not ok:
                cand.rejected = True
                cand.reject_reason = reason
                cand.score = -1e6
                cand.reasons.append(reason)
            elif not estimate.fits:
                unit = "VRAM" if hw.is_discrete_cuda else "RAM"
                cand.rejected = True
                cand.reject_reason = (
                    f"needs ~{estimate.total_gb:.0f}GB {unit}, "
                    f"have {estimate.available_gb:.0f}GB"
                )
                cand.score = -1e6
                cand.reasons.append(cand.reject_reason)
            else:
                cand.score = _score_candidate(
                    model,
                    variant,
                    estimate,
                    profile,
                    order,
                    prefer_coding,
                    preferred_engine=preferred_engine,
                )
                cand.reasons.append(f"engine preference order: {order}")
                if preferred_engine:
                    cand.reasons.append(f"target preferred engine: {preferred_engine}")
                if hw.is_discrete_cuda:
                    cand.reasons.append(
                        f"fit against GPU VRAM {avail_gb:.0f}GB ({hw.gpu_name or 'CUDA'})"
                    )
                cand.reasons.append(
                    f"est {estimate.total_gb:.0f}GB "
                    f"(weights {estimate.weight_gb:.0f} + kv {estimate.kv_gb:.1f} "
                    f"+ act {estimate.activation_gb:.1f} + reserve {estimate.os_reserve_gb:.0f})"
                )
            candidates.append(cand)

    viable = [c for c in candidates if not c.rejected]
    viable.sort(key=lambda c: c.score, reverse=True)
    selected = viable[0] if viable else None

    hw_dict = {
        "platform": hw.platform,
        "arch": hw.arch,
        "hostname": hw.hostname,
        "ram_gb": hw.ram_gb,
        "vram_gb": hw.vram_gb,
        "gpu_count": hw.gpu_count,
        "is_apple_silicon": hw.is_apple_silicon,
        "is_spark": hw.is_spark,
        "is_discrete_cuda": hw.is_discrete_cuda,
        "cuda_available": hw.cuda_available,
        "cuda_capability": hw.cuda_capability,
        "gpu_name": hw.gpu_name,
    }
    return PlanResult(
        hardware=hw_dict,
        profile=profile,
        selected=selected,
        candidates=candidates,
        target_id=target_id,
    )


def save_plan(result: PlanResult, name: str = "last") -> Path:
    path = plans_dir() / f"{name}.json"
    payload = {
        "hardware": result.hardware,
        "profile": result.profile,
        "target_id": result.target_id,
        "selected": asdict(result.selected) if result.selected else None,
        "candidates": [asdict(c) for c in result.candidates],
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def load_plan(name: str = "last") -> dict[str, Any] | None:
    path = plans_dir() / f"{name}.json"
    if not path.is_file():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        return None
    return data
