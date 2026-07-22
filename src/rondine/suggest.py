"""Hardware-aware model suggestions and launch configuration."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from rondine.catalog import Catalog, get_target, resolve_engine_args
from rondine.detect import HardwareInfo
from rondine.planner import PlanCandidate, match_target, plan_model


@dataclass
class Suggestion:
    rank: int
    model_id: str
    display_name: str
    engine: str
    format: str
    quant: str
    repo: str
    provider: str
    profile: str
    context: int
    score: float
    estimate: dict[str, Any]
    engine_args: dict[str, Any]
    sampling: dict[str, Any]
    reasons: list[str] = field(default_factory=list)
    next_steps: list[str] = field(default_factory=list)
    curated_hint: bool = False
    selected: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SuggestResult:
    hardware: dict[str, Any]
    target_id: str | None
    target_label: str | None
    profile: str
    preferred_engine: str | None
    engine_order: list[str]
    missing_engines: list[str]
    suggestions: list[Suggestion]
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "hardware": self.hardware,
            "target_id": self.target_id,
            "target_label": self.target_label,
            "profile": self.profile,
            "preferred_engine": self.preferred_engine,
            "engine_order": self.engine_order,
            "missing_engines": self.missing_engines,
            "suggestions": [s.to_dict() for s in self.suggestions],
            "notes": self.notes,
        }


def _candidate_to_selected(cand: PlanCandidate, engine_args: dict[str, Any]) -> dict[str, Any]:
    payload = asdict(cand)
    payload["engine_args"] = engine_args
    return payload


def _next_steps(model_id: str, profile: str, preset_hint: str | None = None) -> list[str]:
    steps = [
        f"rondine plan {model_id} --profile {profile}",
        "rondine setup",
        f"rondine pull {model_id}",
        f"rondine serve {model_id} --profile {profile}",
    ]
    if preset_hint:
        steps.append(f"rondine serve {model_id} --save-as {preset_hint}")
    else:
        steps.append(f"rondine serve {model_id} --save-as {model_id}")
    return steps


def suggest_for_hardware(
    catalog: Catalog,
    hw: HardwareInfo,
    *,
    profile: str = "coding",
    limit: int = 5,
    include_opt_in: bool = False,
) -> SuggestResult:
    """Rank fitting curated configs for this machine and attach engine knobs."""
    from rondine.planner import engine_order

    target_id = match_target(catalog, hw)
    target = get_target(catalog, target_id) if target_id else None
    order = engine_order(catalog, hw)
    preferred = target.preferred_engine if target else (order[0] if order else None)

    available = {e.name for e in hw.engines if e.available}
    missing = [e for e in order if e not in available]

    result = plan_model(
        catalog,
        hw,
        None,
        profile=profile,
        include_opt_in=include_opt_in,
    )

    # Boost candidates that match the hardware target's suggested model ids.
    suggested_ids = set(target.suggested_models if target else [])
    viable = [c for c in result.candidates if not c.rejected]
    for cand in viable:
        if cand.model_id in suggested_ids:
            cand.score += 40
            cand.reasons.append(f"recommended for target {target_id}")
        if preferred and cand.engine == preferred:
            cand.score += 20
            if f"preferred engine: {preferred}" not in cand.reasons:
                cand.reasons.append(f"preferred engine: {preferred}")
    viable.sort(key=lambda c: c.score, reverse=True)

    # Deduplicate by model_id — keep best engine/quant per model.
    seen: set[str] = set()
    picked: list[PlanCandidate] = []
    for cand in viable:
        if cand.model_id in seen:
            continue
        seen.add(cand.model_id)
        picked.append(cand)
        if len(picked) >= limit:
            break

    # If target lists models that didn't win auto-rank, try planning them explicitly.
    if target and len(picked) < limit:
        for mid in target.suggested_models:
            if mid in seen:
                continue
            explicit = plan_model(
                catalog, hw, mid, profile=profile, include_opt_in=True
            )
            if explicit.selected is None:
                continue
            seen.add(mid)
            picked.append(explicit.selected)
            if len(picked) >= limit:
                break

    template_layer = target.engine_template if target else None
    suggestions: list[Suggestion] = []
    for i, cand in enumerate(picked, start=1):
        engine_args = resolve_engine_args(
            catalog,
            cand.engine,
            profile=profile,
            target_template=template_layer,
            overrides=cand.variant.get("engine_args")
            if isinstance(cand.variant.get("engine_args"), dict)
            else None,
        )
        selected = _candidate_to_selected(cand, engine_args)
        provider = str((cand.variant or {}).get("provider") or "")
        suggestions.append(
            Suggestion(
                rank=i,
                model_id=cand.model_id,
                display_name=cand.display_name,
                engine=cand.engine,
                format=cand.format,
                quant=cand.quant,
                repo=cand.repo,
                provider=provider,
                profile=profile,
                context=cand.context,
                score=cand.score,
                estimate=asdict(cand.estimate),
                engine_args=engine_args,
                sampling=dict(cand.sampling),
                reasons=list(cand.reasons),
                next_steps=_next_steps(cand.model_id, profile),
                curated_hint=cand.model_id in suggested_ids,
                selected=selected,
            )
        )

    notes: list[str] = []
    if target and target.notes:
        notes.append(target.notes)
    if missing:
        notes.append(
            "install missing engines with: rondine setup — "
            + ", ".join(missing)
        )
    if not suggestions:
        notes.append("no curated model fits; try lower --context, --opt-in, or Hub search")

    return SuggestResult(
        hardware={
            "platform": hw.platform,
            "arch": hw.arch,
            "hostname": hw.hostname,
            "ram_gb": hw.ram_gb,
            "is_apple_silicon": hw.is_apple_silicon,
            "is_spark": hw.is_spark,
            "is_discrete_cuda": hw.is_discrete_cuda,
            "cuda_available": hw.cuda_available,
            "gpu_name": hw.gpu_name,
            "vram_gb": hw.vram_gb,
            "gpu_count": hw.gpu_count,
        },
        target_id=target_id,
        target_label=target.label if target else None,
        profile=profile,
        preferred_engine=preferred,
        engine_order=order,
        missing_engines=missing,
        suggestions=suggestions,
        notes=notes,
    )
