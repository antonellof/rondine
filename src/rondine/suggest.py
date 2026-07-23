"""Hardware-aware model suggestions and launch configuration."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, cast

from rondine.catalog import Catalog, get_target, profile_settings, resolve_engine_args
from rondine.detect import HardwareInfo
from rondine.hub import (
    EngineHint,
    apply_hub_hardware_budget,
    curated_repo_ids,
    infer_model_family,
    inspect_repo,
    search_hub,
)
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
    source: str = "catalog"
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


def _default_hub_query(profile: str) -> str:
    return "coder" if profile == "coding" else "instruct"


def _hub_suggestions(
    catalog: Catalog,
    hw: HardwareInfo,
    *,
    profile: str,
    query: str,
    preferred_engine: str | None,
    target_template: str | None,
    existing_repos: set[str],
    limit: int,
    context_override: int | None,
) -> list[Suggestion]:
    engine = (
        cast(EngineHint, preferred_engine)
        if preferred_engine in {"llama.cpp", "mlx", "vllm"}
        else None
    )
    hits = search_hub(
        query,
        limit=max(limit * 3, 6),
        engine=engine,
        curated_repos=curated_repo_ids(catalog.models),
    )
    families = {model.family for model in catalog.models}
    suggestions: list[Suggestion] = []

    for hit in hits:
        if hit.repo_id in existing_repos or hit.curated:
            continue
        try:
            inspected = inspect_repo(hit.repo_id)
        except Exception:
            continue
        family = infer_model_family(hit.repo_id, families) or ""
        family_max_context = max(
            (
                model.max_context
                for model in catalog.models
                if model.family == family
            ),
            default=0,
        )
        if (
            context_override is not None
            and family_max_context
            and context_override > family_max_context
        ):
            continue
        settings = profile_settings(catalog, profile, family)
        context = int(context_override or settings.get("context", 32768))
        sampling = {
            k: v for k, v in settings.items() if k not in {"context", "description"}
        }
        selected = inspected.to_plan_selected(
            profile=profile,
            context=context,
            sampling=sampling,
        )
        engine_args = resolve_engine_args(
            catalog,
            str(selected["engine"]),
            profile=profile,
            target_template=target_template,
        )
        selected["engine_args"] = engine_args
        apply_hub_hardware_budget(selected, catalog, hw)
        estimate = selected["estimate"]
        if not estimate["fits"] or inspected.weight_gb <= 0:
            continue

        score = 55.0 + min(hit.score, 60.0)
        if preferred_engine and selected["engine"] == preferred_engine:
            score += 15.0
        score += min(float(estimate["headroom_gb"]), 30.0) * 0.25
        score -= inspected.weight_gb * 0.05
        reasons = [
            f"Hugging Face search match for {query!r}",
            f"{hit.downloads:,} Hub downloads",
            *(
                [
                    f"requested context: {context_override:,} tokens"
                    + (
                        f" (known {family} capability: {family_max_context:,})"
                        if family_max_context
                        else " (Hub capability not independently verified)"
                    )
                ]
                if context_override is not None
                else []
            ),
            *selected.get("reasons", []),
        ]
        selected["score"] = round(score, 2)
        selected["reasons"] = reasons
        provider = hit.repo_id.split("/")[0] if "/" in hit.repo_id else ""
        suggestions.append(
            Suggestion(
                rank=0,
                model_id=str(selected["model_id"]),
                display_name=hit.repo_id,
                engine=str(selected["engine"]),
                format=str(selected["format"]),
                quant=str(selected["quant"]),
                repo=hit.repo_id,
                provider=provider,
                profile=profile,
                context=context,
                score=round(score, 2),
                estimate=dict(estimate),
                engine_args=dict(selected.get("engine_args") or {}),
                sampling=sampling,
                reasons=reasons,
                next_steps=_next_steps(
                    hit.repo_id,
                    profile,
                    preset_hint=str(selected["model_id"]),
                ),
                source="huggingface",
                selected=selected,
            )
        )
        existing_repos.add(hit.repo_id)
        if len(suggestions) >= limit:
            break

    suggestions.sort(key=lambda suggestion: suggestion.score, reverse=True)
    return suggestions


def suggest_for_hardware(
    catalog: Catalog,
    hw: HardwareInfo,
    *,
    profile: str = "coding",
    limit: int = 5,
    include_opt_in: bool = False,
    include_hub: bool = False,
    hub_query: str | None = None,
    context_override: int | None = None,
) -> SuggestResult:
    """Rank fitting catalog and optional Hub configs for this machine."""
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
        context_override=context_override,
    )

    # Boost candidates that match the hardware target's suggested model ids.
    suggested_ids = set(target.suggested_models if target else [])
    viable = [
        c
        for c in result.candidates
        if not c.rejected
        and (context_override is None or c.context >= context_override)
    ]
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
                catalog,
                hw,
                mid,
                profile=profile,
                include_opt_in=True,
                context_override=context_override,
            )
            if explicit.selected is None or (
                context_override is not None
                and explicit.selected.context < context_override
            ):
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
    if include_hub and limit > 0:
        query = hub_query or _default_hub_query(profile)
        existing_repos = {suggestion.repo for suggestion in suggestions}
        hub_slots = 1 if limit < 6 else 2
        try:
            hub_suggestions = _hub_suggestions(
                catalog,
                hw,
                profile=profile,
                query=query,
                preferred_engine=preferred,
                target_template=template_layer,
                existing_repos=existing_repos,
                limit=hub_slots,
                context_override=context_override,
            )
        except Exception as exc:
            hub_suggestions = []
            notes.append(
                f"Hugging Face search unavailable; showing catalog results ({exc})"
            )
        if hub_suggestions:
            if limit == 1:
                suggestions = sorted(
                    [*suggestions, *hub_suggestions],
                    key=lambda suggestion: suggestion.score,
                    reverse=True,
                )[:1]
            else:
                suggestions = [
                    *suggestions[: max(0, limit - len(hub_suggestions))],
                    *hub_suggestions,
                ]
                suggestions.sort(key=lambda suggestion: suggestion.score, reverse=True)
            for rank, suggestion in enumerate(suggestions, start=1):
                suggestion.rank = rank
            notes.append(
                f"supplemented curated recommendations with Hugging Face search {query!r}"
            )
    if target and target.notes:
        notes.append(target.notes)
    if context_override is not None:
        notes.append(
            f"required context: {context_override:,} tokens; "
            "catalog models below this capability were excluded"
        )
    if missing:
        notes.append(
            "install missing engines with: rondine setup — "
            + ", ".join(missing)
        )
    if not suggestions:
        notes.append("no model fits; try lower --context or --opt-in")

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
