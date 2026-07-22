"""Rondine CLI."""

from __future__ import annotations

import json
import shutil
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import click

from rondine import __version__
from rondine.catalog import load_catalog, profile_settings
from rondine.cluster import ClusterInventory, ClusterNode, doctor_cluster, plan_cluster_commands
from rondine.detect import detect_hardware
from rondine.engines.base import EngineAdapter
from rondine.engines.llama_cpp import LlamaCppAdapter
from rondine.engines.mlx import MlxAdapter
from rondine.engines.vllm import VllmAdapter
from rondine.hub import curated_repo_ids, inspect_repo, search_hub
from rondine.paths import brand_logo, plans_dir
from rondine.planner import available_memory_gb, load_plan, plan_model, save_plan
from rondine.presets import (
    delete_preset,
    list_presets,
    load_preset,
    preset_from_selected,
    save_preset,
    selected_with_preset_overrides,
)
from rondine.process import is_running, load_record, start_server, stop_server
from rondine.suggest import suggest_for_hardware
from rondine.ux import spinner
from rondine.verify import verify_server

ADAPTERS: dict[str, EngineAdapter] = {
    "llama.cpp": LlamaCppAdapter(),
    "mlx": MlxAdapter(),
    "vllm": VllmAdapter(),
}


def _format_logo(width: int | None = None) -> str:
    """Center the logo when the terminal is wider than the artwork."""
    lines = brand_logo().splitlines()
    if not lines:
        return ""
    logo_width = max(len(line) for line in lines)
    terminal_width = width or shutil.get_terminal_size(fallback=(80, 24)).columns
    padding = " " * max(0, (terminal_width - logo_width) // 2)
    return "\n".join(f"{padding}{line}" for line in lines)


def _echo_logo() -> None:
    click.echo(_format_logo())
    click.echo()


def _print_estimate(est: dict[str, Any]) -> None:
    click.echo(
        f"  memory est: {est['total_gb']} GB "
        f"(weights {est['weight_gb']} + kv {est['kv_gb']} + "
        f"act {est['activation_gb']} + os {est['os_reserve_gb']}) "
        f"| available {est['available_gb']} GB | headroom {est['headroom_gb']} GB"
    )


def _print_engine_args(args: dict[str, Any], indent: str = "  ") -> None:
    if not args:
        return
    click.echo(f"{indent}engine config:")
    for key, value in sorted(args.items()):
        click.echo(f"{indent}  {key}: {value}")


def _selected_payload(result: Any) -> dict[str, Any] | None:
    if result.selected is None:
        return None
    return asdict(result.selected)


def _adapter_for(engine: str) -> EngineAdapter:
    if engine not in ADAPTERS:
        raise click.ClickException(f"unsupported engine: {engine}")
    return ADAPTERS[engine]


def _is_hf_repo(value: str) -> bool:
    return "/" in value and not value.startswith("auto")


def _save_hub_plan(selected: dict[str, Any], profile: str) -> Path:
    payload = {
        "hardware": {},
        "profile": profile,
        "target_id": None,
        "selected": selected,
        "candidates": [selected],
        "source": "huggingface",
    }
    path = plans_dir() / "last.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def _apply_hub_hardware_budget(
    selected: dict[str, Any],
    catalog: Any,
    hw: Any,
    *,
    memory_mode: str = "auto",
    allow_oversize: bool = False,
) -> tuple[float, str]:
    """Recalculate a Hub plan and apply an explicit memory strategy."""
    if memory_mode == "mmap" and not allow_oversize:
        raise click.ClickException("--memory-mode mmap requires --allow-oversize")
    available_gb, reserve_gb = available_memory_gb(catalog, hw)
    estimate = selected["estimate"]
    weight_gb = float(estimate["weight_gb"])
    activation_gb = float(estimate["activation_gb"])
    kv_gb = float(estimate.get("kv_gb") or 0.0)
    total_gb = round(weight_gb + activation_gb + kv_gb + reserve_gb, 2)
    resident_fits = available_gb >= total_gb
    mode = "resident"
    if (
        memory_mode == "hybrid"
        or (
            memory_mode == "auto"
            and not resident_fits
            and hw.is_discrete_cuda
            and selected.get("engine") == "llama.cpp"
            and selected.get("format") == "gguf"
        )
    ):
        if (
            not hw.is_discrete_cuda
            or selected.get("engine") != "llama.cpp"
            or selected.get("format") != "gguf"
        ):
            raise click.ClickException(
                "hybrid mode requires llama.cpp GGUF on a discrete CUDA host"
            )
        mode = "hybrid"
        available_gb = hw.ram_gb + hw.vram_gb
        reserve_gb = catalog.policy.os_reserve_gb + catalog.policy.vram_reserve_gb
        total_gb = round(weight_gb + activation_gb + kv_gb + reserve_gb, 2)
        selected["engine_args"] = {
            **dict(selected.get("engine_args") or {}),
            "n_gpu_layers": "auto",
            "fit": True,
            "fit_target": 1536,
            "mmap": True,
            "mlock": False,
        }
    elif memory_mode == "mmap":
        if selected.get("engine") != "llama.cpp" or selected.get("format") != "gguf":
            raise click.ClickException("mmap mode requires a llama.cpp GGUF repo")
        mode = "mmap"
        selected["experimental"] = True
        selected["warnings"] = [
            "experimental SSD demand paging; likely unusably slow and not a supported fit"
        ]
        selected["engine_args"] = {
            **dict(selected.get("engine_args") or {}),
            "n_gpu_layers": 0 if hw.is_apple_silicon else "auto",
            "fit": not hw.is_apple_silicon,
            "fit_target": 2048,
            "mmap": True,
            "mlock": False,
            "parallel": 1,
            "batch_size": 128,
            "ubatch_size": 64,
            "cache_type_k": "q4_1",
            "cache_type_v": "q4_1",
        }
    disk_required = round(weight_gb * 1.05, 2)
    estimate.update(
        {
            "os_reserve_gb": round(reserve_gb, 2),
            "total_gb": total_gb,
            "available_gb": available_gb,
            "headroom_gb": round(available_gb - total_gb, 2),
            "fits": available_gb >= total_gb,
            "memory_mode": mode,
            "experimental": mode == "mmap",
            "resident_shortfall_gb": round(max(0.0, total_gb - available_gb), 2),
            "disk_required_gb": disk_required,
            "disk_available_gb": hw.disk_free_gb,
        }
    )
    selected["memory_mode"] = mode
    if mode == "mmap" and hw.disk_free_gb < disk_required:
        raise click.ClickException(
            f"insufficient disk: need ~{disk_required:.0f}GB free, "
            f"have {hw.disk_free_gb:.0f}GB"
        )
    unit = "VRAM" if hw.is_discrete_cuda else "RAM"
    if mode == "hybrid":
        unit = "combined RAM+VRAM"
    return total_gb, unit


def _plan_from_hub(
    repo: str,
    *,
    profile: str,
    context: int | None,
    quant: str | None,
    memory_mode: str = "auto",
    allow_oversize: bool = False,
) -> dict[str, Any]:
    catalog = load_catalog()
    hw = detect_hardware()
    with spinner(f"inspecting Hub repo {repo}"):
        inspected = inspect_repo(repo, prefer_quant=quant)
    settings: dict[str, Any] = {"context": context or 32768}
    repo_l = repo.lower().replace("_", "-")
    family_map = {
        "qwen3.6": "qwen3.6",
        "qwen3-6": "qwen3.6",
        "gemma-4": "gemma-4",
        "deepseek-v4": "deepseek-v4",
        "glm-5.2": "glm-5.2",
        "glm-5-2": "glm-5.2",
    }
    for needle, family in family_map.items():
        if needle in repo_l:
            try:
                settings = profile_settings(catalog, profile, family)
            except KeyError:
                pass
            break
    ctx = int(context or settings.get("context", 32768))
    sampling = {k: v for k, v in settings.items() if k not in {"context", "description"}}
    selected = inspected.to_plan_selected(profile=profile, context=ctx, sampling=sampling)
    total, unit = _apply_hub_hardware_budget(
        selected,
        catalog,
        hw,
        memory_mode=memory_mode,
        allow_oversize=allow_oversize,
    )
    if not selected["estimate"]["fits"]:
        click.echo(
            f"warning: Hub model needs ~{total:.0f}GB {unit}, "
            f"machine has {selected['estimate']['available_gb']:.0f}GB",
            err=True,
        )
    _save_hub_plan(selected, profile)
    return selected


def _resolve_model_arg(
    model: str | None,
    *,
    profile: str,
    context: int | None = None,
    quant: str | None = None,
    require_fit: bool = False,
    memory_mode: str = "auto",
    allow_oversize: bool = False,
) -> dict[str, Any]:
    """Resolve curated id or org/name Hub repo into a selected plan dict."""
    catalog = load_catalog()
    hw = detect_hardware()
    if model and _is_hf_repo(model):
        selected = _plan_from_hub(
            model,
            profile=profile,
            context=context,
            quant=quant,
            memory_mode=memory_mode,
            allow_oversize=allow_oversize,
        )
        if require_fit and not selected["estimate"]["fits"] and not selected.get("experimental"):
            raise click.ClickException(
                f"Hub model does not fit (~{selected['estimate']['total_gb']}GB needed)"
            )
        return selected
    if model:
        result = plan_model(
            catalog,
            hw,
            model,
            profile=profile,
            include_opt_in=True,
            context_override=context,
            memory_mode=memory_mode,
            allow_oversize=allow_oversize,
        )
        if result.selected is None:
            raise click.ClickException(
                f"unknown curated model '{model}'. "
                f"Use a catalog id or a Hub repo like org/name. Try: rondine search {model}"
            )
        save_plan(result)
        payload = asdict(result.selected)
        if isinstance(payload.get("variant"), dict):
            payload["engine_args"] = dict(payload["variant"].get("engine_args") or {})
        return payload
    plan_data = load_plan()
    existing = (plan_data or {}).get("selected")
    if isinstance(existing, dict):
        return existing
    result = plan_model(catalog, hw, None, profile=profile, context_override=context)
    if result.selected is None:
        raise click.ClickException("no fitting model; pass a catalog id or Hub repo")
    save_plan(result)
    payload = asdict(result.selected)
    if isinstance(payload.get("variant"), dict):
        payload["engine_args"] = dict(payload["variant"].get("engine_args") or {})
    return payload


def _maybe_save_preset(
    save_as: str | None,
    selected: dict[str, Any],
    *,
    profile: str,
    host: str,
    port: int,
    run_name: str,
    target_id: str | None = None,
    quiet: bool = False,
) -> None:
    if not save_as:
        return
    preset = preset_from_selected(
        save_as,
        selected,
        profile=profile,
        host=host,
        port=port,
        run_name=run_name,
        target_id=target_id,
        engine_args=dict(selected.get("engine_args") or {}),
    )
    path = save_preset(preset)
    if not quiet:
        click.echo(f"saved preset: {path}")
        click.echo(f"restart later: rondine preset serve {save_as}")


@click.group()
@click.version_option(__version__, prog_name="rondine")
def main() -> None:
    """Hardware-aware local LLM launcher for Mac, NVIDIA GPUs, and DGX Spark."""


@main.command()
def doctor() -> None:
    """Probe hardware, memory, and installed engines."""
    _echo_logo()
    with spinner("scanning hardware"):
        hw = detect_hardware()
        catalog = load_catalog()
    click.echo(f"host: {hw.hostname}")
    click.echo(f"platform: {hw.platform}/{hw.arch}")
    click.echo(f"cpu: {hw.cpu_brand or '(unknown)'}")
    click.echo(f"ram: {hw.ram_gb} GB")
    click.echo(f"disk free: {hw.disk_free_gb} GB / {hw.disk_total_gb} GB")
    click.echo(f"apple silicon: {hw.is_apple_silicon}")
    click.echo(f"dgx spark: {hw.is_spark}")
    if hw.cuda_available:
        cap = ".".join(str(x) for x in hw.cuda_capability) if hw.cuda_capability else "?"
        click.echo(f"cuda: yes ({hw.gpu_name or 'gpu'}, compute {cap})")
        if hw.vram_gb:
            click.echo(f"vram: {hw.vram_gb} GB" + (f" ×{hw.gpu_count}" if hw.gpu_count > 1 else ""))
        if hw.is_discrete_cuda:
            click.echo(f"fit budget: {hw.usable_ram_gb} GB VRAM (discrete GPU)")
    else:
        click.echo("cuda: no")
    click.echo(f"metal: {hw.metal_available}")
    click.echo("engines:")
    for eng in hw.engines:
        status = "ok" if eng.available else "missing"
        extra = eng.version or eng.detail or eng.path or ""
        click.echo(f"  - {eng.name}: {status} {extra}".rstrip())
    from rondine.planner import match_target

    target = match_target(catalog, hw)
    click.echo(f"matched target: {target or '(none)'}")
    for w in hw.warnings:
        click.echo(f"warning: {w}", err=True)
    click.echo()
    click.echo("next: rondine suggest --profile coding")


@main.command()
@click.option(
    "--profile",
    type=click.Choice(["coding", "chat"]),
    default="coding",
    show_default=True,
    help="Workload preset: coding uses 32K context; chat uses 16K and more parallel slots.",
)
@click.option("--limit", default=5, show_default=True, help="How many model configs to show.")
@click.option(
    "--opt-in/--no-opt-in",
    default=False,
    help="Include oversized or specialist models excluded from normal recommendations.",
)
@click.option(
    "--json",
    "as_json",
    is_flag=True,
    help="Print the complete suggestion result as JSON for scripts.",
)
@click.option(
    "--configure",
    "configure_rank",
    type=int,
    default=None,
    help="Save suggestion rank N as the active plan.",
)
@click.option(
    "--save-as",
    default=None,
    metavar="NAME",
    help="With --configure, also save a reusable named preset.",
)
def suggest(
    profile: str,
    limit: int,
    opt_in: bool,
    as_json: bool,
    configure_rank: int | None,
    save_as: str | None,
) -> None:
    """Suggest models that fit this machine and show launch configs."""
    catalog = load_catalog()
    with spinner("matching models to hardware"):
        hw = detect_hardware()
        result = suggest_for_hardware(
            catalog, hw, profile=profile, limit=limit, include_opt_in=opt_in
        )

    if as_json:
        click.echo(json.dumps(result.to_dict(), indent=2))
        return

    _echo_logo()
    click.echo(f"target: {result.target_id or '(generic)'}  ({result.target_label or 'unmatched'})")
    ram = result.hardware.get("ram_gb")
    vram = result.hardware.get("vram_gb")
    if result.hardware.get("is_discrete_cuda") and vram:
        click.echo(
            f"vram: {vram} GB  ({result.hardware.get('gpu_name') or 'CUDA'})  "
            f"system ram: {ram} GB  profile: {profile}"
        )
    else:
        click.echo(f"ram: {ram} GB  profile: {profile}")
    click.echo(f"engines: {' → '.join(result.engine_order)}")
    if result.preferred_engine:
        click.echo(f"preferred engine: {result.preferred_engine}")
    if result.missing_engines:
        click.echo(f"missing engines: {', '.join(result.missing_engines)}")
    click.echo()

    if not result.suggestions:
        click.echo("no fitting suggestions")
        for note in result.notes:
            click.echo(f"note: {note}")
        sys.exit(1)

    click.echo("recommended configs:")
    for s in result.suggestions:
        star = "*" if s.curated_hint else " "
        click.echo(
            f"{star}#{s.rank}  {s.display_name} ({s.model_id})  score={s.score:.1f}"
        )
        click.echo(
            f"     engine={s.engine}  format={s.format}  quant={s.quant}  "
            f"provider={s.provider}"
        )
        click.echo(f"     repo={s.repo}")
        click.echo(
            f"     context={s.context}  "
            f"~{s.estimate['total_gb']}GB "
            f"(headroom {s.estimate['headroom_gb']}GB)"
        )
        _print_engine_args(s.engine_args, indent="     ")
        if s.sampling:
            samp = ", ".join(
                f"{k}={v}"
                for k, v in s.sampling.items()
                if k != "chat_template_kwargs" and not isinstance(v, dict)
            )
            if samp:
                click.echo(f"     sampling: {samp}")
        click.echo(f"     run: rondine serve {s.model_id} --profile {profile}")
        click.echo()

    for note in result.notes:
        click.echo(f"note: {note}")
    click.echo("tip: rondine suggest --configure 1 --save-as coding")
    click.echo("tip: rondine search \"Qwen3.6 GGUF\"  # discover more on Hub")

    if configure_rank is None:
        return
    match = next((s for s in result.suggestions if s.rank == configure_rank), None)
    if match is None:
        raise click.ClickException(f"no suggestion #{configure_rank}")
    # Persist as last plan so pull/serve work without re-args.
    path = plans_dir() / "last.json"
    path.write_text(
        json.dumps(
            {
                "hardware": result.hardware,
                "profile": profile,
                "target_id": result.target_id,
                "selected": match.selected,
                "candidates": [match.selected],
                "source": "suggest",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    click.echo(f"configured plan: {path}")
    click.echo(f"  {match.model_id} via {match.engine}/{match.quant}")
    host = catalog.policy.default_host
    port = catalog.policy.default_port
    _maybe_save_preset(
        save_as,
        match.selected,
        profile=profile,
        host=host,
        port=port,
        run_name=save_as or "default",
        target_id=result.target_id,
    )
    click.echo("next: rondine setup && rondine pull && rondine serve")


@main.command("models")
@click.option("--profile", default="coding", show_default=True)
@click.option("--opt-in/--no-opt-in", default=False, help="Include large opt-in models.")
def models_cmd(profile: str, opt_in: bool) -> None:
    """List curated catalog models and whether they fit this machine."""
    catalog = load_catalog()
    hw = detect_hardware()
    result = plan_model(catalog, hw, None, profile=profile, include_opt_in=opt_in)
    click.echo(f"profile={profile} ram={hw.ram_gb}GB  (curated allowlist)")
    click.echo("tip: rondine suggest  # ranked configs for this machine")
    seen: set[str] = set()
    for cand in result.candidates:
        if cand.model_id in seen and cand.rejected:
            continue
        mark = "FIT" if not cand.rejected else "NO"
        provider = (cand.variant or {}).get("provider") or ""
        if cand.model_id not in seen:
            seen.add(cand.model_id)
            model = next(m for m in catalog.models if m.id == cand.model_id)
            opt = " [opt-in]" if model.opt_in else ""
            click.echo(f"{mark:3} {cand.model_id}{opt} — {cand.display_name}")
        detail = cand.reject_reason or f"{cand.engine}/{cand.quant} score={cand.score:.1f}"
        click.echo(f"     {provider:18} {cand.engine:10} {cand.quant:16} {detail}")


@main.command()
@click.argument("query")
@click.option(
    "--engine",
    type=click.Choice(["llama.cpp", "mlx", "vllm"]),
    default=None,
    help="Filter Hub results toward an engine/format.",
)
@click.option("--limit", default=15, show_default=True)
@click.option("--json", "as_json", is_flag=True)
def search(query: str, engine: str | None, limit: int, as_json: bool) -> None:
    """Search Hugging Face for GGUF / MLX / safetensors repos."""
    catalog = load_catalog()
    curated = curated_repo_ids(catalog.models)
    with spinner(f"searching Hub for {query!r}"):
        hits = search_hub(query, limit=limit, engine=engine, curated_repos=curated)  # type: ignore[arg-type]
    if as_json:
        click.echo(json.dumps([h.to_dict() for h in hits], indent=2))
        return
    if not hits:
        click.echo("no Hub hits")
        return
    click.echo(f"Hugging Face search: {query!r}  (curated marked with *)")
    for hit in hits:
        star = "*" if hit.curated else " "
        click.echo(
            f"{star} {hit.repo_id:55} {hit.engine_hint:10} "
            f"dl={hit.downloads:<8} score={hit.score:.1f}"
        )
    click.echo("next: rondine inspect <org/name>  |  rondine plan <org/name>")


@main.command()
@click.argument("repo")
@click.option("--quant", default=None, help="Prefer a quant label (e.g. Q4_K_M).")
@click.option("--json", "as_json", is_flag=True)
def inspect(repo: str, quant: str | None, as_json: bool) -> None:
    """Inspect a Hub repo: files, sizes, recommended quant."""
    with spinner(f"inspecting {repo}"):
        result = inspect_repo(repo, prefer_quant=quant)
    if as_json:
        click.echo(
            json.dumps(
                {
                    "repo_id": result.repo_id,
                    "engine_hint": result.engine_hint,
                    "format_hint": result.format_hint,
                    "recommended_quant": result.recommended_quant,
                    "recommended_file": result.recommended_file,
                    "weight_gb": result.weight_gb,
                    "downloads": result.downloads,
                    "files": [asdict(f) for f in result.files[:40]],
                    "notes": result.notes,
                },
                indent=2,
            )
        )
        return
    click.echo(f"repo: {result.repo_id}")
    click.echo(f"engine/format: {result.engine_hint} / {result.format_hint}")
    click.echo(f"downloads: {result.downloads}")
    click.echo(
        f"recommended: {result.recommended_quant} "
        f"({result.recommended_file}) ~{result.weight_gb} GB"
    )
    for note in result.notes:
        click.echo(f"note: {note}")
    click.echo("files:")
    for f in result.files[:20]:
        flag = " mmproj" if f.is_mmproj else ""
        q = f.quant or "-"
        click.echo(f"  {f.size_gb:8.2f} GB  {q:12} {f.path}{flag}")
    if len(result.files) > 20:
        click.echo(f"  … {len(result.files) - 20} more")


@main.command()
@click.argument("model", default="auto")
@click.option("--profile", default="coding", show_default=True)
@click.option("--context", type=int, default=None, help="Override context length.")
@click.option("--quant", default=None, help="Prefer Hub quant when model is org/name.")
@click.option("--opt-in/--no-opt-in", default=False)
@click.option(
    "--memory-mode",
    type=click.Choice(["auto", "resident", "hybrid", "mmap"]),
    default="auto",
    show_default=True,
)
@click.option(
    "--allow-oversize",
    is_flag=True,
    help="Acknowledge experimental SSD paging for --memory-mode mmap.",
)
@click.option("--save-as", default=None, help="Also save a named preset from this plan.")
@click.option("--json", "as_json", is_flag=True)
def plan(
    model: str,
    profile: str,
    context: int | None,
    quant: str | None,
    opt_in: bool,
    memory_mode: str,
    allow_oversize: bool,
    save_as: str | None,
    as_json: bool,
) -> None:
    """Recommend engine / format / quant (catalog id, auto, or Hub org/name)."""
    if memory_mode == "mmap" and not allow_oversize:
        raise click.ClickException("--memory-mode mmap requires --allow-oversize")
    catalog = load_catalog()
    hw = detect_hardware()

    if _is_hf_repo(model):
        selected = _plan_from_hub(
            model,
            profile=profile,
            context=context,
            quant=quant,
            memory_mode=memory_mode,
            allow_oversize=allow_oversize,
        )
        _maybe_save_preset(
            save_as,
            selected,
            profile=profile,
            host=catalog.policy.default_host,
            port=catalog.policy.default_port,
            run_name=save_as or "default",
            quiet=as_json,
        )
        if as_json:
            click.echo(json.dumps({"selected": selected, "source": "huggingface"}, indent=2))
            return
        _echo_logo()
        click.echo("source: Hugging Face Hub")
        click.echo(f"selected: {selected['display_name']}")
        click.echo(
            f"  engine: {selected['engine']}  format: {selected['format']}  "
            f"quant: {selected['quant']}"
        )
        click.echo(f"  repo: {selected['repo']}")
        click.echo(f"  context: {selected['context']}")
        _print_estimate(selected["estimate"])
        _print_engine_args(selected.get("engine_args") or {})
        for reason in selected.get("reasons") or []:
            click.echo(f"  note: {reason}")
        click.echo(f"saved plan: {plans_dir() / 'last.json'}")
        return

    result = plan_model(
        catalog,
        hw,
        None if model == "auto" else model,
        profile=profile,
        include_opt_in=opt_in or model != "auto",
        context_override=context,
        memory_mode=memory_mode,
        allow_oversize=allow_oversize,
    )
    path = save_plan(result)
    if as_json:
        if result.selected is not None:
            payload = asdict(result.selected)
            engine_args = (result.selected.variant or {}).get("engine_args") or {}
            payload["engine_args"] = engine_args if isinstance(engine_args, dict) else {}
            _maybe_save_preset(
                save_as,
                payload,
                profile=profile,
                host=catalog.policy.default_host,
                port=catalog.policy.default_port,
                run_name=save_as or "default",
                target_id=result.target_id,
                quiet=True,
            )
        click.echo(
            json.dumps(
                {
                    "selected": _selected_payload(result),
                    "target_id": result.target_id,
                    "saved": str(path),
                    "source": "catalog",
                },
                indent=2,
            )
        )
        return
    _echo_logo()
    click.echo("source: curated catalog")
    click.echo(f"target: {result.target_id or '(generic)'}")
    click.echo(f"profile: {profile}")
    if result.selected is None:
        click.echo("no fitting candidate — try --opt-in, lower --context, or Hub search")
        rejected = [c for c in result.candidates if c.rejected][:8]
        for c in rejected:
            click.echo(f"  rejected {c.model_id} {c.engine}/{c.quant}: {c.reject_reason}")
        sys.exit(1)
    sel = result.selected
    provider = (sel.variant or {}).get("provider") or ""
    click.echo(f"selected: {sel.display_name} ({sel.model_id})")
    click.echo(
        f"  engine: {sel.engine}  format: {sel.format}  quant: {sel.quant}  "
        f"provider: {provider}"
    )
    click.echo(f"  repo: {sel.repo}")
    click.echo(f"  context: {sel.context}")
    click.echo(
        f"  memory mode: {sel.memory_mode}"
        + (" (EXPERIMENTAL)" if sel.experimental else "")
    )
    _print_estimate(asdict(sel.estimate))
    engine_args = (sel.variant or {}).get("engine_args") or {}
    _print_engine_args(engine_args if isinstance(engine_args, dict) else {})
    for reason in sel.reasons:
        click.echo(f"  note: {reason}")
    for warning in sel.warnings:
        click.echo(f"  warning: {warning}", err=True)
    click.echo(f"saved plan: {path}")
    payload = asdict(sel)
    payload["engine_args"] = engine_args if isinstance(engine_args, dict) else {}
    _maybe_save_preset(
        save_as,
        payload,
        profile=profile,
        host=catalog.policy.default_host,
        port=catalog.policy.default_port,
        run_name=save_as or "default",
        target_id=result.target_id,
    )


@main.command()
@click.option("--engine", "engines", multiple=True, type=click.Choice(sorted(ADAPTERS)))
@click.option("--dry-run", is_flag=True)
def setup(engines: tuple[str, ...], dry_run: bool) -> None:
    """Install pinned engine toolchains for this machine."""
    hw = detect_hardware()
    if not engines:
        chosen: list[str] = ["llama.cpp"]
        if hw.is_apple_silicon:
            chosen.append("mlx")
        if hw.is_spark or (hw.platform == "linux" and hw.cuda_available):
            chosen.append("vllm")
        engines = tuple(chosen)
    for name in engines:
        click.echo(f"== setup {name} ==")
        adapter = _adapter_for(name)
        with spinner(f"setting up {name}", enabled=not dry_run):
            lines = adapter.setup(dry_run=dry_run)
        for line in lines:
            click.echo(line)


@main.command()
@click.argument("model", required=False)
@click.option("--profile", default="coding", show_default=True)
@click.option("--quant", default=None, help="Prefer Hub quant when model is org/name.")
@click.option(
    "--memory-mode",
    type=click.Choice(["auto", "resident", "hybrid", "mmap"]),
    default="auto",
    show_default=True,
)
@click.option("--allow-oversize", is_flag=True)
@click.option("--dry-run", is_flag=True)
def pull(
    model: str | None,
    profile: str,
    quant: str | None,
    memory_mode: str,
    allow_oversize: bool,
    dry_run: bool,
) -> None:
    """Download weights for a catalog id or Hub org/name."""
    selected = _resolve_model_arg(
        model,
        profile=profile,
        quant=quant,
        memory_mode=memory_mode,
        allow_oversize=allow_oversize,
    )
    adapter = _adapter_for(str(selected["engine"]))
    click.echo(f"pulling {selected['repo']} via {selected['engine']} ...")
    with spinner(f"downloading {selected['repo']}", enabled=not dry_run):
        lines = adapter.pull({"selected": selected}, dry_run=dry_run)
    for line in lines:
        click.echo(line)


@main.command()
@click.argument("model", required=False)
@click.option("--profile", default="coding", show_default=True)
@click.option("--quant", default=None, help="Prefer Hub quant when model is org/name.")
@click.option("--host", default=None)
@click.option("--port", default=None, type=int)
@click.option("--dry-run", is_flag=True)
@click.option("--foreground", is_flag=True, help="Attach to server process.")
@click.option("--name", default="default", show_default=True, help="Run record name.")
@click.option("--preset", "preset_name", default=None, help="Serve a saved preset.")
@click.option("--save-as", default=None, help="Save this launch as a named preset.")
@click.option(
    "--memory-mode",
    type=click.Choice(["auto", "resident", "hybrid", "mmap"]),
    default="auto",
    show_default=True,
)
@click.option("--allow-oversize", is_flag=True)
def serve(
    model: str | None,
    profile: str,
    quant: str | None,
    host: str | None,
    port: int | None,
    dry_run: bool,
    foreground: bool,
    name: str,
    preset_name: str | None,
    save_as: str | None,
    memory_mode: str,
    allow_oversize: bool,
) -> None:
    """Launch an OpenAI-compatible local server (catalog id, Hub org/name, or preset)."""
    catalog = load_catalog()
    target_id = None
    if preset_name:
        preset = load_preset(preset_name)
        selected = selected_with_preset_overrides(preset)
        profile = preset.profile
        host = host or preset.host
        port = port or preset.port
        name = name if name != "default" else preset.run_name
        target_id = preset.target_id
    else:
        selected = _resolve_model_arg(
            model,
            profile=profile,
            quant=quant,
            require_fit=bool(model),
            memory_mode=memory_mode,
            allow_oversize=allow_oversize,
        )
        plan_data = load_plan()
        target_id = (plan_data or {}).get("target_id")
    host = host or catalog.policy.default_host
    port = port or catalog.policy.default_port
    adapter = _adapter_for(str(selected["engine"]))
    if selected.get("experimental"):
        for warning in selected.get("warnings") or [
            "experimental oversized launch"
        ]:
            click.echo(f"warning: {warning}", err=True)
    spec = adapter.build_serve({"selected": selected}, host=host, port=port)
    click.echo(f"engine: {spec.engine}")
    click.echo(f"command: {spec.command_line()}")
    for note in spec.notes:
        click.echo(f"note: {note}")
    click.echo(f"openai base url: {spec.base_url}")
    _maybe_save_preset(
        save_as,
        selected,
        profile=profile,
        host=host,
        port=port,
        run_name=name,
        target_id=str(target_id) if target_id else None,
    )
    if dry_run:
        return
    with spinner(f"starting {spec.engine} on :{port}", enabled=not foreground):
        record = start_server(spec, name=name, foreground=foreground)
    if not foreground:
        click.echo(f"started pid={record.pid} log={record.log_path}")
        click.echo(f"stop with: rondine stop --name {name}")


@main.command()
@click.option("--name", default="default", show_default=True)
def stop(name: str) -> None:
    """Stop a managed server."""
    result = stop_server(name)
    if not result.get("ok"):
        raise click.ClickException(result.get("error") or "stop failed")
    click.echo(result.get("message", "stopped"))


@main.command()
@click.option("--profile", default="coding", show_default=True)
@click.option("--base-url", default=None)
@click.option("--name", default="default", show_default=True)
@click.option("--timeout", default=120, show_default=True)
def verify(profile: str, base_url: str | None, name: str, timeout: int) -> None:
    """Health check + coding smoke tests against a running server."""
    if base_url is None:
        record = load_record(name)
        if record is None or not is_running(record):
            raise click.ClickException("no running server; pass --base-url or rondine serve first")
        base_url = f"http://{record.host}:{record.port}/v1"
        model = record.alias or record.model_id
    else:
        model = None
        if not base_url.rstrip("/").endswith("/v1"):
            base_url = base_url.rstrip("/") + "/v1"
    with spinner(f"verifying {base_url}"):
        result = verify_server(base_url, model=model, profile=profile, timeout=timeout)
    for check in result.checks:
        mark = "PASS" if check["ok"] else "FAIL"
        click.echo(f"{mark} {check['name']}: {check['detail']}")
    if not result.ok:
        sys.exit(1)


@main.group()
def preset() -> None:
    """Save and restart named launch presets."""


@preset.command("list")
def preset_list() -> None:
    """List saved presets under ~/.rondine/presets."""
    items = list_presets()
    if not items:
        click.echo("no presets yet — try: rondine suggest --configure 1 --save-as coding")
        return
    for p in items:
        eng = p.selected.get("engine", "?")
        quant = p.selected.get("quant", "?")
        mid = p.selected.get("model_id") or p.selected.get("display_name") or "?"
        mode = p.selected.get("memory_mode", "resident")
        marker = " [EXPERIMENTAL]" if p.selected.get("experimental") else ""
        click.echo(
            f"{p.name:20} {mid:24} {eng}/{quant}  "
            f"{p.host}:{p.port}  profile={p.profile} mode={mode}{marker}"
        )


@preset.command("show")
@click.argument("name")
def preset_show(name: str) -> None:
    """Show a saved preset."""
    p = load_preset(name)
    click.echo(json.dumps(p.to_dict(), indent=2))


@preset.command("delete")
@click.argument("name")
def preset_delete(name: str) -> None:
    """Delete a saved preset."""
    delete_preset(name)
    click.echo(f"deleted preset {name}")


@preset.command("save")
@click.argument("name")
@click.option("--profile", default="coding", show_default=True)
@click.option("--host", default=None)
@click.option("--port", default=None, type=int)
@click.option("--run-name", default="default", show_default=True)
def preset_save(
    name: str, profile: str, host: str | None, port: int | None, run_name: str
) -> None:
    """Save the current plan (last.json) as a named preset."""
    catalog = load_catalog()
    plan_data = load_plan()
    selected = (plan_data or {}).get("selected")
    if not isinstance(selected, dict):
        raise click.ClickException("no saved plan; run rondine suggest/plan first")
    host = host or catalog.policy.default_host
    port = port or catalog.policy.default_port
    preset = preset_from_selected(
        name,
        selected,
        profile=str((plan_data or {}).get("profile") or profile),
        host=host,
        port=port,
        run_name=run_name,
        target_id=(plan_data or {}).get("target_id"),
        engine_args=dict(selected.get("engine_args") or {}),
    )
    path = save_preset(preset)
    click.echo(f"saved preset: {path}")


@preset.command("serve")
@click.argument("name")
@click.option("--dry-run", is_flag=True)
@click.option("--foreground", is_flag=True)
def preset_serve(name: str, dry_run: bool, foreground: bool) -> None:
    """Restart a server from a saved preset."""
    ctx = click.get_current_context()
    ctx.invoke(
        serve,
        model=None,
        profile="coding",
        quant=None,
        host=None,
        port=None,
        dry_run=dry_run,
        foreground=foreground,
        name="default",
        preset_name=name,
        save_as=None,
        memory_mode="auto",
        allow_oversize=False,
    )


@main.group()
def cluster() -> None:
    """Homogeneous dual-node cluster helpers."""


@cluster.command("init")
@click.argument("name")
@click.option("--kind", type=click.Choice(["mac", "spark"]), required=True)
@click.option("--head", "head_host", required=True)
@click.option("--worker", "workers", multiple=True, required=True)
@click.option("--user", default=None)
@click.option("--interface", default=None, help="RoCE/QSFP interface (Spark).")
def cluster_init(
    name: str,
    kind: str,
    head_host: str,
    workers: tuple[str, ...],
    user: str | None,
    interface: str | None,
) -> None:
    """Write a cluster inventory file under ~/.rondine/clusters/."""
    nodes = [ClusterNode(host=head_host, user=user, role="head")]
    nodes.extend(ClusterNode(host=w, user=user, role="worker") for w in workers)
    inv = ClusterInventory(name=name, kind=kind, nodes=nodes, interface=interface)
    path = inv.save()
    click.echo(f"wrote {path}")


@cluster.command("doctor")
@click.argument("name")
def cluster_doctor_cmd(name: str) -> None:
    """Validate SSH and inventory constraints."""
    inv = ClusterInventory.load(name)
    report = doctor_cluster(inv)
    click.echo(json.dumps(report, indent=2))
    if not report["ok"]:
        sys.exit(1)


@cluster.command("plan")
@click.argument("name")
@click.option("--model-repo", required=True, help="HF repo id for the chosen format.")
@click.option("--engine", type=click.Choice(["mlx", "vllm", "llama.cpp"]), required=True)
@click.option("--port", default=8080, show_default=True)
def cluster_plan_cmd(name: str, model_repo: str, engine: str, port: int) -> None:
    """Print upstream multi-node launch commands (no custom scheduler)."""
    inv = ClusterInventory.load(name)
    for line in plan_cluster_commands(inv, model_repo=model_repo, engine=engine, port=port):
        click.echo(line)


@cluster.command("serve")
@click.argument("name")
@click.option("--model-repo", required=True)
@click.option("--engine", type=click.Choice(["mlx", "vllm", "llama.cpp"]), required=True)
@click.option("--port", default=8080, show_default=True)
@click.option("--dry-run", is_flag=True, default=True, show_default=True)
def cluster_serve_cmd(name: str, model_repo: str, engine: str, port: int, dry_run: bool) -> None:
    """Show (default) or refuse unsupervised cluster serve.

    Cluster bring-up stays explicit: print validated upstream commands.
    Automatic remote orchestration is intentionally out of scope for v0.2.
    """
    inv = ClusterInventory.load(name)
    report = doctor_cluster(inv)
    if not report["ok"]:
        raise click.ClickException("cluster doctor failed; fix inventory/ssh first")
    lines = plan_cluster_commands(inv, model_repo=model_repo, engine=engine, port=port)
    for line in lines:
        click.echo(line)
    if dry_run:
        click.echo("# dry-run only — run the printed upstream commands on each node")
        return
    raise click.ClickException(
        "refusing unsupervised multi-node serve; run the printed commands on each host"
    )


if __name__ == "__main__":  # pragma: no cover
    main()
