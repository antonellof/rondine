"""Rondine CLI."""

from __future__ import annotations

import json
import shutil
import subprocess
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
from rondine.hub import (
    apply_hub_hardware_budget,
    curated_repo_ids,
    infer_model_family,
    inspect_repo,
    search_hub,
)
from rondine.paths import brand_logo, plans_dir
from rondine.planner import load_plan, plan_model, save_plan
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
from rondine.ux import (
    echo_heading,
    echo_kv,
    echo_note,
    echo_rule,
    echo_success,
    echo_warning,
    is_interactive_terminal,
    select_menu,
    spinner,
    styled,
)
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
    click.echo(styled(_format_logo(), "heading", bold=True))
    click.echo()


def _print_estimate(est: dict[str, Any]) -> None:
    total = styled(f"{est['total_gb']} GB", "accent", bold=True)
    headroom = styled(f"{est['headroom_gb']} GB", "success")
    click.echo(
        f"  {styled('memory:', 'label', bold=True)} "
        f"{total} "
        f"(weights {est['weight_gb']} + kv {est['kv_gb']} + "
        f"act {est['activation_gb']} + os {est['os_reserve_gb']}) "
        f"| available {est['available_gb']} GB | "
        f"headroom {headroom}"
    )


def _print_engine_args(args: dict[str, Any], indent: str = "  ") -> None:
    if not args:
        return
    click.echo(f"{indent}{styled('engine config:', 'label', bold=True)}")
    for key, value in sorted(args.items()):
        if isinstance(value, bool):
            rendered = styled(
                value,
                "success" if value else "warning",
                bold=True,
            )
        else:
            rendered = styled(value, "accent")
        click.echo(
            f"{indent}  {styled(key + ':', 'label', bold=True)} {rendered}"
        )


def _print_hybrid_unavailable_guidance(hw: Any) -> None:
    """Explain why CPU/GPU hybrid mode is unavailable on this host."""
    click.echo()
    click.echo("why hybrid mode is unavailable:")
    click.echo(
        "  A discrete CUDA host is a computer with a separate NVIDIA GPU "
        "and CUDA support."
    )
    click.echo(
        "  Hybrid mode splits llama.cpp GGUF weights between NVIDIA GPU VRAM "
        "and system RAM."
    )
    if hw.is_apple_silicon:
        click.echo(
            "  This Apple Silicon Mac uses unified memory and does not provide CUDA, "
            "so that split does not apply."
        )
    elif hw.is_spark:
        click.echo(
            "  This DGX Spark uses unified CPU/GPU memory rather than the supported "
            "discrete-VRAM split."
        )
    else:
        click.echo("  This host has no supported discrete CUDA GPU.")
    click.echo()
    click.echo("what to try instead:")
    click.echo("  - Find a fitting model: rondine suggest --profile coding")
    click.echo(
        f"  - Use resident mode for a model that fits in this host's {hw.ram_gb:g} GB RAM."
    )
    click.echo(
        "  - For a llama.cpp GGUF only, --memory-mode mmap --allow-oversize can "
        "page from SSD,"
    )
    click.echo(
        "    but it requires enough free disk and may be unusably slow for a model "
        "far larger than RAM."
    )


def _selected_payload(result: Any) -> dict[str, Any] | None:
    if result.selected is None:
        return None
    return asdict(result.selected)


def _adapter_for(engine: str) -> EngineAdapter:
    if engine not in ADAPTERS:
        raise click.ClickException(f"unsupported engine: {engine}")
    return ADAPTERS[engine]


def _compatible_engines(hw: Any) -> list[str]:
    """Return engines that can run on this hardware, in recommended setup order."""
    compatible = ["llama.cpp"] if hw.platform in {"darwin", "linux"} else []
    if hw.is_apple_silicon:
        compatible.append("mlx")
    if hw.platform == "linux" and hw.cuda_available:
        compatible.append("vllm")
    return compatible


def _engine_status(hw: Any, engine: str) -> Any | None:
    return next((status for status in hw.engines if status.name == engine), None)


def _engine_install_guidance(hw: Any, engine: str | None = None) -> list[str]:
    compatible = _compatible_engines(hw)
    chosen = [engine] if engine else compatible
    lines = [
        f"Install automatically: rondine setup --engine {name}"
        for name in chosen
        if name in compatible
    ]
    if hw.platform == "darwin":
        if "llama.cpp" in chosen:
            lines.append("macOS alternative: brew install llama.cpp")
        if not hw.is_apple_silicon:
            lines.append("MLX is unavailable: it requires Apple Silicon.")
    elif hw.platform == "linux":
        if "llama.cpp" in chosen:
            lines.append(
                "Linux llama.cpp setup requires git, cmake, and a C++ compiler."
            )
        if not hw.cuda_available:
            lines.append("vLLM is unavailable: no NVIDIA CUDA device was detected.")
        elif "vllm" in chosen:
            lines.append(
                "vLLM uses Docker when available; otherwise setup creates a managed uv environment."
            )
    else:
        lines.append("Native Windows engines are unsupported; run Rondine inside WSL2.")
    return lines


def _require_engine_ready(hw: Any, engine: str) -> None:
    status = _engine_status(hw, engine)
    if status is not None and status.available:
        return
    detail = status.detail if status is not None else "not detected"
    guidance = _engine_install_guidance(hw, engine)
    if engine not in _compatible_engines(hw):
        guidance.insert(
            0,
            f"{engine} is not compatible with detected hardware "
            f"({hw.platform}/{hw.arch}).",
        )
    message = f"engine '{engine}' is not ready: {detail}"
    if guidance:
        message += "\n" + "\n".join(guidance)
    raise click.ClickException(message)


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
    """CLI-compatible wrapper around the reusable Hub memory calculator."""
    try:
        return apply_hub_hardware_budget(
            selected,
            catalog,
            hw,
            memory_mode=memory_mode,
            allow_oversize=allow_oversize,
        )
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc


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
    family = infer_model_family(repo, {model.family for model in catalog.models})
    if family:
        settings = profile_settings(catalog, profile, family)
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


@click.group(
    context_settings={"auto_envvar_prefix": "RONDINE"},
    invoke_without_command=True,
)
@click.version_option(__version__, prog_name="rondine")
@click.option(
    "--color/--no-color",
    default=None,
    help="Force colored output on or off (env: RONDINE_COLOR).",
)
@click.pass_context
def main(ctx: click.Context, color: bool | None) -> None:
    """Hardware-aware local LLM launcher for Mac, NVIDIA GPUs, and DGX Spark."""
    ctx.color = color
    if ctx.invoked_subcommand is not None:
        return
    if not is_interactive_terminal():
        click.echo(ctx.get_help())
        return
    _interactive_cli(ctx)


@main.command()
def doctor() -> None:
    """Probe hardware, memory, and installed engines."""
    _echo_logo()
    with spinner("scanning hardware"):
        hw = detect_hardware()
        catalog = load_catalog()
    echo_heading("Hardware")
    echo_kv("host", hw.hostname)
    echo_kv("platform", f"{hw.platform}/{hw.arch}")
    echo_kv("cpu", hw.cpu_brand or "(unknown)")
    echo_kv("ram", f"{hw.ram_gb} GB", value_role="accent")
    echo_kv("disk free", f"{hw.disk_free_gb} GB / {hw.disk_total_gb} GB")
    echo_kv("apple silicon", hw.is_apple_silicon)
    echo_kv("dgx spark", hw.is_spark)
    if hw.cuda_available:
        cap = ".".join(str(x) for x in hw.cuda_capability) if hw.cuda_capability else "?"
        echo_kv(
            "cuda",
            f"yes ({hw.gpu_name or 'gpu'}, compute {cap})",
            value_role="success",
        )
        if hw.vram_gb:
            echo_kv(
                "vram",
                f"{hw.vram_gb} GB" + (f" ×{hw.gpu_count}" if hw.gpu_count > 1 else ""),
                value_role="accent",
            )
        if hw.is_discrete_cuda:
            echo_kv("fit budget", f"{hw.usable_ram_gb} GB VRAM (discrete GPU)")
    else:
        echo_kv("cuda", "no", value_role="muted")
    echo_kv("metal", hw.metal_available)
    echo_heading("Engines")
    for eng in hw.engines:
        status = styled(
            "✓ ready" if eng.available else "○ missing",
            "success" if eng.available else "warning",
            bold=eng.available,
        )
        extra = eng.version or eng.detail or eng.path or ""
        name = styled(f"{eng.name:20}", "label", bold=True)
        click.echo(f"  {name} {status} {extra}".rstrip())
    from rondine.planner import match_target

    target = match_target(catalog, hw)
    echo_kv("matched target", target or "(none)", value_role="accent")
    for w in hw.warnings:
        echo_warning(w)
    compatible = _compatible_engines(hw)
    ready = [
        name
        for name in compatible
        if (engine_status := _engine_status(hw, name)) is not None
        and engine_status.available
    ]
    click.echo()
    if ready:
        click.echo(
            f"{styled('next:', 'label', bold=True)} "
            f"{styled('rondine suggest --profile coding', 'command')}"
        )
    else:
        echo_warning("no runnable inference engine detected")
        echo_heading("Install an engine")
        for line in _engine_install_guidance(hw):
            click.echo(f"  {line}")
        click.echo()
        click.echo(
            f"{styled('recommended:', 'label', bold=True)} "
            f"{styled('rondine setup', 'command')}"
        )


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
    "--hub/--no-hub",
    default=True,
    show_default=True,
    help="Supplement the catalog with fitting models from Hugging Face search.",
)
@click.option(
    "--hub-query",
    default=None,
    metavar="TEXT",
    help="Override the Hugging Face search text (default: coder or instruct).",
)
@click.option(
    "--context",
    type=click.IntRange(min=1024),
    default=None,
    metavar="TOKENS",
    help="Require this context window and include it in memory-fit ranking.",
)
@click.option(
    "--json",
    "as_json",
    is_flag=True,
    help="Print the complete suggestion result as JSON for scripts.",
)
@click.option(
    "-i",
    "--interactive",
    is_flag=True,
    help="Select and configure a suggestion with an interactive menu.",
)
@click.option(
    "--no-interactive",
    is_flag=True,
    help="Do not offer interactive selection after showing suggestions.",
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
    hub: bool,
    hub_query: str | None,
    context: int | None,
    as_json: bool,
    interactive: bool,
    no_interactive: bool,
    configure_rank: int | None,
    save_as: str | None,
) -> None:
    """Suggest models that fit this machine and show launch configs."""
    if interactive and as_json:
        raise click.ClickException("--interactive cannot be combined with --json")
    if interactive and configure_rank is not None:
        raise click.ClickException(
            "--interactive cannot be combined with --configure"
        )
    if interactive and no_interactive:
        raise click.ClickException(
            "--interactive cannot be combined with --no-interactive"
        )
    catalog = load_catalog()
    with spinner("matching models to hardware"):
        hw = detect_hardware()
        result = suggest_for_hardware(
            catalog,
            hw,
            profile=profile,
            limit=limit,
            include_opt_in=opt_in,
            include_hub=hub,
            hub_query=hub_query,
            context_override=context,
        )

    if as_json:
        click.echo(json.dumps(result.to_dict(), indent=2))
        return

    _echo_logo()
    echo_heading("Hardware match")
    echo_kv(
        "target",
        f"{result.target_id or '(generic)'} ({result.target_label or 'unmatched'})",
        value_role="accent",
    )
    ram = result.hardware.get("ram_gb")
    vram = result.hardware.get("vram_gb")
    if result.hardware.get("is_discrete_cuda") and vram:
        echo_kv(
            "memory",
            f"{vram} GB VRAM ({result.hardware.get('gpu_name') or 'CUDA'}), "
            f"{ram} GB system RAM",
            value_role="accent",
        )
    else:
        echo_kv("memory", f"{ram} GB RAM", value_role="accent")
    echo_kv("profile", profile)
    if context is not None:
        echo_kv("required context", f"{context:,} tokens", value_role="accent")
    echo_kv("engines", " → ".join(result.engine_order))
    if result.preferred_engine:
        echo_kv("preferred engine", result.preferred_engine, value_role="success")
    if result.missing_engines:
        echo_kv(
            "missing engines",
            ", ".join(result.missing_engines),
            value_role="warning",
        )
    click.echo()

    if not result.suggestions:
        click.echo("no fitting suggestions")
        for note in result.notes:
            click.echo(f"note: {note}")
        sys.exit(1)

    echo_heading("Recommended configs")
    for s in result.suggestions:
        echo_rule()
        if s.source == "huggingface":
            marker = styled("HUB", "accent", bold=True)
        elif s.curated_hint:
            marker = styled("TOP", "warning", bold=True)
        else:
            marker = styled("CAT", "muted", bold=True)
        click.echo(
            f"{marker} {styled(f'#{s.rank}', 'heading', bold=True)}  "
            f"{styled(s.display_name, 'success', bold=True)}"
        )
        click.echo(f"     {styled('id:', 'muted')} {s.model_id}")
        click.echo(
            f"     {styled('engine', 'label')}={s.engine}  "
            f"{styled('format', 'label')}={s.format}  "
            f"{styled('quant', 'label')}={s.quant}  "
            f"{styled('provider', 'label')}={s.provider}  "
            f"{styled('score', 'label')}={styled(f'{s.score:.1f}', 'accent')}"
        )
        click.echo(f"     {styled('repo:', 'label')} {s.repo}")
        headroom = styled(f"{s.estimate['headroom_gb']}GB", "success")
        click.echo(
            f"     {styled('context:', 'label')} {s.context:,}  "
            f"{styled('memory:', 'label')} ~{s.estimate['total_gb']}GB "
            f"(headroom {headroom})"
        )
        _print_engine_args(s.engine_args, indent="     ")
        if s.sampling:
            samp = ", ".join(
                f"{k}={v}"
                for k, v in s.sampling.items()
                if k != "chat_template_kwargs" and not isinstance(v, dict)
            )
            if samp:
                click.echo(f"     {styled('sampling:', 'label')} {samp}")
        model_arg = s.repo if s.source == "huggingface" else s.model_id
        command = f"rondine serve {model_arg} --profile {profile}"
        click.echo(f"     {styled('run:', 'label', bold=True)} {styled(command, 'command')}")

    echo_rule()
    click.echo()
    for note in result.notes:
        echo_note(note)
    click.echo(
        f"{styled('tip:', 'warning', bold=True)} "
        f"{styled('rondine suggest --configure 1 --save-as coding', 'command')}"
    )
    click.echo(
        f"{styled('legend:', 'muted')} "
        f"{styled('TOP', 'warning', bold=True)} target pick · "
        f"{styled('HUB', 'accent', bold=True)} Hugging Face · "
        f"{styled('CAT', 'muted', bold=True)} catalog"
    )

    if (
        not interactive
        and not no_interactive
        and configure_rank is None
        and is_interactive_terminal()
    ):
        click.echo()
        interactive = click.confirm(
            "Select and configure one of these suggestions now?",
            default=False,
        )

    if interactive:
        labels = [
            f"#{s.rank} {s.display_name} — {s.engine}/{s.quant}, "
            f"~{s.estimate['total_gb']}GB"
            for s in result.suggestions
        ]
        can_show_more = len(result.suggestions) >= limit and limit < 50
        if can_show_more:
            next_limit = min(limit + 5, 50)
            labels.append(
                f"Show more recommendations — up to {next_limit} results"
            )
        selected_index = select_menu(labels)
        if selected_index is None:
            echo_note("selection cancelled; no plan was changed")
            return
        if can_show_more and selected_index == len(result.suggestions):
            echo_heading(f"Expanding recommendations to {next_limit}")
            click.get_current_context().invoke(
                suggest,
                profile=profile,
                limit=next_limit,
                opt_in=opt_in,
                hub=hub,
                hub_query=hub_query,
                context=context,
                as_json=False,
                interactive=True,
                no_interactive=False,
                configure_rank=None,
                save_as=save_as,
            )
            return
        configure_rank = result.suggestions[selected_index].rank

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
    echo_success(f"✓ configured plan: {path}")
    click.echo(
        f"  {styled(match.model_id, 'success', bold=True)} "
        f"via {match.engine}/{match.quant}"
    )
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
    click.echo(
        f"{styled('next:', 'label', bold=True)} "
        f"{styled('rondine setup && rondine pull && rondine serve', 'command')}"
    )


@main.command("models")
@click.option("--profile", default="coding", show_default=True)
@click.option("--opt-in/--no-opt-in", default=False, help="Include large opt-in models.")
def models_cmd(profile: str, opt_in: bool) -> None:
    """List curated catalog models and whether they fit this machine."""
    catalog = load_catalog()
    hw = detect_hardware()
    result = plan_model(catalog, hw, None, profile=profile, include_opt_in=opt_in)
    echo_heading("Curated models")
    echo_kv("profile", profile)
    echo_kv("memory", f"{hw.ram_gb}GB RAM")
    click.echo(
        f"{styled('tip:', 'warning', bold=True)} "
        f"{styled('rondine suggest', 'command')} — ranked configs for this machine"
    )
    seen: set[str] = set()
    for cand in result.candidates:
        if cand.model_id in seen and cand.rejected:
            continue
        mark = styled(
            "FIT" if not cand.rejected else "NO ",
            "success" if not cand.rejected else "error",
            bold=True,
        )
        provider = (cand.variant or {}).get("provider") or ""
        if cand.model_id not in seen:
            seen.add(cand.model_id)
            model = next(m for m in catalog.models if m.id == cand.model_id)
            opt = " [opt-in]" if model.opt_in else ""
            click.echo(
                f"{mark} {styled(cand.model_id + opt, 'heading', bold=True)} "
                f"— {cand.display_name}"
            )
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
    echo_heading(f"Hugging Face search: {query!r}")
    for hit in hits:
        marker = (
            styled("CAT", "warning", bold=True)
            if hit.curated
            else styled("HUB", "accent", bold=True)
        )
        click.echo(
            f"{marker} {styled(f'{hit.repo_id:55}', 'success', bold=hit.curated)} "
            f"{styled(f'{hit.engine_hint:10}', 'label')} "
            f"downloads={hit.downloads:<8} score={styled(f'{hit.score:.1f}', 'accent')}"
        )
    click.echo(
        f"{styled('next:', 'label', bold=True)} "
        f"{styled('rondine inspect <org/name>', 'command')}  |  "
        f"{styled('rondine plan <org/name>', 'command')}"
    )


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
    echo_heading("Hugging Face repository")
    echo_kv("repo", result.repo_id, value_role="success")
    echo_kv("engine / format", f"{result.engine_hint} / {result.format_hint}")
    echo_kv("downloads", f"{result.downloads:,}")
    echo_kv(
        "recommended",
        f"{result.recommended_quant} ({result.recommended_file}) ~{result.weight_gb} GB",
        value_role="accent",
    )
    for note in result.notes:
        echo_note(note)
    echo_heading("Files")
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
        echo_heading("Selected model")
        echo_kv("source", "Hugging Face Hub", value_role="accent")
        echo_kv("model", selected["display_name"], value_role="success")
        click.echo(
            f"  {styled('engine:', 'label')} {selected['engine']}  "
            f"{styled('format:', 'label')} {selected['format']}  "
            f"{styled('quant:', 'label')} {selected['quant']}"
        )
        echo_kv("repo", selected["repo"], indent="  ")
        echo_kv("context", f"{selected['context']:,}", indent="  ")
        _print_estimate(selected["estimate"])
        _print_engine_args(selected.get("engine_args") or {})
        for reason in selected.get("reasons") or []:
            click.echo(f"  {styled('note:', 'warning', bold=True)} {reason}")
        echo_success(f"✓ saved plan: {plans_dir() / 'last.json'}")
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
    echo_heading("Selected model")
    echo_kv("source", "curated catalog", value_role="accent")
    echo_kv("target", result.target_id or "(generic)")
    echo_kv("profile", profile)
    if result.selected is None:
        click.echo("no fitting candidate — try --opt-in, lower --context, or Hub search")
        rejected = [c for c in result.candidates if c.rejected][:8]
        for c in rejected:
            click.echo(f"  rejected {c.model_id} {c.engine}/{c.quant}: {c.reject_reason}")
        if memory_mode == "hybrid" and not hw.is_discrete_cuda:
            _print_hybrid_unavailable_guidance(hw)
        sys.exit(1)
    sel = result.selected
    provider = (sel.variant or {}).get("provider") or ""
    echo_kv("model", f"{sel.display_name} ({sel.model_id})", value_role="success")
    click.echo(
        f"  {styled('engine:', 'label')} {sel.engine}  "
        f"{styled('format:', 'label')} {sel.format}  "
        f"{styled('quant:', 'label')} {sel.quant}  "
        f"{styled('provider:', 'label')} {provider}"
    )
    echo_kv("repo", sel.repo, indent="  ")
    echo_kv("context", f"{sel.context:,}", indent="  ")
    click.echo(
        f"  {styled('memory mode:', 'label')} {sel.memory_mode}"
        + (
            f" {styled('(EXPERIMENTAL)', 'warning', bold=True)}"
            if sel.experimental
            else ""
        )
    )
    _print_estimate(asdict(sel.estimate))
    engine_args = (sel.variant or {}).get("engine_args") or {}
    _print_engine_args(engine_args if isinstance(engine_args, dict) else {})
    for reason in sel.reasons:
        click.echo(f"  {styled('note:', 'warning', bold=True)} {reason}")
    for warning in sel.warnings:
        echo_warning(warning)
    echo_success(f"✓ saved plan: {path}")
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
    compatible = _compatible_engines(hw)
    if not compatible:
        raise click.ClickException("\n".join(_engine_install_guidance(hw)))
    if not engines:
        engines = tuple(compatible)
    for name in engines:
        if name not in compatible:
            detail = _engine_status(hw, name)
            reason = detail.detail if detail is not None else "unsupported on this host"
            raise click.ClickException(
                f"cannot set up {name}: {reason}\n"
                + "\n".join(_engine_install_guidance(hw, name))
            )
        click.echo(f"== setup {name} ==")
        adapter = _adapter_for(name)
        try:
            with spinner(f"setting up {name}", enabled=not dry_run):
                lines = adapter.setup(dry_run=dry_run)
        except (OSError, RuntimeError, subprocess.SubprocessError) as exc:
            raise click.ClickException(
                f"failed to set up {name}: {exc}\n"
                + "\n".join(_engine_install_guidance(hw, name))
            ) from exc
        for line in lines:
            click.echo(line)
    if not dry_run:
        refreshed = detect_hardware()
        missing = [
            name
            for name in engines
            if not (
                (status := _engine_status(refreshed, name)) is not None
                and status.available
            )
        ]
        if missing:
            raise click.ClickException(
                "setup completed, but these engines are still unavailable: "
                + ", ".join(missing)
                + "\nRun `rondine doctor` for details."
            )
        echo_success("✓ engine setup complete")


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
    if not dry_run:
        _require_engine_ready(detect_hardware(), str(selected["engine"]))
    if selected.get("experimental"):
        for warning in selected.get("warnings") or [
            "experimental oversized launch"
        ]:
            echo_warning(str(warning))
    spec = adapter.build_serve({"selected": selected}, host=host, port=port)
    echo_heading("Launch configuration")
    echo_kv("engine", spec.engine, value_role="accent")
    echo_kv("command", spec.command_line(), value_role="command")
    for note in spec.notes:
        echo_note(note)
    echo_kv("openai base url", spec.base_url, value_role="success")
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
        echo_success(f"✓ started pid={record.pid}")
        echo_kv("log", record.log_path)
        click.echo(
            f"{styled('stop with:', 'label', bold=True)} "
            f"{styled(f'rondine stop --name {name}', 'command')}"
        )


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


def _wizard_suggest(ctx: click.Context) -> str:
    echo_rule()
    echo_heading("Find a model")
    profile_index = select_menu(
        [
            "Coding — larger context and coding-focused ranking",
            "Chat — lower context and conversational sampling",
        ],
        title="Choose a workload",
    )
    if profile_index is None:
        echo_note("recommendation step skipped")
        return "coding"
    profile = "coding" if profile_index == 0 else "chat"
    default_context = 32768 if profile == "coding" else 16384
    context_values: list[int | None] = [None, 4096, 16384, 32768, 65536, 131072]
    context_index = select_menu(
        [
            f"Profile default — {default_context:,} tokens",
            "4,096 tokens — lowest memory",
            "16,384 tokens",
            "32,768 tokens",
            "65,536 tokens",
            "131,072 tokens",
        ],
        title="Choose required context",
    )
    if context_index is None:
        echo_note("recommendation step skipped")
        return profile
    discovery_index = select_menu(
        [
            "Catalog + Hugging Face — discover current models",
            "Curated catalog only — faster and offline",
        ],
        title="Choose model sources",
    )
    if discovery_index is None:
        echo_note("recommendation step skipped")
        return profile

    ctx.invoke(
        suggest,
        profile=profile,
        limit=5,
        opt_in=False,
        hub=discovery_index == 0,
        hub_query=None,
        context=context_values[context_index],
        as_json=False,
        interactive=True,
        no_interactive=False,
        configure_rank=None,
        save_as=None,
    )
    return profile


def _selected_label(selected: dict[str, Any]) -> str:
    return str(
        selected.get("display_name")
        or selected.get("model_id")
        or selected.get("repo")
        or "unknown model"
    )


def _show_current_config() -> None:
    plan_data = load_plan()
    selected = (plan_data or {}).get("selected")
    if not isinstance(selected, dict):
        echo_warning("no active configuration; choose a model first")
        return
    echo_heading("Current configuration")
    echo_kv("model", _selected_label(selected), value_role="success")
    echo_kv("profile", (plan_data or {}).get("profile") or "coding")
    echo_kv(
        "runtime",
        f"{selected.get('engine', '?')} / {selected.get('quant', '?')}",
        value_role="accent",
    )
    echo_kv("context", f"{int(selected.get('context') or 0):,} tokens")
    echo_kv("repo", selected.get("repo") or "(unknown)")
    preset_name = (plan_data or {}).get("preset_name")
    if preset_name:
        echo_kv("preset", preset_name, value_role="warning")


def _activate_preset(name: str) -> tuple[str, str]:
    preset = load_preset(name)
    selected = selected_with_preset_overrides(preset)
    path = plans_dir() / "last.json"
    path.write_text(
        json.dumps(
            {
                "hardware": {},
                "profile": preset.profile,
                "target_id": preset.target_id,
                "selected": selected,
                "candidates": [selected],
                "source": "preset",
                "preset_name": preset.name,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    echo_success(f"✓ loaded preset {preset.name}")
    return preset.profile, preset.name


def _choose_preset() -> tuple[str, str] | None:
    presets = list_presets()
    if not presets:
        echo_note("no saved presets")
        return None
    labels = [
        f"{item.name} — {_selected_label(item.selected)} "
        f"({item.selected.get('engine', '?')}/{item.selected.get('quant', '?')})"
        for item in presets
    ]
    index = select_menu(labels, title="Load a saved preset")
    if index is None:
        return None
    return _activate_preset(presets[index].name)


def _model_actions(
    ctx: click.Context,
    *,
    profile: str,
    preset_name: str | None,
) -> None:
    if not isinstance((load_plan() or {}).get("selected"), dict):
        echo_warning("no active configuration; choose a model first")
        return
    actions = [
        "Preview serve command",
        "Download model weights",
        "Start model server",
        "Verify running server",
        "Stop running server",
        "Back to main menu",
    ]
    while True:
        action = select_menu(actions, title="Run selected model")
        if action is None or action == 5:
            return
        if action == 0:
            ctx.invoke(
                serve,
                model=None,
                profile=profile,
                quant=None,
                host=None,
                port=None,
                dry_run=True,
                foreground=False,
                name="default",
                preset_name=preset_name,
                save_as=None,
                memory_mode="auto",
                allow_oversize=False,
            )
        elif action == 1:
            if click.confirm("Download the selected model weights?", default=False):
                ctx.invoke(
                    pull,
                    model=None,
                    profile=profile,
                    quant=None,
                    memory_mode="auto",
                    allow_oversize=False,
                    dry_run=False,
                )
        elif action == 2:
            if click.confirm("Start the selected model server?", default=True):
                ctx.invoke(
                    serve,
                    model=None,
                    profile=profile,
                    quant=None,
                    host=None,
                    port=None,
                    dry_run=False,
                    foreground=False,
                    name="default",
                    preset_name=preset_name,
                    save_as=None,
                    memory_mode="auto",
                    allow_oversize=False,
                )
        elif action == 3:
            ctx.invoke(
                verify,
                profile=profile,
                base_url=None,
                name="default",
                timeout=120,
            )
        elif action == 4:
            name = click.prompt("Server name", default="default")
            if click.confirm(f"Stop server {name!r}?", default=True):
                ctx.invoke(stop, name=name)


def _change_model_actions(
    ctx: click.Context,
    *,
    profile: str,
    preset_name: str | None,
) -> tuple[str, str | None]:
    actions = [
        "Get hardware-aware recommendations",
        "Browse curated catalog",
        "Search Hugging Face",
        "Plan by catalog ID or Hub repository",
        "Back to main menu",
    ]
    while True:
        action = select_menu(actions, title="Choose or change model")
        if action is None or action == 4:
            return profile, preset_name
        if action == 0:
            return _wizard_suggest(ctx), None
        if action == 1:
            ctx.invoke(models_cmd, profile=profile, opt_in=False)
        elif action == 2:
            query = click.prompt("Search Hugging Face", type=str)
            engine_index = select_menu(
                ["Any compatible engine", "llama.cpp", "MLX", "vLLM"],
                title="Filter by engine",
            )
            if engine_index is not None:
                engines = [None, "llama.cpp", "mlx", "vllm"]
                ctx.invoke(
                    search,
                    query=query,
                    engine=engines[engine_index],
                    limit=15,
                    as_json=False,
                )
        elif action == 3:
            model = click.prompt("Catalog ID or Hugging Face org/name", type=str)
            context = click.prompt(
                "Context tokens",
                type=click.IntRange(min=1024),
                default=32768 if profile == "coding" else 16384,
            )
            ctx.invoke(
                plan,
                model=model,
                profile=profile,
                context=context,
                quant=None,
                opt_in=True,
                memory_mode="auto",
                allow_oversize=False,
                save_as=None,
                as_json=False,
            )
            return profile, None


def _environment_actions(ctx: click.Context) -> None:
    actions = [
        "Set up or update inference engines",
        "Run hardware doctor",
        "Back to main menu",
    ]
    while True:
        action = select_menu(actions, title="Environment")
        if action is None or action == 2:
            return
        if action == 0 and click.confirm(
            "Install or update recommended engines?", default=True
        ):
            ctx.invoke(setup, engines=(), dry_run=False)
        elif action == 1:
            ctx.invoke(doctor)


def _preset_actions(
    ctx: click.Context,
    *,
    profile: str,
) -> tuple[str, str | None]:
    actions = [
        "Load a saved preset",
        "Save current configuration as a preset",
        "List saved presets",
        "Back to main menu",
    ]
    while True:
        action = select_menu(actions, title="Presets")
        if action is None or action == 3:
            return profile, (load_plan() or {}).get("preset_name")
        if action == 0:
            loaded = _choose_preset()
            if loaded:
                return loaded
        elif action == 1:
            name = click.prompt("Preset name", type=str)
            ctx.invoke(
                preset_save,
                name=name,
                profile=profile,
                host=None,
                port=None,
                run_name="default",
            )
        elif action == 2:
            ctx.invoke(preset_list)


def _startup_state(ctx: click.Context) -> tuple[str, str | None] | None:
    plan_data = load_plan()
    selected = (plan_data or {}).get("selected")
    active_selected = selected if isinstance(selected, dict) else None
    has_active = active_selected is not None
    presets = list_presets()
    if not has_active and not presets:
        ctx.invoke(doctor)
        return _wizard_suggest(ctx), None

    echo_heading("Welcome back")
    if active_selected is not None:
        echo_kv("active", _selected_label(active_selected), value_role="success")
        echo_kv("profile", (plan_data or {}).get("profile") or "coding")
    if presets:
        echo_kv("saved presets", len(presets), value_role="accent")

    options: list[str] = []
    actions: list[str] = []
    if active_selected is not None:
        options.append(f"Continue with {_selected_label(active_selected)}")
        actions.append("continue")
    if presets:
        options.append("Load a saved preset")
        actions.append("preset")
    options.extend(
        [
            "Open main menu",
            "Start a new guided setup",
            "Exit Rondine",
        ]
    )
    actions.extend(["menu", "new", "exit"])
    index = select_menu(options, title="Resume your work")
    if index is None or actions[index] == "exit":
        return None
    if actions[index] == "preset":
        return _choose_preset()
    if actions[index] == "new":
        ctx.invoke(doctor)
        return _wizard_suggest(ctx), None
    return (
        str((plan_data or {}).get("profile") or "coding"),
        (plan_data or {}).get("preset_name"),
    )


def _interactive_cli(ctx: click.Context) -> None:
    """Run the guided no-argument Rondine experience."""
    echo_heading("Rondine interactive")
    click.echo(styled("Local models, without the guesswork", "muted"))
    click.echo()
    startup = _startup_state(ctx)
    if startup is None:
        echo_success("Goodbye from Rondine")
        return
    profile, preset_name = startup

    actions = [
        "Run selected model — preview, download, start, verify",
        "Choose or change model — recommendations, catalog, Hub",
        "Environment — engines and hardware",
        "Presets — load, save, list",
        "Show current configuration",
        "Exit Rondine",
    ]
    while True:
        action = select_menu(actions, title="Main menu")
        if action is None or action == 5:
            echo_success("Goodbye from Rondine")
            return
        try:
            if action == 0:
                _model_actions(
                    ctx,
                    profile=profile,
                    preset_name=preset_name,
                )
            elif action == 1:
                profile, preset_name = _change_model_actions(
                    ctx,
                    profile=profile,
                    preset_name=preset_name,
                )
            elif action == 2:
                _environment_actions(ctx)
            elif action == 3:
                profile, preset_name = _preset_actions(
                    ctx,
                    profile=profile,
                )
            elif action == 4:
                _show_current_config()
        except click.ClickException as exc:
            echo_warning(exc.format_message())
        except (OSError, ValueError) as exc:
            echo_warning(str(exc))
        except SystemExit as exc:
            if exc.code:
                echo_warning(f"command exited with status {exc.code}")
        except click.Abort:
            echo_note("action cancelled")


if __name__ == "__main__":  # pragma: no cover
    main()
