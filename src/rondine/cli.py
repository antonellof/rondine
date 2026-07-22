"""Rondine CLI."""

from __future__ import annotations

import json
import sys
from dataclasses import asdict
from typing import Any

import click

from rondine import __version__
from rondine.catalog import load_catalog
from rondine.cluster import ClusterInventory, ClusterNode, doctor_cluster, plan_cluster_commands
from rondine.detect import detect_hardware
from rondine.engines.base import EngineAdapter
from rondine.engines.llama_cpp import LlamaCppAdapter
from rondine.engines.mlx import MlxAdapter
from rondine.engines.vllm import VllmAdapter
from rondine.paths import brand_logo
from rondine.planner import load_plan, plan_model, save_plan
from rondine.process import is_running, load_record, start_server, stop_server
from rondine.verify import verify_server

ADAPTERS: dict[str, EngineAdapter] = {
    "llama.cpp": LlamaCppAdapter(),
    "mlx": MlxAdapter(),
    "vllm": VllmAdapter(),
}


def _echo_logo() -> None:
    click.echo(brand_logo())
    click.echo()


def _print_estimate(est: dict[str, Any]) -> None:
    click.echo(
        f"  memory est: {est['total_gb']} GB "
        f"(weights {est['weight_gb']} + kv {est['kv_gb']} + "
        f"act {est['activation_gb']} + os {est['os_reserve_gb']}) "
        f"| available {est['available_gb']} GB | headroom {est['headroom_gb']} GB"
    )


def _selected_payload(result: Any) -> dict[str, Any] | None:
    if result.selected is None:
        return None
    return asdict(result.selected)


def _adapter_for(engine: str) -> EngineAdapter:
    if engine not in ADAPTERS:
        raise click.ClickException(f"unsupported engine: {engine}")
    return ADAPTERS[engine]


@click.group()
@click.version_option(__version__, prog_name="rondine")
def main() -> None:
    """Hardware-aware local LLM launcher for Mac and DGX Spark."""


@main.command()
def doctor() -> None:
    """Probe hardware, memory, and installed engines."""
    _echo_logo()
    hw = detect_hardware()
    catalog = load_catalog()
    click.echo(f"host: {hw.hostname}")
    click.echo(f"platform: {hw.platform}/{hw.arch}")
    click.echo(f"cpu: {hw.cpu_brand or '(unknown)'}")
    click.echo(f"ram: {hw.ram_gb} GB")
    click.echo(f"apple silicon: {hw.is_apple_silicon}")
    click.echo(f"dgx spark: {hw.is_spark}")
    if hw.cuda_available:
        cap = ".".join(str(x) for x in hw.cuda_capability) if hw.cuda_capability else "?"
        click.echo(f"cuda: yes ({hw.gpu_name or 'gpu'}, compute {cap})")
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


@main.command("models")
@click.option("--profile", default="coding", show_default=True)
@click.option("--opt-in/--no-opt-in", default=False, help="Include large opt-in models.")
def models_cmd(profile: str, opt_in: bool) -> None:
    """List catalog models and whether they fit this machine."""
    catalog = load_catalog()
    hw = detect_hardware()
    result = plan_model(catalog, hw, None, profile=profile, include_opt_in=opt_in)
    click.echo(f"profile={profile} ram={hw.ram_gb}GB")
    seen: set[str] = set()
    for cand in result.candidates:
        if cand.model_id in seen and cand.rejected:
            continue
        mark = "FIT" if not cand.rejected else "NO"
        if cand.model_id not in seen:
            seen.add(cand.model_id)
            model = next(m for m in catalog.models if m.id == cand.model_id)
            opt = " [opt-in]" if model.opt_in else ""
            click.echo(f"{mark:3} {cand.model_id}{opt} — {cand.display_name}")
        detail = cand.reject_reason or f"{cand.engine}/{cand.quant} score={cand.score:.1f}"
        click.echo(f"     {cand.engine:10} {cand.quant:16} {detail}")


@main.command()
@click.argument("model", default="auto")
@click.option("--profile", default="coding", show_default=True)
@click.option("--context", type=int, default=None, help="Override context length.")
@click.option("--opt-in/--no-opt-in", default=False)
@click.option("--json", "as_json", is_flag=True)
def plan(model: str, profile: str, context: int | None, opt_in: bool, as_json: bool) -> None:
    """Recommend engine / format / quant for a model (or auto)."""
    catalog = load_catalog()
    hw = detect_hardware()
    result = plan_model(
        catalog,
        hw,
        None if model == "auto" else model,
        profile=profile,
        include_opt_in=opt_in or model != "auto",
        context_override=context,
    )
    path = save_plan(result)
    if as_json:
        click.echo(
            json.dumps(
                {
                    "selected": _selected_payload(result),
                    "target_id": result.target_id,
                    "saved": str(path),
                },
                indent=2,
            )
        )
        return
    _echo_logo()
    click.echo(f"target: {result.target_id or '(generic)'}")
    click.echo(f"profile: {profile}")
    if result.selected is None:
        click.echo("no fitting candidate — try --opt-in, lower --context, or a smaller model")
        rejected = [c for c in result.candidates if c.rejected][:8]
        for c in rejected:
            click.echo(f"  rejected {c.model_id} {c.engine}/{c.quant}: {c.reject_reason}")
        sys.exit(1)
    sel = result.selected
    click.echo(f"selected: {sel.display_name} ({sel.model_id})")
    click.echo(f"  engine: {sel.engine}  format: {sel.format}  quant: {sel.quant}")
    click.echo(f"  repo: {sel.repo}")
    click.echo(f"  context: {sel.context}")
    _print_estimate(asdict(sel.estimate))
    for reason in sel.reasons:
        click.echo(f"  note: {reason}")
    click.echo(f"saved plan: {path}")


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
        for line in adapter.setup(dry_run=dry_run):
            click.echo(line)


@main.command()
@click.argument("model", required=False)
@click.option("--profile", default="coding", show_default=True)
@click.option("--dry-run", is_flag=True)
def pull(model: str | None, profile: str, dry_run: bool) -> None:
    """Download weights for a planned model."""
    catalog = load_catalog()
    hw = detect_hardware()
    plan_data = load_plan()
    if model:
        result = plan_model(catalog, hw, model, profile=profile, include_opt_in=True)
        save_plan(result)
        selected = _selected_payload(result)
    else:
        selected = (plan_data or {}).get("selected")
    if not selected:
        raise click.ClickException("no plan selected; run `rondine plan <model>` first")
    adapter = _adapter_for(selected["engine"])
    click.echo(f"pulling {selected['repo']} via {selected['engine']} ...")
    for line in adapter.pull({"selected": selected}, dry_run=dry_run):
        click.echo(line)


@main.command()
@click.argument("model", required=False)
@click.option("--profile", default="coding", show_default=True)
@click.option("--host", default=None)
@click.option("--port", default=None, type=int)
@click.option("--dry-run", is_flag=True)
@click.option("--foreground", is_flag=True, help="Attach to server process.")
@click.option("--name", default="default", show_default=True, help="Run record name.")
def serve(
    model: str | None,
    profile: str,
    host: str | None,
    port: int | None,
    dry_run: bool,
    foreground: bool,
    name: str,
) -> None:
    """Launch an OpenAI-compatible local server."""
    catalog = load_catalog()
    hw = detect_hardware()
    selected: dict[str, Any]
    if model:
        result = plan_model(catalog, hw, model, profile=profile, include_opt_in=True)
        if result.selected is None:
            raise click.ClickException("model does not fit this machine")
        save_plan(result)
        selected = asdict(result.selected)
    else:
        plan_data = load_plan()
        existing = (plan_data or {}).get("selected")
        if isinstance(existing, dict):
            selected = existing
        else:
            result = plan_model(catalog, hw, None, profile=profile)
            if result.selected is None:
                raise click.ClickException("no fitting model; pass an explicit model id")
            save_plan(result)
            selected = asdict(result.selected)

    host = host or catalog.policy.default_host
    port = port or catalog.policy.default_port
    adapter = _adapter_for(str(selected["engine"]))
    spec = adapter.build_serve({"selected": selected}, host=host, port=port)
    click.echo(f"engine: {spec.engine}")
    click.echo(f"command: {spec.command_line()}")
    for note in spec.notes:
        click.echo(f"note: {note}")
    click.echo(f"openai base url: {spec.base_url}")
    if dry_run:
        return
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
    result = verify_server(base_url, model=model, profile=profile, timeout=timeout)
    for check in result.checks:
        mark = "PASS" if check["ok"] else "FAIL"
        click.echo(f"{mark} {check['name']}: {check['detail']}")
    if not result.ok:
        sys.exit(1)


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
