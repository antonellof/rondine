"""Homogeneous dual-node cluster inventory and launch helpers."""

from __future__ import annotations

import json
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from rondine.paths import clusters_dir, which


@dataclass
class ClusterNode:
    host: str
    user: str | None = None
    ram_gb: float | None = None
    role: str = "worker"  # head | worker

    @property
    def ssh_target(self) -> str:
        if self.user:
            return f"{self.user}@{self.host}"
        return self.host


@dataclass
class ClusterInventory:
    name: str
    kind: str  # mac | spark
    nodes: list[ClusterNode] = field(default_factory=list)
    interface: str | None = None
    notes: list[str] = field(default_factory=list)

    def path(self) -> Path:
        return clusters_dir() / f"{self.name}.json"

    def save(self) -> Path:
        path = self.path()
        path.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")
        return path

    @classmethod
    def load(cls, name: str) -> ClusterInventory:
        path = clusters_dir() / f"{name}.json"
        raw = json.loads(path.read_text(encoding="utf-8"))
        nodes = [ClusterNode(**n) for n in raw.get("nodes", [])]
        return cls(
            name=raw["name"],
            kind=raw["kind"],
            nodes=nodes,
            interface=raw.get("interface"),
            notes=list(raw.get("notes", [])),
        )


def ssh_ok(node: ClusterNode) -> tuple[bool, str]:
    ssh = which("ssh")
    if not ssh:
        return False, "ssh not found"
    cmd = [
        ssh,
        "-o",
        "BatchMode=yes",
        "-o",
        "ConnectTimeout=5",
        "-o",
        "StrictHostKeyChecking=accept-new",
        node.ssh_target,
        "echo",
        "ok",
    ]
    try:
        proc = subprocess.run(cmd, check=False, capture_output=True, text=True, timeout=15)
    except Exception as exc:  # noqa: BLE001
        return False, str(exc)
    if proc.returncode != 0:
        return False, (proc.stderr or proc.stdout or "ssh failed").strip()
    return True, "passwordless ssh ok"


def doctor_cluster(inv: ClusterInventory) -> dict[str, Any]:
    report: dict[str, Any] = {
        "name": inv.name,
        "kind": inv.kind,
        "nodes": len(inv.nodes),
        "ok": True,
        "checks": [],
        "warnings": [],
    }
    if len(inv.nodes) < 2:
        report["ok"] = False
        report["checks"].append({"name": "node_count", "ok": False, "detail": "need >= 2 nodes"})
        return report
    if len(inv.nodes) > 2:
        report["warnings"].append("only dual-node clusters are validated in 0.2")

    heads = [n for n in inv.nodes if n.role == "head"]
    if len(heads) != 1:
        report["ok"] = False
        report["checks"].append(
            {"name": "head", "ok": False, "detail": "exactly one node must have role=head"}
        )

    for node in inv.nodes:
        ok, detail = ssh_ok(node)
        report["checks"].append({"name": f"ssh:{node.host}", "ok": ok, "detail": detail})
        if not ok:
            report["ok"] = False

    if inv.kind == "spark" and not inv.interface:
        report["warnings"].append(
            "set inventory.interface to the RoCE/QSFP ifname; NCCL must not fall back to Ethernet"
        )
    if inv.kind == "mac":
        report["warnings"].append(
            "MLX JACCL preferred; llama.cpp RPC is experimental and trusted-LAN only"
        )
    return report


def plan_cluster_commands(
    inv: ClusterInventory,
    *,
    model_repo: str,
    engine: str,
    port: int = 8080,
) -> list[str]:
    """Generate documented upstream launch commands (do not invent a scheduler)."""
    if len(inv.nodes) < 2:
        return ["# inventory needs two nodes"]
    head = next((n for n in inv.nodes if n.role == "head"), inv.nodes[0])
    workers = [n for n in inv.nodes if n is not head]
    lines: list[str] = []

    if inv.kind == "mac" and engine == "mlx":
        hostfile = clusters_dir() / f"{inv.name}-jaccl.json"
        hosts = [{"ssh": n.ssh_target} for n in inv.nodes]
        hostfile.write_text(json.dumps(hosts, indent=2), encoding="utf-8")
        lines.append(f"# wrote {hostfile}")
        lines.append(
            "mlx.launch --backend jaccl --hostfile "
            f"{hostfile} $(which mlx_lm.server) --model {model_repo} --port {port}"
        )
        lines.append("# Ensure the same MLX env + model are present on every Mac.")
        return lines

    if inv.kind == "mac" and engine == "llama.cpp":
        lines.append("# EXPERIMENTAL: llama.cpp RPC — trusted private LAN only")
        for w in workers:
            lines.append(f"# on {w.host}: ggml-rpc-server -c -p 50052")
        rpc = ",".join(f"{w.host}:50052" for w in workers)
        lines.append(
            f"llama-server -hf {model_repo} -ngl 99 --rpc {rpc} --host 0.0.0.0 --port {port}"
        )
        lines.append("# Never expose ggml-rpc-server on an untrusted network.")
        return lines

    if inv.kind == "spark":
        iface = inv.interface or "<ROCE_IFNAME>"
        lines.append(
            "# Validate RoCE bandwidth before serving "
            "(NVIDIA connect-two-sparks playbook)."
        )
        lines.append(f"export MN_IF_NAME={iface}")
        lines.append("export UCX_NET_DEVICES=$MN_IF_NAME")
        lines.append("export NCCL_SOCKET_IFNAME=$MN_IF_NAME")
        lines.append("export GLOO_SOCKET_IFNAME=$MN_IF_NAME")
        lines.append("export NCCL_IB_DISABLE=0")
        lines.append("# export NCCL_IB_HCA=<roce0>,<roce1>  # both halves of ConnectX-7")
        lines.append(f"# head {head.host}:")
        lines.append(
            "./multi-node-serving.sh leader --ray_port=6379 --ray_cluster_size=2 && "
            f"vllm serve {model_repo} --port {port} --tensor-parallel-size 2 "
            "--distributed-executor-backend ray"
        )
        for w in workers:
            lines.append(f"# worker {w.host}:")
            lines.append(
                f"./multi-node-serving.sh worker --ray_address={head.host} --ray_port=6379"
            )
        lines.append("# Do not silently fall back to ordinary Ethernet.")
        return lines

    return [f"# unsupported kind/engine combination: {inv.kind}/{engine}"]
