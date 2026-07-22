"""Probe local hardware and installed engines."""

from __future__ import annotations

import os
import platform
import re
import shutil
import subprocess
from dataclasses import dataclass, field

from rondine.paths import which


@dataclass
class EngineStatus:
    name: str
    available: bool
    path: str | None = None
    version: str | None = None
    detail: str = ""


@dataclass
class HardwareInfo:
    platform: str  # darwin / linux / windows
    arch: str
    hostname: str
    ram_gb: float
    cpu_brand: str = ""
    is_apple_silicon: bool = False
    is_spark: bool = False
    cuda_available: bool = False
    cuda_capability: tuple[int, int] | None = None
    gpu_name: str = ""
    metal_available: bool = False
    engines: list[EngineStatus] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def usable_ram_gb(self) -> float:
        return self.ram_gb


def _run(cmd: list[str], timeout: float = 5.0) -> str:
    try:
        proc = subprocess.run(
            cmd,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except (OSError, subprocess.TimeoutExpired):
        return ""
    return (proc.stdout or "") + (proc.stderr or "")


def _ram_gb() -> float:
    system = platform.system().lower()
    if system == "darwin":
        out = _run(["sysctl", "-n", "hw.memsize"])
        try:
            return int(out.strip()) / (1024**3)
        except ValueError:
            return 0.0
    if system == "linux":
        try:
            mem_kb = 0
            with open("/proc/meminfo", encoding="utf-8") as fh:
                for line in fh:
                    if line.startswith("MemTotal:"):
                        mem_kb = int(line.split()[1])
                        break
            return mem_kb / (1024**2)
        except OSError:
            return 0.0
    # Fallback
    try:
        import psutil  # type: ignore

        return float(psutil.virtual_memory().total) / (1024**3)
    except Exception:
        return 0.0


def _cpu_brand() -> str:
    system = platform.system().lower()
    if system == "darwin":
        return _run(["sysctl", "-n", "machdep.cpu.brand_string"]).strip()
    if system == "linux":
        try:
            with open("/proc/cpuinfo", encoding="utf-8") as fh:
                for line in fh:
                    if line.startswith("model name") or line.startswith("Hardware"):
                        return line.split(":", 1)[1].strip()
        except OSError:
            pass
    return platform.processor() or ""


def _detect_cuda() -> tuple[bool, tuple[int, int] | None, str]:
    if which("nvidia-smi") is None:
        return False, None, ""
    out = _run(
        [
            "nvidia-smi",
            "--query-gpu=name,compute_cap",
            "--format=csv,noheader",
        ]
    )
    if not out.strip():
        return True, None, ""
    first = out.strip().splitlines()[0]
    parts = [p.strip() for p in first.split(",")]
    name = parts[0] if parts else ""
    cap: tuple[int, int] | None = None
    if len(parts) > 1 and "." in parts[1]:
        try:
            major, minor = parts[1].split(".", 1)
            cap = (int(major), int(minor))
        except ValueError:
            cap = None
    return True, cap, name


def _looks_like_spark(gpu_name: str, arch: str, ram_gb: float) -> bool:
    name = gpu_name.lower()
    if "spark" in name or "gb10" in name:
        return True
    # Heuristic: aarch64 Linux + Blackwell (sm12x) + ~128GB unified
    if arch in {"aarch64", "arm64"} and platform.system().lower() == "linux":
        if 100 <= ram_gb <= 140 and ("blackwell" in name or "nvidia" in name):
            return True
    return bool(os.environ.get("RONDINE_FORCE_SPARK"))


def _engine_llama() -> EngineStatus:
    path = which("llama-server") or which("llama-cli")
    if not path:
        brew = which("brew")
        detail = "not found (brew install llama.cpp or rondine setup)"
        if brew:
            detail += f"; brew={brew}"
        return EngineStatus("llama.cpp", False, detail=detail)
    ver_out = _run([path, "--version"]) or _run([path, "-h"])
    version = None
    match = re.search(r"version[:\s]+([0-9a-zA-Z.\-]+)", ver_out, re.I)
    if match:
        version = match.group(1)
    return EngineStatus("llama.cpp", True, path=path, version=version)


def _engine_mlx() -> EngineStatus:
    if platform.system().lower() != "darwin" or platform.machine() != "arm64":
        return EngineStatus("mlx", False, detail="Apple Silicon only")
    path = which("mlx_lm.server")
    if path:
        return EngineStatus("mlx", True, path=path, detail="mlx_lm.server on PATH")
    # Check for python module via uv/python
    py = which("python3") or which("python")
    if py:
        out = _run([py, "-c", "import mlx_lm; print(getattr(mlx_lm, '__version__', 'ok'))"])
        if out.strip() and "Error" not in out and "Traceback" not in out:
            return EngineStatus("mlx", True, path=py, version=out.strip(), detail="python mlx_lm")
    return EngineStatus("mlx", False, detail="not installed (rondine setup)")


def _engine_vllm() -> EngineStatus:
    if which("docker"):
        # Presence of docker does not mean image is pulled; mark as installable.
        detail = "docker available"
    else:
        detail = "docker not found"
    path = which("vllm")
    if path:
        ver = _run([path, "--version"])
        version = ver.strip().splitlines()[0] if ver.strip() else None
        return EngineStatus("vllm", True, path=path, version=version, detail=detail)
    py = which("python3") or which("python")
    if py:
        out = _run([py, "-c", "import vllm; print(vllm.__version__)"])
        if out.strip() and "Error" not in out and "Traceback" not in out:
            return EngineStatus("vllm", True, path=py, version=out.strip(), detail=detail)
    return EngineStatus("vllm", False, detail=detail)


def detect_hardware() -> HardwareInfo:
    system = platform.system().lower()
    if system == "darwin":
        plat = "darwin"
    elif system == "linux":
        plat = "linux"
    else:
        plat = system

    arch = platform.machine().lower()
    if arch == "aarch64":
        arch_norm = "aarch64"
    elif arch in {"arm64", "arm"}:
        arch_norm = "arm64" if plat == "darwin" else "aarch64"
    elif arch in {"x86_64", "amd64"}:
        arch_norm = "x86_64"
    else:
        arch_norm = arch

    ram = _ram_gb()
    cuda_ok, cap, gpu = _detect_cuda()
    apple = plat == "darwin" and arch_norm == "arm64"
    spark = _looks_like_spark(gpu, arch_norm, ram)
    if cap and cap[0] == 12 and plat == "linux" and arch_norm == "aarch64":
        spark = True

    info = HardwareInfo(
        platform=plat,
        arch=arch_norm,
        hostname=platform.node(),
        ram_gb=round(ram, 1),
        cpu_brand=_cpu_brand(),
        is_apple_silicon=apple,
        is_spark=spark,
        cuda_available=cuda_ok,
        cuda_capability=cap,
        gpu_name=gpu,
        metal_available=apple,
        engines=[_engine_llama(), _engine_mlx(), _engine_vllm()],
    )
    if apple and not shutil.which("llama-server"):
        info.warnings.append(
            "llama-server not on PATH; run `rondine setup` or `brew install llama.cpp`"
        )
    if spark and not cuda_ok:
        info.warnings.append("DGX Spark detected heuristically but nvidia-smi missing")
    return info
