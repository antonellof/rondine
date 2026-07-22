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
    vram_gb: float = 0.0
    gpu_count: int = 0
    metal_available: bool = False
    disk_free_gb: float = 0.0
    disk_total_gb: float = 0.0
    engines: list[EngineStatus] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def is_discrete_cuda(self) -> bool:
        """True for consumer/workstation NVIDIA GPUs (not Apple Silicon / Spark unified)."""
        return bool(
            self.cuda_available
            and self.vram_gb > 0
            and not self.is_spark
            and not self.is_apple_silicon
        )

    @property
    def usable_ram_gb(self) -> float:
        """Memory budget for weight+KV fit estimates."""
        if self.is_discrete_cuda:
            return self.vram_gb
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
    # Windows / fallback
    try:
        import psutil  # type: ignore

        return float(psutil.virtual_memory().total) / (1024**3)
    except Exception:
        return 0.0


def _disk_gb() -> tuple[float, float]:
    """Return free and total space for the Rondine model volume."""
    try:
        usage = shutil.disk_usage(os.environ.get("RONDINE_HOME") or os.path.expanduser("~"))
    except OSError:
        return 0.0, 0.0
    scale = float(1024**3)
    return usage.free / scale, usage.total / scale


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


@dataclass
class CudaProbe:
    available: bool = False
    capability: tuple[int, int] | None = None
    gpu_name: str = ""
    vram_gb: float = 0.0
    gpu_count: int = 0


def _detect_cuda() -> CudaProbe:
    if which("nvidia-smi") is None:
        return CudaProbe()
    out = _run(
        [
            "nvidia-smi",
            "--query-gpu=name,compute_cap,memory.total",
            "--format=csv,noheader,nounits",
        ]
    )
    if not out.strip():
        return CudaProbe(available=True)

    lines = [ln.strip() for ln in out.strip().splitlines() if ln.strip()]
    names: list[str] = []
    vrams: list[float] = []
    cap: tuple[int, int] | None = None
    for line in lines:
        parts = [p.strip() for p in line.split(",")]
        if not parts:
            continue
        names.append(parts[0])
        if len(parts) > 1 and "." in parts[1]:
            try:
                major_s, minor_s = parts[1].split(".", 1)
                parsed = (int(major_s), int(minor_s))
                if cap is None or parsed > cap:
                    cap = parsed
            except ValueError:
                pass
        if len(parts) > 2:
            try:
                # memory.total with nounits is MiB
                vrams.append(float(parts[2]) / 1024.0)
            except ValueError:
                pass

    primary_vram = vrams[0] if vrams else 0.0
    return CudaProbe(
        available=True,
        capability=cap,
        gpu_name=names[0] if names else "",
        vram_gb=round(primary_vram, 1),
        gpu_count=len(names) or 1,
    )


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
        detail = "not found (rondine setup"
        if brew:
            detail += " or brew install llama.cpp"
        detail += ")"
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
    elif system in {"windows", "win32"}:
        plat = "windows"
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
    disk_free, disk_total = _disk_gb()
    cuda = _detect_cuda()
    apple = plat == "darwin" and arch_norm == "arm64"
    spark = _looks_like_spark(cuda.gpu_name, arch_norm, ram)
    if cuda.capability and cuda.capability[0] == 12 and plat == "linux" and arch_norm == "aarch64":
        spark = True

    info = HardwareInfo(
        platform=plat,
        arch=arch_norm,
        hostname=platform.node(),
        ram_gb=round(ram, 1),
        cpu_brand=_cpu_brand(),
        is_apple_silicon=apple,
        is_spark=spark,
        cuda_available=cuda.available,
        cuda_capability=cuda.capability,
        gpu_name=cuda.gpu_name,
        vram_gb=cuda.vram_gb if not spark else round(ram, 1),
        gpu_count=cuda.gpu_count,
        metal_available=apple,
        disk_free_gb=round(disk_free, 1),
        disk_total_gb=round(disk_total, 1),
        engines=[_engine_llama(), _engine_mlx(), _engine_vllm()],
    )
    if apple and not shutil.which("llama-server"):
        info.warnings.append(
            "llama-server not on PATH; run `rondine setup` or `brew install llama.cpp`"
        )
    if spark and not cuda.available:
        info.warnings.append("DGX Spark detected heuristically but nvidia-smi missing")
    if info.is_discrete_cuda and info.gpu_count > 1:
        info.warnings.append(
            f"{info.gpu_count} GPUs detected; planner sizes for GPU0 "
            f"({info.vram_gb}GB). Multi-GPU tensor parallel is opt-in via engine_args."
        )
    if cuda.available and not spark and not apple and info.vram_gb <= 0:
        info.warnings.append("nvidia-smi present but VRAM not reported; fit estimates may be wrong")
    return info
