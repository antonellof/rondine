"""MLX-LM adapter for Apple Silicon."""

from __future__ import annotations

import os
import platform
import subprocess
from pathlib import Path
from typing import Any

from rondine.engines.base import EngineAdapter, LaunchSpec, selected_engine_args
from rondine.paths import engines_dir, which


class MlxAdapter(EngineAdapter):
    name = "mlx"

    def _venv(self) -> Path:
        return engines_dir() / "mlx-venv"

    def _python(self) -> str:
        venv = self._venv()
        candidate = venv / "bin" / "python"
        if candidate.is_file():
            return str(candidate)
        return which("python3") or "python3"

    def setup(self, *, dry_run: bool = False) -> list[str]:
        cmds: list[str] = []
        if platform.system().lower() != "darwin" or platform.machine() != "arm64":
            cmds.append("# MLX setup skipped: Apple Silicon required")
            return cmds
        venv = self._venv()
        uv = which("uv")
        if uv:
            create = [uv, "venv", str(venv), "--python", "3.12"]
            install = [
                uv,
                "pip",
                "install",
                "--python",
                str(venv / "bin" / "python"),
                "mlx-lm",
                "huggingface_hub",
            ]
        else:
            create = ["python3", "-m", "venv", str(venv)]
            install = [str(venv / "bin" / "pip"), "install", "-U", "mlx-lm", "huggingface_hub"]
        cmds.append(" ".join(create))
        cmds.append(" ".join(install))
        if not dry_run:
            if not (venv / "bin" / "python").exists():
                subprocess.run(create, check=True)
            subprocess.run(install, check=True)
        return cmds

    def pull(self, plan: dict[str, Any], *, dry_run: bool = False) -> list[str]:
        selected = plan.get("selected") or plan
        repo = selected["repo"]
        # MLX-LM downloads on first load; optionally prefetch via huggingface_hub
        cmd = [
            self._python(),
            "-c",
            (
                "from huggingface_hub import snapshot_download; "
                f"snapshot_download('{repo}')"
            ),
        ]
        line = " ".join(cmd)
        if not dry_run:
            subprocess.run(cmd, check=True)
        return [line]

    def build_serve(self, plan: dict[str, Any], *, host: str, port: int) -> LaunchSpec:
        selected = plan.get("selected") or plan
        repo = selected["repo"]
        model_id = selected["model_id"]
        alias = f"rondine/{model_id}"
        py = self._python()
        # Prefer module form so venv isolation works
        argv = [
            py,
            "-m",
            "mlx_lm.server",
            "--model",
            repo,
            "--host",
            host,
            "--port",
            str(port),
        ]
        args = selected_engine_args(plan)
        notes = [
            "MLX loads Hugging Face MLX repos; first run downloads weights.",
            f"alias/model id for clients: {alias} (engine may report HF id)",
        ]
        env = dict(os.environ)
        if args.get("metal_fast_synch", True):
            env["MLX_METAL_FAST_SYNCH"] = "1"
            notes.append("MLX_METAL_FAST_SYNCH=1 (Metal sync throughput)")
        if args.get("trust_remote_code"):
            notes.append("trust_remote_code enabled for Hub config")
        if args:
            notes.append(
                "engine template: "
                + ", ".join(f"{k}={v}" for k, v in sorted(args.items()) if v not in (None, ""))
            )
        # Sampling stays client-side for MLX server; surface it for presets.
        sampling = selected.get("sampling") or {}
        if sampling:
            notes.append(
                "client sampling: "
                + ", ".join(
                    f"{k}={v}"
                    for k, v in sampling.items()
                    if k != "chat_template_kwargs" and v not in (None, "")
                )
            )
        return LaunchSpec(
            engine=self.name,
            argv=argv,
            env=env,
            host=host,
            port=port,
            alias=alias,
            model_id=model_id,
            notes=notes,
        )
