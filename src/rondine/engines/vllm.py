"""vLLM adapter — NVFP4 / safetensors on DGX Spark via container or local CLI."""

from __future__ import annotations

import os
import platform
import subprocess
from typing import Any

from rondine.engines.base import EngineAdapter, LaunchSpec, append_flag, selected_engine_args
from rondine.paths import which

# Pinned NGC tag used by NVIDIA Spark playbooks (override with RONDINE_VLLM_IMAGE).
DEFAULT_VLLM_IMAGE = os.environ.get("RONDINE_VLLM_IMAGE", "nvcr.io/nvidia/vllm:26.04-py3")


class VllmAdapter(EngineAdapter):
    name = "vllm"

    def setup(self, *, dry_run: bool = False) -> list[str]:
        cmds: list[str] = []
        if platform.system().lower() != "linux":
            cmds.append("# vLLM Spark path is Linux-only; skipping install on this host")
            return cmds
        if which("docker"):
            pull = ["docker", "pull", DEFAULT_VLLM_IMAGE]
            cmds.append(" ".join(pull))
            if not dry_run:
                subprocess.run(pull, check=False)
            return cmds
        # Local uv install fallback (non-container)
        uv = which("uv")
        if uv:
            venv_path = os.path.expanduser("~/.rondine/engines/vllm-venv")
            create = [uv, "venv", venv_path, "--python", "3.13"]
            install = [
                uv,
                "pip",
                "install",
                "--python",
                f"{venv_path}/bin/python",
                "vllm>=0.25.0",
                "flashinfer-python>=0.6.13",
                "nvidia-cutlass-dsl>=4.5.2",
                "--torch-backend=auto",
            ]
            cmds.append(" ".join(create))
            cmds.append(" ".join(install))
            if not dry_run:
                subprocess.run(create, check=False)
                subprocess.run(install, check=False)
            return cmds
        cmds.append("# neither docker nor uv found; install docker or uv for vLLM")
        return cmds

    def pull(self, plan: dict[str, Any], *, dry_run: bool = False) -> list[str]:
        selected = plan.get("selected") or plan
        repo = selected["repo"]
        cmd = ["hf", "download", repo]
        line = " ".join(cmd)
        if not dry_run:
            if which("hf"):
                subprocess.run(cmd, check=False)
            else:
                from huggingface_hub import snapshot_download

                snapshot_download(repo_id=repo)
        return [line]

    def build_serve(self, plan: dict[str, Any], *, host: str, port: int) -> LaunchSpec:
        selected = plan.get("selected") or plan
        repo = selected["repo"]
        model_id = selected["model_id"]
        variant = selected.get("variant") or {}
        alias = f"rondine/{model_id}"
        notes: list[str] = []
        args = selected_engine_args(plan)
        context = int(selected.get("context") or args.get("max_model_len") or 32768)

        inner = ["vllm", "serve", repo, "--host", host, "--port", str(port)]
        append_flag(inner, "--max-model-len", int(args.get("max_model_len") or context))
        if "gpu_memory_utilization" in args:
            append_flag(
                inner, "--gpu-memory-utilization", float(args["gpu_memory_utilization"])
            )
        if "tensor_parallel_size" in args:
            append_flag(inner, "--tensor-parallel-size", int(args["tensor_parallel_size"]))
        if args.get("dtype"):
            append_flag(inner, "--dtype", args["dtype"])
        if args.get("enable_prefix_caching"):
            append_flag(inner, "--enable-prefix-caching")
        if args.get("enforce_eager"):
            append_flag(inner, "--enforce-eager")

        moe = variant.get("spark_moe_backend")
        env = {**os.environ}
        if moe:
            inner += ["--moe-backend", str(moe)]
            env["CUTE_DSL_ARCH"] = env.get("CUTE_DSL_ARCH", "sm_121a")
            notes.append(f"Spark MoE backend: {moe}")
            notes.append("CUTE_DSL_ARCH=sm_121a")
        if args:
            notes.append(
                "engine template: "
                + ", ".join(f"{k}={v}" for k, v in sorted(args.items()) if v not in (None, ""))
            )

        if which("docker"):
            argv = [
                "docker",
                "run",
                "--rm",
                "--gpus",
                "all",
                "--network",
                "host",
                "-e",
                f"CUTE_DSL_ARCH={env.get('CUTE_DSL_ARCH', 'sm_121a')}",
                DEFAULT_VLLM_IMAGE,
                *inner,
            ]
            notes.append(f"container image: {DEFAULT_VLLM_IMAGE}")
            return LaunchSpec(
                engine=self.name,
                argv=argv,
                env=env,
                host=host,
                port=port,
                alias=alias,
                model_id=model_id,
                notes=notes,
                container=True,
            )

        return LaunchSpec(
            engine=self.name,
            argv=inner,
            env=env,
            host=host,
            port=port,
            alias=alias,
            model_id=model_id,
            notes=notes,
        )
