"""llama.cpp adapter — GGUF via Hugging Face + llama-server."""

from __future__ import annotations

import json
import os
import platform
import subprocess
from pathlib import Path
from typing import Any

from rondine.engines.base import EngineAdapter, LaunchSpec, append_flag, selected_engine_args
from rondine.paths import engines_dir, models_dir, which


class LlamaCppAdapter(EngineAdapter):
    name = "llama.cpp"

    def setup(self, *, dry_run: bool = False) -> list[str]:
        cmds: list[str] = []
        if which("llama-server"):
            cmds.append("# llama-server already on PATH")
            return cmds
        system = platform.system().lower()
        if system == "darwin" and which("brew"):
            cmd = ["brew", "install", "llama.cpp"]
            cmds.append(" ".join(cmd))
            if not dry_run:
                subprocess.run(cmd, check=False)
            return cmds
        dest = engines_dir() / "llama.cpp"
        clone = [
            "git",
            "clone",
            "--depth",
            "1",
            "https://github.com/ggml-org/llama.cpp",
            str(dest),
        ]
        cmds.append(" ".join(clone))
        cmake_flags = ["-DBUILD_SHARED_LIBS=OFF"]
        if which("nvcc"):
            cmake_flags.append("-DGGML_CUDA=ON")
        else:
            cmake_flags.append("-DGGML_CUDA=OFF")
        build = ["cmake", str(dest), "-B", str(dest / "build"), *cmake_flags]
        make = [
            "cmake",
            "--build",
            str(dest / "build"),
            "--config",
            "Release",
            "-j",
            "--target",
            "llama-cli",
            "llama-server",
        ]
        cmds.append(" ".join(build))
        cmds.append(" ".join(make))
        if not dry_run:
            if not dest.exists():
                subprocess.run(clone, check=True)
            subprocess.run(build, check=True)
            subprocess.run(make, check=True)
        return cmds

    def _server_bin(self) -> str:
        path = which("llama-server")
        if path:
            return path
        local = engines_dir() / "llama.cpp" / "build" / "bin" / "llama-server"
        if local.is_file():
            return str(local)
        local2 = engines_dir() / "llama.cpp" / "llama-server"
        if local2.is_file():
            return str(local2)
        return "llama-server"

    def model_dir(self, repo: str) -> Path:
        safe = repo.replace("/", "__")
        path = models_dir() / "gguf" / safe
        path.mkdir(parents=True, exist_ok=True)
        return path

    def pull(self, plan: dict[str, Any], *, dry_run: bool = False) -> list[str]:
        selected = plan.get("selected") or plan
        repo = selected["repo"]
        include = (selected.get("variant") or {}).get("include") or [f"*{selected['quant']}*"]
        dest = self.model_dir(repo)
        cmds: list[str] = []
        for pattern in include:
            cmd = [
                "hf",
                "download",
                repo,
                "--local-dir",
                str(dest),
                "--include",
                pattern,
            ]
            cmds.append(" ".join(cmd))
            if not dry_run:
                if which("hf"):
                    subprocess.run(cmd, check=True)
                else:
                    from huggingface_hub import snapshot_download

                    snapshot_download(
                        repo_id=repo,
                        local_dir=str(dest),
                        allow_patterns=[pattern],
                    )
        return cmds

    def _find_primary_gguf(self, dest: Path, quant: str) -> Path | None:
        if not dest.exists():
            return None
        files = sorted(dest.rglob("*.gguf"))
        preferred = [f for f in files if quant in f.name and "mmproj" not in f.name.lower()]
        if preferred:
            for f in preferred:
                if "00001-of-" in f.name or "-of-" not in f.name:
                    return f
            return preferred[0]
        non_mm = [f for f in files if "mmproj" not in f.name.lower()]
        return non_mm[0] if non_mm else None

    def _find_mmproj(self, dest: Path) -> Path | None:
        for f in sorted(dest.rglob("*.gguf")):
            if "mmproj" in f.name.lower():
                return f
        return None

    def build_serve(self, plan: dict[str, Any], *, host: str, port: int) -> LaunchSpec:
        selected = plan.get("selected") or plan
        repo = selected["repo"]
        quant = selected["quant"]
        model_id = selected["model_id"]
        context = int(selected.get("context", 32768))
        sampling = selected.get("sampling") or {}
        variant = selected.get("variant") or {}
        dest = self.model_dir(repo)
        primary = self._find_primary_gguf(dest, quant)
        bin_path = self._server_bin()

        notes: list[str] = []
        argv = [bin_path]
        if primary is None:
            argv += ["-hf", f"{repo}:{quant}"]
            notes.append("model not found locally; llama-server will download via -hf")
        else:
            argv += ["--model", str(primary)]
            mmproj = self._find_mmproj(dest)
            if mmproj:
                argv += ["--mmproj", str(mmproj)]

        alias = f"rondine/{model_id}"
        argv += [
            "--host",
            host,
            "--port",
            str(port),
            "--alias",
            alias,
            "--ctx-size",
            str(context),
        ]

        if "temperature" in sampling:
            argv += ["--temp", str(sampling["temperature"])]
        if "top_p" in sampling:
            argv += ["--top-p", str(sampling["top_p"])]
        if "top_k" in sampling:
            argv += ["--top-k", str(sampling["top_k"])]
        if "min_p" in sampling:
            argv += ["--min-p", str(sampling["min_p"])]
        if "presence_penalty" in sampling:
            argv += ["--presence-penalty", str(sampling["presence_penalty"])]

        kwargs = sampling.get("chat_template_kwargs")
        if isinstance(kwargs, dict) and kwargs:
            argv += ["--chat-template-kwargs", json.dumps(kwargs)]

        if variant.get("mtp"):
            argv += ["--spec-type", "draft-mtp", "--spec-draft-n-max", "2"]
            notes.append("MTP speculative decoding enabled")

        args = selected_engine_args(plan)
        ngl = args.get("n_gpu_layers")
        if ngl is None and (platform.system().lower() == "darwin" or which("nvidia-smi")):
            ngl = 99
        if ngl is not None:
            argv += ["-ngl", str(int(ngl))]

        if args.get("flash_attn"):
            append_flag(argv, "--flash-attn", "on")
        if args.get("cont_batching", True):
            append_flag(argv, "--cont-batching")
        if args.get("mlock"):
            append_flag(argv, "--mlock")
        prio = int(args.get("prio") or 0)
        if prio > 0:
            append_flag(argv, "--prio", prio)
        if "parallel" in args:
            append_flag(argv, "--parallel", int(args["parallel"]))
        if "batch_size" in args:
            append_flag(argv, "--batch-size", int(args["batch_size"]))
        if "ubatch_size" in args:
            append_flag(argv, "--ubatch-size", int(args["ubatch_size"]))
        threads = int(args.get("threads") or 0)
        if threads > 0:
            append_flag(argv, "--threads", threads)
        if args.get("cache_type_k"):
            append_flag(argv, "--cache-type-k", args["cache_type_k"])
        if args.get("cache_type_v"):
            append_flag(argv, "--cache-type-v", args["cache_type_v"])
        if args:
            notes.append(
                "engine template: "
                + ", ".join(f"{k}={v}" for k, v in sorted(args.items()) if v not in (None, ""))
            )

        env = {**os.environ, "LLAMA_CACHE": str(dest)}
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
