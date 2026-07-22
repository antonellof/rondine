"""Dry-run command generation snapshots."""

from __future__ import annotations

from rondine.engines.llama_cpp import LlamaCppAdapter
from rondine.engines.mlx import MlxAdapter
from rondine.engines.vllm import VllmAdapter


def _plan(engine: str, **kwargs: object) -> dict:
    base = {
        "selected": {
            "model_id": "qwen3.6-35b-a3b",
            "repo": "unsloth/Qwen3.6-35B-A3B-GGUF",
            "quant": "UD-Q4_K_XL",
            "engine": engine,
            "format": "gguf",
            "context": 32768,
            "sampling": {
                "temperature": 0.6,
                "top_p": 0.95,
                "top_k": 20,
                "chat_template_kwargs": {"enable_thinking": True},
            },
            "variant": {
                "include": ["*UD-Q4_K_XL*"],
                "mtp": False,
                "spark_moe_backend": None,
            },
        }
    }
    base["selected"].update(kwargs)  # type: ignore[arg-type]
    return base


def test_llama_serve_dry_flags() -> None:
    adapter = LlamaCppAdapter()
    plan = _plan("llama.cpp")
    plan["selected"]["engine_args"] = {
        "n_gpu_layers": 99,
        "flash_attn": True,
        "parallel": 1,
        "batch_size": 512,
        "ubatch_size": 128,
        "cache_type_k": "q4_0",
        "cache_type_v": "q4_0",
    }
    spec = adapter.build_serve(plan, host="127.0.0.1", port=8080)
    cmd = spec.command_line()
    assert "llama-server" in cmd
    assert "--temp" in cmd
    assert "0.6" in cmd
    assert "--chat-template-kwargs" in cmd
    assert "--parallel" in cmd
    assert "--batch-size" in cmd
    assert "--cache-type-k" in cmd
    assert spec.base_url == "http://127.0.0.1:8080/v1"


def test_llama_mtp_flags() -> None:
    plan = _plan(
        "llama.cpp",
        repo="unsloth/Qwen3.6-35B-A3B-MTP-GGUF",
        variant={"include": ["*UD-Q4_K_XL*"], "mtp": True},
    )
    spec = LlamaCppAdapter().build_serve(plan, host="127.0.0.1", port=8081)
    assert "--spec-type" in spec.argv
    assert "draft-mtp" in spec.argv


def test_mlx_serve_module() -> None:
    plan = _plan(
        "mlx",
        repo="unsloth/Qwen3.6-35B-A3B-UD-MLX-4bit",
        format="mlx",
        quant="4bit",
    )
    spec = MlxAdapter().build_serve(plan, host="0.0.0.0", port=9000)
    assert "mlx_lm.server" in spec.argv
    assert "unsloth/Qwen3.6-35B-A3B-UD-MLX-4bit" in spec.argv


def test_vllm_spark_moe_backend() -> None:
    plan = _plan(
        "vllm",
        repo="unsloth/Qwen3.6-35B-A3B-NVFP4-Fast",
        format="nvfp4",
        quant="NVFP4-Fast",
        variant={"spark_moe_backend": "flashinfer_b12x"},
    )
    plan["selected"]["engine_args"] = {
        "gpu_memory_utilization": 0.92,
        "max_model_len": 65536,
        "enable_prefix_caching": True,
        "tensor_parallel_size": 1,
        "dtype": "auto",
    }
    spec = VllmAdapter().build_serve(plan, host="0.0.0.0", port=8000)
    joined = spec.command_line()
    assert "vllm" in joined
    assert "flashinfer_b12x" in joined
    assert "unsloth/Qwen3.6-35B-A3B-NVFP4-Fast" in joined
    assert "--gpu-memory-utilization" in joined
    assert "--max-model-len" in joined
    assert "--enable-prefix-caching" in joined


def test_setup_dry_run_returns_commands() -> None:
    cmds = LlamaCppAdapter().setup(dry_run=True)
    assert cmds
    mlx_cmds = MlxAdapter().setup(dry_run=True)
    assert mlx_cmds
