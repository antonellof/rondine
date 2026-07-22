# Changelog

## Unreleased

- Memory strategies: `resident`, supported CUDA `hybrid` (RAM+VRAM / `--cpu-moe`), and explicit experimental `mmap` SSD paging (`--memory-mode`, `--allow-oversize`)
- Disk free-space detection in `doctor` and a pre-download guard for sharded GGUFs
- Expanded GLM-5.2 Unsloth quant ladder with verified Hub sizes (`UD-IQ1_S` … `UD-Q4_K_M`)
- llama.cpp serve flags for fit helpers, mmap/mlock policy, MoE CPU offload, and tensor split
- Docs and blog updates for hybrid / oversized MoE workflows

## 0.1.0

- Initial Rondine control-plane CLI (Python 3.11+)
- Catalog presets: Qwen3.6, Gemma 4, DeepSeek-V4-Flash, GLM-5.2
- Engines: llama.cpp (GGUF), MLX-LM (Apple Silicon), vLLM (DGX Spark / NVFP4)
- Discrete NVIDIA GPU targets (`cuda-8`…`cuda-80`) with VRAM-based fit
- Commands: doctor, models, suggest, plan, setup, pull, serve, preset, stop, verify
- Coding profile + docs/coding.md + docs/engine-tuning.md
- Cluster inventory helpers (`cluster init/doctor/plan/serve` dry-run)

## 0.2.0 (planned)

- Validated dual-Mac MLX JACCL and dual-Spark Ray/NCCL hardware gates
