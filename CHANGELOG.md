# Changelog

## Unreleased

## 0.1.0

- Initial Rondine control-plane CLI (Python 3.11+)
- Catalog presets: Qwen3.6, Gemma 4, DeepSeek-V4-Flash, GLM-5.2
- Engines: llama.cpp (GGUF), MLX-LM (Apple Silicon), vLLM (DGX Spark / NVFP4)
- Discrete NVIDIA GPU targets (`cuda-8`…`cuda-80`) with VRAM-based fit
- Commands: doctor, models, suggest, plan, setup, pull, serve, preset, stop, verify
- Coding profile + docs/coding.md + docs/engine-tuning.md
- Cluster inventory helpers (`cluster init/doctor/plan/serve` dry-run)
- Memory strategies: `resident`, supported CUDA `hybrid` (RAM+VRAM / `--cpu-moe`), and explicit experimental `mmap` SSD paging (`--memory-mode`, `--allow-oversize`)
- Disk free-space detection in `doctor` and a pre-download guard for sharded GGUFs
- Official curl / PowerShell installers with GitHub Release checksum verification
- Documentation site at [rondine.dev](https://rondine.dev) (GitHub Pages)

## 0.2.0 (planned)

- Validated dual-Mac MLX JACCL and dual-Spark Ray/NCCL hardware gates
