# Changelog

## 0.1.0

- Initial Rondine control-plane CLI (Python 3.11+)
- Catalog presets: Qwen3.6, Gemma 4, DeepSeek-V4-Flash, GLM-5.2
- Engines: llama.cpp (GGUF), MLX-LM (Apple Silicon), vLLM (DGX Spark / NVFP4)
- Commands: doctor, models, plan, setup, pull, serve, stop, verify
- Coding profile + docs/coding.md
- Cluster inventory helpers (`cluster init/doctor/plan/serve` dry-run)

## 0.2.0 (planned)

- Validated dual-Mac MLX JACCL and dual-Spark Ray/NCCL hardware gates
