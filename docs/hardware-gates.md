# Hardware validation records

Single-node (`0.1.0`) and dual-node (`0.2.0`) gates. Fill rows on real hardware.

## Template

| Field | Value |
|---|---|
| Date | |
| Host | |
| `rondine doctor` summary | |
| Plan | |
| Engine / quant | |
| Launch command | |
| `/v1/models` | |
| `rondine verify` | |
| Peak memory | |
| tok/s (prompt / gen) | |
| Notes | |

## Single-node NVIDIA (discrete CUDA)

- [ ] `rondine doctor` reports VRAM + GPU name + free disk
- [ ] `rondine suggest` matches `cuda-*` target and fits against VRAM when resident
- [ ] `rondine plan <moe> --memory-mode hybrid` selects hybrid when VRAM alone is short but RAM+VRAM is enough
- [ ] Hybrid dry-run emits `--cpu-moe` and `-ngl auto` (or equivalent) for MoE GGUFs
- [ ] `rondine setup` builds/installs llama.cpp with CUDA (or vLLM on Linux)
- [ ] `rondine serve` + `rondine verify --profile coding` pass

## Single-node Mac (Apple Silicon)

- [ ] `rondine setup` installs llama.cpp and/or MLX
- [ ] `rondine plan auto --profile coding` selects a fitting Qwen/Gemma variant
- [ ] `rondine plan glm-5.2` rejects on ≤48GB hosts without `--allow-oversize`
- [ ] `rondine plan glm-5.2 --memory-mode mmap --allow-oversize` is experimental, `fits=false`, and forces `-ngl 0`
- [ ] Insufficient free disk rejects mmap plans / pulls before download
- [ ] `rondine serve` + `rondine verify --profile coding` pass for a fitting small/medium model

## Single-node DGX Spark

- [ ] `nvidia-smi` reports compute capability 12.x
- [ ] vLLM container/image pulls successfully
- [ ] NVFP4 serve uses `flashinfer_b12x` when required
- [ ] `rondine verify --profile coding` pass

## Dual Mac (0.2)

- [ ] `rondine cluster doctor` passwordless SSH ok
- [ ] MLX `mlx.launch` JACCL serve works for a catalog MLX repo
- [ ] llama.cpp RPC marked experimental; not exposed beyond trusted LAN

## Dual Spark (0.2)

- [ ] RoCE / ConnectX bandwidth validated (no TCP fallback)
- [ ] NCCL interface pinning set (`NCCL_IB_HCA`, `MN_IF_NAME`, …)
- [ ] Ray leader/worker + `vllm serve --tensor-parallel-size 2` works
