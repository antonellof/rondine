# Engine tuning notes

Rondine merges performance knobs from `catalog/hardware.toml`:

`defaults` → profile (`coding` / `chat`) → hardware template (`mac`, `mac-tight`, `spark`).

## Apple Silicon + llama.cpp

Useful baseline for Metal:

- `-ngl 99` — offload all layers
- `--flash-attn on` — required for efficient long context and KV quant
- `--batch-size` / `--ubatch-size` `2048` when memory allows (faster prefill)
- `--cache-type-k/v q8_0` — roughly half KV RAM vs f16 with little quality loss
- `--parallel 1` for single-user coding (context budget is not split)
- `--mlock` / `--prio 2` when there is headroom (avoid swap under load)

On tight unified-memory machines (≤36GB with large models), Rondine uses the
`mac-tight` template (smaller batches) to reduce prefill OOM risk.

Throughput sweeps over `--parallel`, `--batch-size`, and `--ubatch-size` are
useful when serving many concurrent clients; coding presets stay single-slot.

## Apple Silicon + MLX

- Prefer native MLX weights over GGUF when both fit.
- Set `MLX_METAL_FAST_SYNCH=1` for Metal sync throughput (large wins on clusters).
- Single-host `mlx_lm.server`; multi-Mac uses JACCL / `mlx.launch` (see cluster docs).

## Discrete NVIDIA + llama.cpp / vLLM

- Fit against **VRAM**, not system RAM (`cuda-8` … `cuda-80` targets).
- Default engine order: llama.cpp then vLLM (consumer-friendly GGUF path).
- Large VRAM targets (`cuda-48`+) prefer vLLM when available.
- Templates: `cuda` (batch 2048) and `cuda-tight` (≤12GB VRAM).

## DGX Spark + vLLM

- Prefer NVFP4 / engine-native formats when available.
- High `--gpu-memory-utilization` (~0.90–0.92) and long `--max-model-len`.
- MoE backends (e.g. `flashinfer_b12x`) come from the model variant, not the template.

## Memory planning

Unified-memory machines typically cannot dedicate 100% of RAM to weights + KV.
Rondine reserves OS headroom in the planner; treat ~70–75% of system RAM as a
practical inference budget when estimating by hand.

## External references

Community launchers / calculators that informed these defaults:

- [llama-throughput-lab](https://github.com/alexziskind1/llama-throughput-lab) — llama-server parallel/batch sweeps
- [mlx-jaccl-cluster](https://github.com/alexziskind1/mlx-jaccl-cluster) — MLX JACCL + `MLX_METAL_FAST_SYNCH`
- [llm-inference-calculator](https://github.com/alexziskind1/llm-inference-calculator) — VRAM / unified-memory estimates
- [draftbench](https://github.com/alexziskind1/draftbench) — speculative decoding sweeps
- [alexziskind1 repositories](https://github.com/alexziskind1?tab=repositories)
