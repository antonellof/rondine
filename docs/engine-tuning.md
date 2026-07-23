# Engine tuning notes

Rondine ships **optimized configurations** for common local hardware. Performance knobs
are merged from `catalog/hardware.toml`:

`defaults` → profile (`coding` / `chat`) → hardware template (`mac`,
`mac-tight`, `cuda`, `cuda-tight`, `spark`, `hybrid-moe`, or
`mmap-experimental`).

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

### Resident, hybrid, and mmap modes

- `resident` requires the full supported estimate to fit VRAM or unified memory.
- `hybrid` is for llama.cpp GGUFs on discrete CUDA hosts. Rondine combines RAM
  and VRAM for capacity, emits auto-fit GPU-layer settings, and uses `--cpu-moe`
  for MoE models. Review the exact resolved flags in `rondine suggest` or
  `rondine plan`; placement is still controlled by llama.cpp, so the combined
  estimate is approximate.
- `mmap` is an explicit experimental mode for a GGUF larger than physical
  memory. It keeps mmap enabled, disables mlock, uses one slot and q4_1 KV, and
  requires `--allow-oversize`. A non-fitting mmap plan remains marked
  `fits=false`; it is not promoted to a supported fit.

On oversized Apple Silicon launches Rondine forces `-ngl 0`. CPU and Metal use
the same unified memory, so GPU offload adds no capacity, and current partial
Metal mmap paths can register a very large residency span and OOM. On discrete
GPUs, supported hybrid plans may use auto-fit GPU layers.

Default mmap is demand paging through the operating-system page cache, not a
dedicated expert streamer. The proposed llama.cpp `--moe-stream*` flags remain
unmerged, so Rondine does not emit them. GLM-5.2's complete IndexShare/DSA and
MTP optimizations are also still pending upstream; current llama.cpp support
means the model can load and generate, not that every architecture optimization
is available.

### GLM-5.2 capacity

The smallest curated quant, `UD-IQ1_S`, has six shards totaling 216.715GB and a
documented practical memory floor near 223GB. `UD-IQ2_M` totals 238.578GB and
needs about 245GB. Rondine also catalogs verified 1-, 3-, and 4-bit quality
tiers for larger-memory machines.

A 32GB M2 Pro can only attempt pathological SSD paging. Even though mmap may
make the file addressable, changing expert routes can repeatedly fault pages
from storage. Reserve at least 230GB free disk for `UD-IQ1_S` and expect
diagnostic, not interactive, performance.

## Measured baseline

The reproducible small-model smoke benchmark is documented in
[benchmarks.md](benchmarks.md). On an Apple M2 Pro with 32GB unified memory,
Qwen2.5-Coder 3B Q4_K_M fully offloaded to Metal and produced a median
66.6 tokens/second across three 128-token coding requests.

## External references

Community launchers / calculators that informed these defaults:

- [llama-throughput-lab](https://github.com/alexziskind1/llama-throughput-lab) — llama-server parallel/batch sweeps
- [mlx-jaccl-cluster](https://github.com/alexziskind1/mlx-jaccl-cluster) — MLX JACCL + `MLX_METAL_FAST_SYNCH`
- [llm-inference-calculator](https://github.com/alexziskind1/llm-inference-calculator) — VRAM / unified-memory estimates
- [draftbench](https://github.com/alexziskind1/draftbench) — speculative decoding sweeps
- [alexziskind1 repositories](https://github.com/alexziskind1?tab=repositories)
