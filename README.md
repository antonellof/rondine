```
                         ___
                   ___.-'   `-.
              .---'            `.
             /   .-.   .-.       \
            |   /  o\ /o  \       |
             \  \  (_X_)  /      /
              `._`--...--'_.---.'
             __/  `.   .'  \__
         _.-' /     `-'     \ `-._
      .-'   .'               `.   `-.
     /    .'    r o n d i n e   `.    \
    |   .'                       `.   |
     \_/                           \_/
   tiny bird · suspiciously large models
```

**Hardware-aware local LLM launcher for Mac, NVIDIA GPUs, and DGX Spark.**

Rondine is a thin control plane over battle-tested backends:

| Hardware | Engine | Format |
|---|---|---|
| Apple Silicon | MLX-LM or llama.cpp | MLX / GGUF |
| NVIDIA discrete GPU | llama.cpp or vLLM | GGUF / safetensors / NVFP4 |
| DGX Spark / GB10 | vLLM or llama.cpp | NVFP4 / GGUF |
| Homogeneous clusters | MLX / vLLM / llama.cpp RPC | native per engine |

It scans your machine (RAM or **GPU VRAM**), suggests models that fit, builds engine-tuned launch configs, downloads weights, and starts an OpenAI-compatible local server. Named presets make restart one command.

Rondine does **not** reinvent inference. It installs and drives llama.cpp, MLX-LM, and vLLM.

## Install

```bash
uv tool install .
# or editable:
uv pip install -e ".[dev]"
```

Requires Python 3.11+.

## Quick start

```bash
rondine doctor                         # scan hardware + engines
rondine suggest --profile coding       # ranked models + engine configs for this machine
rondine suggest --configure 1 --save-as coding
rondine setup
rondine pull                           # uses the configured plan
rondine serve --preset coding
rondine verify --profile coding
```

Discover more on the Hub when the curated list isn’t enough:

```bash
rondine search "Qwen3.6 35B GGUF"
rondine inspect org/model-repo
rondine plan org/model-repo --quant Q4_K_M --save-as qwen-gguf
```

Dry-run any launch:

```bash
rondine serve qwen3.6-27b --profile coding --dry-run
rondine preset serve coding --dry-run
```

## How suggestion works

1. **Detect** RAM / VRAM / Apple Silicon / Spark / CUDA and which engines are installed.
2. **Match** a hardware target (`mac-36`, `cuda-24`, `spark-128`, …) with preferred engine + suggested models.
3. **Score** curated variants (provider, quant, headroom, coding priority).
4. **Configure** engine knobs from `catalog/hardware.toml` templates (llama.cpp `-ngl` / batch / KV cache, vLLM `--gpu-memory-utilization` / `--max-model-len`, …).
5. **Save** a plan + optional named preset under `~/.rondine/presets/` for one-command restart.

On discrete NVIDIA GPUs, fit estimates use **GPU VRAM** (not system RAM). Multi-GPU hosts size against GPU0 by default; tensor parallel is opt-in.

## Commands

| Command | Purpose |
|---|---|
| `doctor` | Probe hardware, engines, and memory |
| `suggest` | Rank models that fit + show launch configs |
| `models` | List curated catalog entries and fit status |
| `search` | Live Hugging Face Hub discovery |
| `inspect` | Hub repo files, sizes, recommended quant |
| `plan` | Recommend engine / quant (catalog id, auto, or `org/name`) |
| `setup` | Install pinned engine toolchains |
| `pull` | Download a model for the resolved plan |
| `serve` | Launch OpenAI-compatible server (`--preset`, `--save-as`) |
| `preset` | `list` / `show` / `save` / `serve` / `delete` named presets |
| `stop` | Stop a managed server |
| `verify` | Health + coding smoke tests |
| `cluster doctor/plan/serve` | Homogeneous dual-node helpers |

## Coding clients

Point any OpenAI-compatible coding client at:

```text
http://127.0.0.1:8080/v1
```

See [docs/coding.md](docs/coding.md) and [docs/engine-tuning.md](docs/engine-tuning.md).

## Cluster notes (0.2)

- Mac: MLX `mlx.launch` JACCL for supported models; llama.cpp RPC is experimental and trusted-LAN only.
- Spark: NVIDIA container + Ray/NCCL over RoCE; no silent Ethernet fallback.

## License

Apache-2.0. Model weights retain their upstream licenses.
