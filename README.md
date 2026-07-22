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
          forked wings · sparse MoE
```

**Hardware-aware local LLM launcher for Mac and DGX Spark.**

Rondine is a thin control plane over battle-tested backends:

| Hardware | Engine | Format |
|---|---|---|
| Apple Silicon | MLX-LM or llama.cpp | MLX / GGUF |
| DGX Spark / GB10 | vLLM or llama.cpp | NVFP4 / GGUF |
| Homogeneous clusters | MLX / vLLM / llama.cpp RPC | native per engine |

It ships reviewed presets for frontier open models (Qwen3.6, Gemma 4, DeepSeek-V4-Flash, GLM-5.2), picks a quant that fits your machine, downloads weights, and launches an OpenAI-compatible local server.

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
rondine doctor
rondine models
rondine plan auto --profile coding
rondine setup
rondine pull qwen3.6-35b-a3b
rondine serve qwen3.6-35b-a3b --profile coding
rondine verify --profile coding
```

Dry-run any launch:

```bash
rondine serve qwen3.6-27b --profile coding --dry-run
```

## Commands

| Command | Purpose |
|---|---|
| `doctor` | Probe hardware, engines, and memory |
| `models` | List catalog entries and fit status |
| `plan` | Recommend engine / quant / context |
| `setup` | Install pinned engine toolchains |
| `pull` | Download a model for the resolved plan |
| `serve` | Launch OpenAI-compatible server |
| `stop` | Stop a managed server |
| `verify` | Health + coding smoke tests |
| `cluster doctor/plan/serve` | Homogeneous dual-node helpers |

## Coding clients

Point any OpenAI-compatible coding client at:

```text
http://127.0.0.1:8080/v1
```

See [docs/coding.md](docs/coding.md).

## Cluster notes (0.2)

- Mac: MLX `mlx.launch` JACCL for supported models; llama.cpp RPC is experimental and trusted-LAN only.
- Spark: NVIDIA container + Ray/NCCL over RoCE; no silent Ethernet fallback.

## License

Apache-2.0. Model weights retain their upstream licenses.
