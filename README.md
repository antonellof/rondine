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

**Hardware-aware local LLM launcher for Apple Silicon, NVIDIA GPUs, and DGX Spark.**

Rondine detects RAM and VRAM, recommends models that fit, applies tuned engine
settings, downloads the weights, and starts an OpenAI-compatible server. It is a
thin control plane over llama.cpp, MLX-LM, and vLLM—not another inference engine.

## Quick start

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/antonellof/rondine.git
cd rondine
uv tool install .

rondine doctor
rondine suggest --profile coding
rondine suggest --configure 1 --save-as coding
rondine setup
rondine pull
rondine serve --preset coding
rondine verify --profile coding
```

Your OpenAI-compatible endpoint is `http://127.0.0.1:8080/v1`.

`--profile coding` favors coding models, a 32K context, reasoning settings, and
single-client engine tuning. Use `--profile chat` for a 16K context and
chat-oriented sampling/concurrency. See the [CLI guide](docs/cli.md#suggest) for
every `suggest` option, including `--limit`, `--opt-in`, `--json`,
`--configure`, and `--save-as`.

Rondine doctor and suggest terminal demo

### Small-model smoke test

Test the complete plan → pull → serve → verify path with a curated 2GB model:

```bash
rondine plan qwen2.5-coder-3b --context 4096 --save-as small-coder
rondine pull qwen2.5-coder-3b
rondine serve --preset small-coder
rondine verify --name small-coder
rondine stop --name small-coder
```

Use `qwen2.5-coder-1.5b` instead for an approximately 1GB download.

## What Rondine configures

- **Apple Silicon:** MLX-LM or llama.cpp with MLX/GGUF models
- **NVIDIA GPUs:** llama.cpp or vLLM with GGUF, safetensors, or NVFP4 models
- **DGX Spark / GB10:** vLLM or llama.cpp with NVFP4/GGUF models
- **Homogeneous clusters:** native MLX/vLLM launchers or llama.cpp RPC

Hardware templates tune GPU offload, batch sizes, KV cache, parallelism, and
memory utilization. `rondine suggest` shows the resolved configuration before
`serve` applies it. On discrete GPUs, fit calculations use VRAM rather than
system RAM.

## Hybrid and oversized models

For a discrete GPU with enough system RAM, llama.cpp can keep MoE experts in
RAM while fitting dense layers to VRAM:

```bash
rondine plan glm-5.2 --memory-mode hybrid --context 4096 --save-as glm-hybrid
rondine pull
rondine serve --preset glm-hybrid
```

`auto` may choose this supported hybrid path when VRAM alone is insufficient but
combined RAM and VRAM meet the model requirement.

SSD-backed `mmap` is a separate, experimental escape hatch. It demand-pages
GGUF weights and is not true expert streaming:

```bash
# Planning is safe: it does not download or launch the model.
rondine plan glm-5.2 --quant UD-IQ1_S --context 4096 \
  --memory-mode mmap --allow-oversize --save-as glm-ssd

# Review the experimental plan before downloading hundreds of gigabytes.
rondine preset show glm-ssd
rondine pull
rondine serve --preset glm-ssd
```

`--allow-oversize` is an explicit safety acknowledgement: it permits Rondine to
create an `mmap` plan even though the model does not fit resident memory. It does
not make the model fit, bypass the free-disk check, or enable this mode in
automatic recommendations. Rondine accepts it only with `--memory-mode mmap`.

GLM-5.2 `UD-IQ1_S` needs about 223GB resident memory for practical use and at
least 230GB free disk. A 32GB Mac may map it, but page thrashing is expected to
make it unusably slow; Rondine therefore never selects this mode automatically.

## Useful commands

```bash
rondine models                         # curated models and fit status
rondine search "Qwen GGUF"             # search Hugging Face
rondine inspect org/model-repo         # inspect files, sizes, and quants
rondine plan org/model-repo --quant Q4_K_M --save-as custom
rondine preset list
rondine preset serve coding --dry-run  # inspect without starting
```



## Performance

On a 32GB M2 Pro, Qwen2.5-Coder 3B Q4_K_M ran fully on Metal at a median
66.6 tokens/second across three 128-token coding runs. See
[benchmarks](docs/benchmarks.md) for the method and limitations.

## Documentation

- [CLI guide and](docs/cli.md) `suggest` [/ memory-mode options](docs/cli.md)
- [Coding-client setup](docs/coding.md)
- [Engine tuning](docs/engine-tuning.md)
- [Hardware gates](docs/hardware-gates.md)
- [Cluster setup](docs/cluster.md)
- [Benchmarks](docs/benchmarks.md)



## License

Apache-2.0. Model weights retain their upstream licenses.

---


*Rondine means “swallow” in Italian.*