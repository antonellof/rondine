# CLI guide

Run `rondine COMMAND --help` for the option list installed with your version.
This guide explains the choices that affect model recommendations.

Rondine enables colors automatically for interactive terminals. If terminal
detection is incorrect, force the mode before the command:

```bash
rondine --color doctor
rondine --color suggest
# Persist for the current shell:
export RONDINE_COLOR=1
```

Use `--no-color` or `RONDINE_COLOR=0` for plain output.

## Suggest

`rondine suggest` detects the current machine, ranks curated configurations, and
supplements them with current Hugging Face search results that fit its available
memory. Hub failures are non-fatal: the command falls back to the catalog.

```bash
rondine suggest --profile coding
```

Each result includes its rank, estimated memory use and headroom, context size,
engine arguments, sampling settings, and an equivalent `serve` command.
Interactive terminals display each configuration as a separated card.

### Profiles

`--profile coding` is the default. It favors models with a higher coding
priority, uses a 32,768-token context by default, enables model-appropriate
reasoning where supported, and tunes engines for one interactive coding client.

`--profile chat` uses a 16,384-token context, disables explicit reasoning for
supported model families, and tunes llama.cpp for more concurrent chat slots.

Profiles affect ranking, context, sampling, reasoning flags, and engine
configuration. They do not change model weights. Model-family settings can
override the general profile defaults; `suggest` prints the resolved values.

```bash
rondine suggest --profile chat
```

### Options

`--profile coding|chat`

Select the workload profile described above. Default: `coding`.

`--limit INTEGER`

Set the maximum number of ranked configurations to display. Default: `5`.

```bash
rondine suggest --limit 10
```

`--opt-in` / `--no-opt-in`

Include or exclude models marked opt-in. These are usually unusually large,
specialized, or impractical on common hardware. They must still pass the
planner's memory checks. Default: `--no-opt-in`.

```bash
rondine suggest --opt-in
```

`--hub` / `--no-hub`

Search Hugging Face and inspect the leading results before merging fitting
repositories into the recommendations. This is enabled by default. Use
`--no-hub` for an offline, deterministic catalog-only result.

```bash
rondine suggest --no-hub
```

`--hub-query TEXT`

Override the Hub search text. The default is `coder` for the coding profile and
`instruct` for the chat profile. Rondine still filters toward the preferred
engine and rejects repositories that do not fit the detected hardware.

```bash
rondine suggest --hub-query "Qwen coder"
```

`--json`

Write the complete result as JSON instead of human-readable output. This is
intended for scripts and includes detected hardware, selected hardware target,
engine order, suggestions, estimates, arguments, and notes.

```bash
rondine suggest --profile coding --json > suggestions.json
```

`-i` / `--interactive`

Show an arrow-key menu after the recommendations. Use `↑`/`↓` (or `j`/`k`) to
move, Enter to select and configure a model, or `q` to cancel. When input is
not attached to a terminal, Rondine falls back to a numbered prompt.

```bash
rondine suggest --interactive
rondine suggest -i --save-as coding
```

Interactive mode cannot be combined with `--json` or `--configure`.

`--configure INTEGER`

Save the suggestion with that rank as the active plan. The rank is the `#N`
shown in the output and must be present in the current result list.

```bash
rondine suggest --configure 1
rondine setup
rondine pull
rondine serve
```

`--save-as NAME`

With `--configure`, also save the selected plan as a reusable preset. The name
is user-defined; `coding` is only an example.

```bash
rondine suggest --profile coding --configure 1 --save-as work
rondine serve --preset work
```

`--help`

Print the command's current usage and option summary.

## Typical workflows

Inspect recommendations without changing configuration:

```bash
rondine suggest --profile coding
```

Select the top recommendation and save it for later:

```bash
rondine suggest --profile coding --configure 1 --save-as coding
rondine setup
rondine pull
rondine serve --preset coding
```

Compare coding and chat recommendations:

```bash
rondine suggest --profile coding --json > coding.json
rondine suggest --profile chat --json > chat.json
```

The detected hardware and installed engines can change the ranking, so rank
`#1` is not guaranteed to identify the same model on another machine.

## Plan, pull, and serve memory modes

`plan`, `pull`, and `serve` share two related options:

`--memory-mode auto|resident|hybrid|mmap`

- `auto` (default): prefer a resident fit; on discrete CUDA GGUFs, fall back to a
  supported hybrid RAM+VRAM plan when VRAM alone is insufficient.
- `resident`: require the estimate to fit the primary budget (VRAM or unified RAM).
- `hybrid`: require llama.cpp GGUF on a discrete CUDA host and estimate against
  combined RAM + VRAM. Emits `--cpu-moe` for MoE models plus auto-fit GPU layers.
- `mmap`: experimental SSD demand paging. Never chosen by `auto`.

`--allow-oversize`

Required acknowledgment for `--memory-mode mmap`. Experimental plans stay marked
`fits=false` and `experimental=true`; presets show an `[EXPERIMENTAL]` marker and
`serve` reprints the warning at launch.

```bash
# Supported hybrid MoE offload on a GPU workstation
rondine plan glm-5.2 --memory-mode hybrid --context 4096 --save-as glm-hybrid

# Explicit experimental SSD paging (not selected automatically)
rondine plan glm-5.2 --quant UD-IQ1_S --context 4096 \
  --memory-mode mmap --allow-oversize --save-as glm-ssd
rondine pull glm-5.2 --memory-mode mmap --allow-oversize
rondine serve --preset glm-ssd
```

`pull` refuses a download when free disk is below the estimated shard total.
`mmap` is OS page-cache demand paging, not the unmerged llama.cpp `--moe-stream*`
expert streamer. See [engine tuning](engine-tuning.md) for capacity notes and
Apple Silicon Metal/mmap hazards.
