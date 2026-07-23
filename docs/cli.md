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
Global options precede the subcommand: `rondine --no-color suggest`, not
`rondine suggest --no-color`.

## Interactive session

Run Rondine without a command to start the full guided interface:

```bash
rondine
```

On first use, the session runs hardware doctor, asks for workload, required
context and model sources, then opens the recommendation selector. On later
runs, Rondine detects the active plan and saved presets first. It offers to
continue the active configuration, load a preset, open the main menu directly,
or start a new guided setup—so hardware detection and recommendation are not
repeated unnecessarily.

The first-run wizard asks for workload (`coding` or `chat`), required context,
and model sources (catalog plus Hub, or offline catalog only) before showing the
arrow-key recommendation selector.

The main menu groups related actions into five focused areas:

- **Run selected model:** preview, download, start, verify or stop
- **Choose or change model:** recommendations, catalog, Hub search or direct plan
- **Environment:** inference-engine setup and hardware doctor
- **Presets:** load, save or list
- **Current configuration:** concise active model and runtime summary

Potentially expensive or state-changing actions such as installing engines,
downloading weights and starting a server require confirmation.
Browsing the catalog or searching the Hub does not replace the active plan;
choose a recommendation or use direct planning to do that. Command failures are
shown as warnings and return to the menu instead of ending the session.

`rondine --help` continues to show the command list. Explicit commands such as
`rondine doctor`, `rondine suggest` and `rondine serve` behave as before. When
standard input is not interactive, bare `rondine` prints help instead of
waiting for input.

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

Cards use three source markers:

- `TOP`: model recommended for the matched hardware target
- `HUB`: live Hugging Face discovery result
- `CAT`: another curated catalog result

On discrete CUDA systems, resident fit and the summary use VRAM rather than
system RAM. Missing engines are reported separately from model fit.

### Profiles

`--profile coding` is the default. It favors models with a higher coding
priority, uses a 32,768-token context by default, enables model-appropriate
reasoning where supported, and tunes engines for one interactive coding client.

`--profile chat` uses a 16,384-token context, disables explicit reasoning for
supported model families, and tunes llama.cpp for more concurrent chat slots.

Profiles affect ranking, context, sampling, reasoning flags, and engine
configuration. They do not change model weights. Model-family settings can
override the general profile defaults; `suggest` prints the resolved values.

Use `--context TOKENS` when the application requires a specific context window.
Rondine recalculates KV-cache memory at that size and excludes catalog models
whose declared maximum context is lower.

```bash
rondine suggest --profile chat
```

### Options

`--profile coding|chat`

Select the workload profile described above. Default: `coding`.

`--limit INTEGER`

Set the maximum number of ranked configurations to display. Default: `5`.
In interactive selection, choose **Show more recommendations** to increase the
same search by five results at a time, up to 50 or until results are exhausted.

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

Rondine reserves one displayed slot for a fitting Hub result when the total
limit is below six, and two slots for larger result sets. API errors are
non-fatal and leave the catalog ranking intact.

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

`--context TOKENS`

Require at least this context window. The value must be at least 1,024 tokens
and directly affects KV-cache estimates and ranking. Catalog models below the
required capability are excluded. Hub results are also excluded when their
inferred catalog family has a lower known limit; unknown Hub families are
marked as not independently verified.

```bash
rondine suggest --context 65536
```

`--json`

Write the complete result as JSON instead of human-readable output. This is
intended for scripts and includes detected hardware, selected hardware target,
engine order, suggestions, estimates, arguments, and notes.

```bash
rondine suggest --profile coding --json > suggestions.json
```

`-i` / `--interactive`

After a normal `suggest`, Rondine offers interactive selection when attached to
a terminal. Use `-i` to open the menu without the confirmation or
`--no-interactive` to suppress the offer. In the menu, use `↑`/`↓` (or `j`/`k`)
to move, Enter to select and configure a model, or `q` to cancel. When input is
not attached to a terminal, explicit `-i` falls back to a numbered prompt.

```bash
rondine suggest --interactive
rondine suggest -i --save-as coding
```

Interactive mode cannot be combined with `--json` or `--configure`.

`--no-interactive`

Suppress the post-results selection offer in an interactive terminal.

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

With `--configure` or interactive selection, also save the selected plan as a
reusable preset. The name is user-defined; `coding` is only an example.

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
`suggest` exits with status 1 when no fitting result is available.

## Command reference

Use `rondine COMMAND --help` for every option.

- `doctor` — inspect RAM/VRAM, disk, platform, engines, and matched target.
- `models` — list curated variants and whether each fits this machine.
- `search QUERY` — search Hugging Face; optionally filter with `--engine`.
- `inspect ORG/REPO` — inspect repository files, sizes, formats, and quants.
- `plan MODEL` — create the active plan from `auto`, a catalog ID, or Hub repo.
- `setup` — install recommended engines; supports `--engine` and `--dry-run`.
- `pull [MODEL]` — download the active or supplied model.
- `serve [MODEL]` — launch the active, supplied, or `--preset` configuration.
- `stop` — stop a managed server by `--name`.
- `verify` — check a managed `--name` or explicit `--base-url`.
- `preset` — `list`, `show`, `save`, `delete`, or `serve` named configurations.
- `cluster` — `init`, `doctor`, `plan`, and dry-run `serve`; see
  [cluster setup](cluster.md).

When `serve` loads an existing active plan or preset, its saved profile and
settings win; use `plan` or the dashboard to change them first. A managed run
name is separate from a preset name, although `--save-as NAME` uses the same
name for both in the common workflow.

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
