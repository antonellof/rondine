# CLI guide

Run `rondine COMMAND --help` for the option list installed with your version.
This guide explains the choices that affect model recommendations.

## Suggest

`rondine suggest` detects the current machine and ranks curated model, format,
quantization, and engine combinations that fit its available memory.

```bash
rondine suggest --profile coding
```

Each result includes its rank, estimated memory use and headroom, context size,
engine arguments, sampling settings, and an equivalent `serve` command.

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

`--json`

Write the complete result as JSON instead of human-readable output. This is
intended for scripts and includes detected hardware, selected hardware target,
engine order, suggestions, estimates, arguments, and notes.

```bash
rondine suggest --profile coding --json > suggestions.json
```

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
