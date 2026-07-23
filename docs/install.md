# Install

Rondine ships as a pure-Python wheel on [GitHub Releases](https://github.com/antonellof/rondine/releases).
The official installers bootstrap [uv](https://docs.astral.sh/uv/) when needed,
download the release assets, verify checksums, and install the `rondine` CLI.

## One-liners

### macOS, Linux, and WSL

```bash
curl -LsSf https://rondine.dev/install.sh | sh
```

### Windows (PowerShell)

Native Windows engine setup is not supported yet. The PowerShell script requires
[WSL2](https://learn.microsoft.com/windows/wsl/install) and runs the same Linux
installer inside your default distro:

```powershell
irm https://rondine.dev/install.ps1 | iex
```

If WSL is missing:

```powershell
wsl --install
# reboot if prompted, open Ubuntu (or your distro), then rerun the installer
```

## Requirements

- macOS (Apple Silicon recommended), Linux, or Windows with WSL2
- Network access to GitHub Releases and astral.sh (for uv bootstrap)
- Python 3.11+ (uv downloads a managed interpreter when needed)

Inference engines (llama.cpp, MLX-LM, vLLM) are installed later with
`rondine setup`, not by the CLI installer. With no `--engine` option, setup
automatically chooses engines supported by the detected platform:

- Linux/WSL without CUDA: llama.cpp
- Linux/WSL with NVIDIA CUDA: llama.cpp and vLLM
- Apple Silicon: llama.cpp and MLX-LM
- Intel macOS: llama.cpp

On Linux, building llama.cpp requires `git`, `cmake`, and a C++ compiler.
Rondine reports a platform-specific install command if no runnable engine is
found, and `rondine serve` refuses to launch until its selected engine is ready.

## Pin a version

```bash
curl -LsSf https://rondine.dev/install.sh | sh -s -- --version 0.1.0
```

```bash
RONDINE_VERSION=0.1.0 curl -LsSf https://rondine.dev/install.sh | sh
```

```powershell
$env:RONDINE_VERSION = "0.1.0"
irm https://rondine.dev/install.ps1 | iex
```

## PATH

uv installs tool executables under its tool bin directory (often
`~/.local/bin`). If `rondine` is not found after install:

```bash
export PATH="$(uv tool dir --bin):$PATH"
# persist for your shell:
uv tool update-shell
```

Then confirm:

```bash
rondine --version
rondine doctor
```

## Update

Re-run the installer to pull the latest release:

```bash
curl -LsSf https://rondine.dev/install.sh | sh
```

Or upgrade with uv after a release is published:

```bash
# download the new wheel from GitHub Releases, then:
uv tool install --force --python 3.11 ./rondine-*-py3-none-any.whl
```

## Uninstall

```bash
uv tool uninstall rondine
```

Engine toolchains and downloaded weights under `~/.rondine/` are left in place.
Remove that directory manually if you want a full cleanup.

## Security and checksums

Every GitHub Release includes:

- `rondine-<version>-py3-none-any.whl`
- `rondine-<version>.tar.gz`
- `SHA256SUMS`

The curl installer downloads the wheel and `SHA256SUMS`, then verifies the
checksum before calling `uv tool install`. Prefer the official
`https://rondine.dev/install.sh` endpoint (or the copy in the repository root)
over unreviewed third-party mirrors.

Manual verification:

```bash
VERSION=0.1.0
BASE="https://github.com/antonellof/rondine/releases/download/v${VERSION}"
curl -fsSL -O "${BASE}/rondine-${VERSION}-py3-none-any.whl"
curl -fsSL -O "${BASE}/SHA256SUMS"
sha256sum -c SHA256SUMS --ignore-missing
```

## Alternative installs

From a clone (development):

```bash
git clone https://github.com/antonellof/rondine.git
cd rondine
uv tool install .
```

From a release asset you already trust:

```bash
uv tool install --force --python 3.11 ./rondine-0.1.0-py3-none-any.whl
```

## Windows / WSL notes

- Run `rondine` **inside** the WSL distro, not from native PowerShell, after
  install.
- GPU passthrough for NVIDIA in WSL follows Microsoft and NVIDIA WSL docs;
  Rondine treats the WSL environment like Linux.
- MLX is Apple Silicon only and will not install under WSL.

## First commands

```bash
rondine doctor
rondine setup
rondine suggest --profile coding
rondine
```

Continue with the [CLI guide](cli.md) and [coding client setup](coding.md).
