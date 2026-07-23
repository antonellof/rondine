#!/usr/bin/env sh
# Rondine installer for macOS, Linux, and Windows (WSL).
# Usage:
#   curl -LsSf https://rondine.dev/install.sh | sh
#   curl -LsSf https://rondine.dev/install.sh | sh -s -- --version 0.1.0
#
# Environment:
#   RONDINE_VERSION   Pin a release without the v prefix (e.g. 0.1.0)
#   RONDINE_REPO      GitHub owner/name (default: antonellof/rondine)
#   RONDINE_NO_MODIFY_PATH  Set to 1 to skip PATH hints for uv

set -eu

REPO="${RONDINE_REPO:-antonellof/rondine}"
APP_NAME="rondine"
MIN_PYTHON="3.11"
UV_INSTALL_URL="https://astral.sh/uv/install.sh"

say() {
  printf '%s\n' "$*"
}

err() {
  printf 'error: %s\n' "$*" >&2
  exit 1
}

need_cmd() {
  command -v "$1" >/dev/null 2>&1 || err "required command not found: $1"
}

download() {
  url="$1"
  dest="$2"
  if command -v curl >/dev/null 2>&1; then
    curl --proto '=https' --tlsv1.2 --fail --silent --show-error --location --output "$dest" "$url"
  elif command -v wget >/dev/null 2>&1; then
    wget --https-only -q -O "$dest" "$url"
  else
    err "need curl or wget to download $url"
  fi
}

download_text() {
  url="$1"
  if command -v curl >/dev/null 2>&1; then
    curl --proto '=https' --tlsv1.2 --fail --silent --show-error --location "$url"
  elif command -v wget >/dev/null 2>&1; then
    wget --https-only -q -O - "$url"
  else
    err "need curl or wget to download $url"
  fi
}

os_name() {
  uname_s="$(uname -s | tr '[:upper:]' '[:lower:]')"
  case "$uname_s" in
    linux*)
      if grep -qi microsoft /proc/version 2>/dev/null || [ -n "${WSL_DISTRO_NAME:-}" ]; then
        printf 'wsl\n'
      else
        printf 'linux\n'
      fi
      ;;
    darwin*) printf 'macos\n' ;;
    msys*|cygwin*|mingw*)
      err "native Windows is not supported; install WSL2 and rerun, or use: irm https://rondine.dev/install.ps1 | iex"
      ;;
    *) err "unsupported OS: $uname_s" ;;
  esac
}

resolve_version() {
  if [ -n "${VERSION_ARG:-}" ]; then
    printf '%s\n' "${VERSION_ARG#v}"
    return
  fi
  if [ -n "${RONDINE_VERSION:-}" ]; then
    printf '%s\n' "${RONDINE_VERSION#v}"
    return
  fi
  api="https://api.github.com/repos/${REPO}/releases/latest"
  json="$(download_text "$api")" || err "failed to query latest release from GitHub"
  version="$(printf '%s\n' "$json" | sed -n 's/.*"tag_name"[[:space:]]*:[[:space:]]*"v\{0,1\}\([^"]*\)".*/\1/p' | head -n 1)"
  [ -n "$version" ] || err "could not parse latest release tag from GitHub"
  printf '%s\n' "$version"
}

sha256_file() {
  file="$1"
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$file" | awk '{print $1}'
  elif command -v shasum >/dev/null 2>&1; then
    shasum -a 256 "$file" | awk '{print $1}'
  else
    err "need sha256sum or shasum to verify downloads"
  fi
}

ensure_uv() {
  if command -v uv >/dev/null 2>&1; then
    return
  fi
  say "uv not found; installing from astral.sh ..."
  need_cmd curl
  # shellcheck disable=SC2312
  curl --proto '=https' --tlsv1.2 --fail --silent --show-error --location "$UV_INSTALL_URL" | sh
  # Common install locations for a fresh uv bootstrap.
  export PATH="${HOME}/.local/bin:${HOME}/.cargo/bin:${PATH}"
  command -v uv >/dev/null 2>&1 || err "uv installed but not on PATH; open a new shell or add ~/.local/bin"
}

print_path_hint() {
  if [ "${RONDINE_NO_MODIFY_PATH:-0}" = "1" ]; then
    return
  fi
  bin_dir="$(uv tool dir --bin 2>/dev/null || true)"
  if [ -z "$bin_dir" ]; then
    bin_dir="${HOME}/.local/bin"
  fi
  case ":${PATH}:" in
    *":${bin_dir}:"*) ;;
    *)
      say ""
      say "Add uv tool binaries to your PATH (current shell):"
      say "  export PATH=\"${bin_dir}:\$PATH\""
      say "Or run: uv tool update-shell"
      ;;
  esac
}

VERSION_ARG=""
while [ "$#" -gt 0 ]; do
  case "$1" in
    --version)
      [ "$#" -ge 2 ] || err "--version requires a value"
      VERSION_ARG="$2"
      shift 2
      ;;
    --help|-h)
      cat <<'EOF'
Install Rondine from the latest (or pinned) GitHub Release.

Usage:
  curl -LsSf https://rondine.dev/install.sh | sh
  curl -LsSf https://rondine.dev/install.sh | sh -s -- --version 0.1.0

Environment:
  RONDINE_VERSION        Pin release (without leading v)
  RONDINE_REPO           GitHub owner/name (default: antonellof/rondine)
  RONDINE_NO_MODIFY_PATH Skip PATH hints when set to 1
EOF
      exit 0
      ;;
    *)
      err "unknown argument: $1"
      ;;
  esac
done

need_cmd uname
os="$(os_name)"
say "Rondine installer (${os})"

version="$(resolve_version)"
tag="v${version}"
wheel="rondine-${version}-py3-none-any.whl"
base="https://github.com/${REPO}/releases/download/${tag}"
tmpdir="$(mktemp -d "${TMPDIR:-/tmp}/rondine-install.XXXXXX")"
trap 'rm -rf "$tmpdir"' EXIT INT HUP TERM

say "Downloading ${tag} ..."
download "${base}/${wheel}" "${tmpdir}/${wheel}"
download "${base}/SHA256SUMS" "${tmpdir}/SHA256SUMS"

expected="$(awk -v f="$wheel" '$2 == f {print $1; exit}' "${tmpdir}/SHA256SUMS")"
[ -n "$expected" ] || err "SHA256SUMS has no entry for ${wheel}"
actual="$(sha256_file "${tmpdir}/${wheel}")"
[ "$expected" = "$actual" ] || err "checksum mismatch for ${wheel} (expected ${expected}, got ${actual})"
say "Checksum OK"

ensure_uv
say "Installing ${APP_NAME} ${version} with uv (Python ${MIN_PYTHON}+) ..."
uv tool install --force --python "${MIN_PYTHON}" "${tmpdir}/${wheel}"

if ! command -v rondine >/dev/null 2>&1; then
  tool_bin="$(uv tool dir --bin 2>/dev/null || echo "${HOME}/.local/bin")"
  PATH="${tool_bin}:${PATH}"
  export PATH
fi

if command -v rondine >/dev/null 2>&1; then
  say ""
  say "Installed: $(command -v rondine)"
  rondine --version || true
else
  say "Installed, but rondine is not yet on PATH."
fi

print_path_hint

say ""
say "Next steps:"
say "  rondine doctor"
say "  rondine"
say ""
say "Docs: https://rondine.dev"
say "Windows users: install via WSL2 (native Windows engines are not supported yet)."
