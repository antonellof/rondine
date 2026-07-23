# Rondine Windows bootstrapper.
# Installs Rondine inside WSL2 (native Windows engines are not supported yet).
#
# Usage:
#   irm https://rondine.dev/install.ps1 | iex
#   irm https://rondine.dev/install.ps1 | iex ; Install-Rondine -Version 0.1.0
#
# Environment:
#   RONDINE_VERSION  Pin a release without the v prefix (e.g. 0.1.0)
#   RONDINE_REPO     GitHub owner/name (default: antonellof/rondine)

[CmdletBinding()]
param(
    [string]$Version = $env:RONDINE_VERSION,
    [string]$Repo = $(if ($env:RONDINE_REPO) { $env:RONDINE_REPO } else { "antonellof/rondine" }),
    [string]$InstallUrl = "https://rondine.dev/install.sh"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Write-Info([string]$Message) {
    Write-Host $Message
}

function Write-Err([string]$Message) {
    Write-Error $Message
}

function Test-WslAvailable {
    if (-not (Get-Command wsl.exe -ErrorAction SilentlyContinue)) {
        return $false
    }
    try {
        $null = & wsl.exe --status 2>$null
        return ($LASTEXITCODE -eq 0)
    } catch {
        return $false
    }
}

function Install-Rondine {
    [CmdletBinding()]
    param(
        [string]$Version = $env:RONDINE_VERSION,
        [string]$Repo = $(if ($env:RONDINE_REPO) { $env:RONDINE_REPO } else { "antonellof/rondine" }),
        [string]$InstallUrl = "https://rondine.dev/install.sh"
    )

    Write-Info "Rondine Windows installer (WSL2 required)"
    Write-Info "Native Windows engines are not supported yet; Rondine runs inside WSL."

    if (-not (Test-WslAvailable)) {
        Write-Err @"
WSL2 was not found.

Install WSL2 first, then rerun this script:
  wsl --install
  # reboot if prompted, open your Linux distro, then:
  irm https://rondine.dev/install.ps1 | iex

Docs: https://rondine.dev/#windows-wsl
"@
        return
    }

    $envArgs = @()
    if ($Repo) {
        $envArgs += "RONDINE_REPO=$Repo"
    }
    if ($Version) {
        $clean = $Version.TrimStart('v', 'V')
        $envArgs += "RONDINE_VERSION=$clean"
    }

    $prefix = if ($envArgs.Count -gt 0) { ($envArgs -join " ") + " " } else { "" }
    $bashCmd = "${prefix}curl -LsSf `"$InstallUrl`" | sh"
    Write-Info "Running inside WSL:"
    Write-Info "  $bashCmd"
    & wsl.exe -e bash -lc $bashCmd
    if ($LASTEXITCODE -ne 0) {
        Write-Err "WSL install failed with exit code $LASTEXITCODE"
        return
    }

    Write-Info ""
    Write-Info "Rondine is installed inside your default WSL distro."
    Write-Info "Open WSL and run:"
    Write-Info "  rondine doctor"
    Write-Info "  rondine"
    Write-Info ""
    Write-Info "Docs: https://rondine.dev"
}

# When piped via `irm ... | iex`, invoke immediately.
# When dot-sourced or imported, only define Install-Rondine.
$isDotSourced = $MyInvocation.InvocationName -eq '.' -or $MyInvocation.Line -match '^\.'
if (-not $isDotSourced) {
    Install-Rondine -Version $Version -Repo $Repo -InstallUrl $InstallUrl
}
