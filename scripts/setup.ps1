param(
  [switch]$Dev
)

Write-Host "Setting up VEGETA environment..." -ForegroundColor Cyan

# Ensure uv is available (install via pipx if missing)
if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
  Write-Host "Installing uv (pip alternative) via pipx..." -ForegroundColor Yellow
  if (-not (Get-Command pipx -ErrorAction SilentlyContinue)) {
    Write-Host "pipx not found. Installing pipx via pip..." -ForegroundColor Yellow
    python -m pip install --user pipx
    python -m pipx ensurepath | Out-Null
  }
  pipx install uv | Out-Null
}

# Make sure current session can find uv even if PATH changes haven't propagated
$localBin = Join-Path $HOME ".local\bin"
if ($null -eq ($env:PATH -split ';' | ForEach-Object { $_.TrimEnd('\\') } | Where-Object { $_ -ieq $localBin })) {
  try {
    [Environment]::SetEnvironmentVariable('PATH', ($env:PATH + ";" + $localBin), 'User')
  } catch {}
  $env:PATH += ";$localBin"
}

# Resolve uv command (by name or common installation paths)
$uvCmd = Get-Command uv -ErrorAction SilentlyContinue
if (-not $uvCmd) {
  $candidatePaths = @(
    (Join-Path $HOME ".local\bin\uv.exe"),
    (Join-Path $HOME "pipx\venvs\uv\Scripts\uv.exe")
  )
  foreach ($p in $candidatePaths) {
    if (Test-Path $p) { $uvCmd = $p; break }
  }
}
if (-not $uvCmd) {
  throw "uv not found. Please restart PowerShell or add `$HOME\.local\bin to PATH, then re-run scripts/setup.ps1."
}

Write-Host "Creating virtual environment with uv..." -ForegroundColor Cyan
& $uvCmd venv

Write-Host "Installing VEGETA package..." -ForegroundColor Cyan
if ($Dev) {
  & $uvCmd pip install -e .[dev]
} else {
  & $uvCmd pip install -e .
}

Write-Host "All set. Activate venv and run: vegeta decide --domain cyber --cve CVE-2025-XXXX" -ForegroundColor Green


