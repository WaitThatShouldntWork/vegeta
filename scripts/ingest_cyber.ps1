param(
  [switch]$Upsert,
  [string]$SnapshotLabel = "v$(Get-Date -Format yyyy_MM_dd)"
)

Write-Host "Building mini cyber snapshot..." -ForegroundColor Cyan
vegeta ingest-cyber --mini --snapshot-label $SnapshotLabel $(if ($Upsert) {"--upsert"} else {"--no-upsert"})


