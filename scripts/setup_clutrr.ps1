$ErrorActionPreference = "Stop"

$ClutrrDir = ".data-sources/clutrr-v1.3"
$ClutrrUrl = "org-16943930@github.com:facebookresearch/clutrr.git"
$Commit = "4f65217"

if ((Test-Path $ClutrrDir) -and (Test-Path "$ClutrrDir/.git")) {
    Write-Host "CLUTRR already cloned, skipping clone."
} else {
    Write-Host "Cloning CLUTRR into $ClutrrDir ..."
    if (Test-Path $ClutrrDir) { Remove-Item -Recurse -Force $ClutrrDir }
    New-Item -ItemType Directory -Force -Path ".data-sources" | Out-Null
    git clone --recurse-submodules $ClutrrUrl $ClutrrDir
}

Set-Location $ClutrrDir
git fetch origin
git checkout $Commit
git submodule update --init --recursive
Write-Host "CLUTRR is ready at $(Get-Location) (commit $Commit)."