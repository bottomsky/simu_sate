<#!
.SYNOPSIS
  运行 example/python_binding_example.py 示例。
.DESCRIPTION
  - 确保虚拟环境与 DLL 可用；
  - 运行示例脚本。
!#>

$ErrorActionPreference = 'Stop'
$repoRoot = Resolve-Path (Join-Path $PSScriptRoot '../../')
Set-Location $repoRoot

function Ensure-Uv {
    try { python -m uv --version | Out-Null } catch {
        python -m pip install -U uv | Write-Output
    }
}

function Ensure-Venv {
    if (-not (Test-Path ".venv")) { python -m uv venv .venv | Write-Output }
}

function Ensure-DLL {
    $dllPath = Join-Path $repoRoot 'build/Release/j2_orbit_propagator.dll'
    if (-not (Test-Path $dllPath)) {
        & (Join-Path $repoRoot 'scripts/build_dynamic_library.ps1') -BuildType Release | Write-Output
    }
}

try {
    Ensure-Uv
    Ensure-Venv
    Ensure-DLL

    Write-Host "[INFO] 运行示例: example/python_binding_example.py"
    & ".\.venv\Scripts\python" "example/python_binding_example.py"
    exit $LASTEXITCODE
} catch {
    Write-Error $_
    exit 1
}