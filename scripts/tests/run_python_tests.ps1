param(
    [string]$TestPattern = "tests/unit/j2_orbit/test_python_binding_minimal.py"
)

<#!
.SYNOPSIS
  在 Windows 上准备 Python 虚拟环境、按需构建 DLL，并运行指定 pytest 用例。

.DESCRIPTION
  - 若未安装 uv，则尝试通过 pip 安装 uv；
  - 若 .venv 不存在，则使用 `python -m uv venv .venv` 创建虚拟环境；
  - 若无 pyproject.toml，则自动初始化最小 pyproject.toml；
  - 使用 `python -m uv add pytest` 安装测试依赖；
  - 若缺少 j2_orbit_propagator.dll，则调用 scripts/build_dynamic_library.ps1 进行 Release 构建（附带 -CleanCache 以解决生成器不匹配）；
  - 最后运行 `./.venv/Scripts/pytest` 执行指定测试。

.PARAMETER TestPattern
  pytest 测试路径或模式，默认运行最小闭环用例。

.NOTES
  需要已安装 Visual Studio 生成工具（或可用的 CMake 生成器）。
!#>

$ErrorActionPreference = 'Stop'
$repoRoot = Resolve-Path (Join-Path $PSScriptRoot '../../')
Set-Location $repoRoot

function Ensure-UvInstalled {
    Write-Host "[INFO] 检查 uv 可用性..."
    $hasUv = $false
    try {
        python -m uv --version | Out-Null
        if ($LASTEXITCODE -eq 0) { $hasUv = $true }
    } catch { $hasUv = $false }
    if (-not $hasUv) {
        Write-Host "[INFO] 未检测到 uv，尝试通过 pip 安装..."
        python -m pip install -U pip | Write-Output
        python -m pip install -U uv | Write-Output
        python -m uv --version | Write-Output
    }
}

function Ensure-Venv {
    if (-not (Test-Path ".venv")) {
        Write-Host "[INFO] 创建虚拟环境 .venv ..."
        python -m uv venv .venv | Write-Output
    } else {
        Write-Host "[INFO] 虚拟环境 .venv 已存在"
    }
}

function Ensure-Project {
    $pyproject = Join-Path $repoRoot 'pyproject.toml'
    if (-not (Test-Path $pyproject)) {
        Write-Host "[INFO] 初始化 pyproject.toml ..."
        @"
[project]
name = "j2-perturbation-orbit-propagator"
version = "0.0.0"
description = "J2 Orbit Propagator Python bindings and tests"
requires-python = ">=3.10"
dependencies = []

[tool.pytest.ini_options]
addopts = "-q"
"@ | Set-Content -Path $pyproject -Encoding UTF8
    } else {
        Write-Host "[INFO] 已存在 pyproject.toml"
    }
}

function Ensure-PyTest {
    Write-Host "[INFO] 安装 pytest ..."
    python -m uv add pytest | Write-Output
}

function Ensure-DLL {
    $dllPath = Join-Path $repoRoot 'build/Release/j2_orbit_propagator.dll'
    if (-not (Test-Path $dllPath)) {
        Write-Host "[INFO] 未发现 DLL，开始构建 Release 动态库..."
        & (Join-Path $repoRoot 'scripts/build_dynamic_library.ps1') -BuildType Release -CleanCache | Write-Output
        if (-not (Test-Path $dllPath)) {
            throw "构建完成后仍未发现 DLL: $dllPath"
        }
    } else {
        Write-Host "[INFO] 已存在 DLL: $dllPath"
    }
}

try {
    Ensure-UvInstalled
    Ensure-Venv
    Ensure-Project
    Ensure-PyTest
    Ensure-DLL

    Write-Host "[INFO] 运行 PyTest: $TestPattern"
    & ".\.venv\Scripts\pytest" $TestPattern -q
    exit $LASTEXITCODE
} catch {
    Write-Error $_
    exit 1
}