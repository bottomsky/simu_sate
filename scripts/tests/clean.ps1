$ErrorActionPreference = 'Stop'
$root = (Resolve-Path "$PSScriptRoot/../..").Path
Remove-Item -Recurse -Force -ErrorAction SilentlyContinue (Join-Path $root 'simu_sate/.pytest_cache')
# 可选：移除持久化输出
# Remove-Item -Recurse -Force -ErrorAction SilentlyContinue (Join-Path $root 'simu_sate/tests/data')
Remove-Item -Recurse -Force -ErrorAction SilentlyContinue (Join-Path $root 'build')