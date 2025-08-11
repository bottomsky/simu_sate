param(
  [string]$Config = "Debug",
  [string]$BuildDir = "",
  [string]$Generator = "",
  [switch]$Run,
  [string]$Target = "integration_tests"
)

$ErrorActionPreference = "Stop"

# 脚本所在目录（tests）
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path

if (-not $BuildDir -or $BuildDir -eq "") {
  $BuildDir = Join-Path $scriptDir "build-integration"
}

Write-Host "[INFO] Using build directory: $BuildDir"
if (-not (Test-Path -Path $BuildDir)) {
  New-Item -ItemType Directory -Force -Path $BuildDir | Out-Null
}

# 生成 CMake 配置参数
$cmakeArgs = @("-S", $scriptDir, "-B", $BuildDir, "-DBUILD_TESTING=ON")
if ($Generator -and $Generator.Trim().Length -gt 0) {
  $cmakeArgs = @("-G", $Generator) + $cmakeArgs
}

try {
  Write-Host "[STEP] Configure CMake for integration tests..." -ForegroundColor Cyan
  & cmake @cmakeArgs

  Write-Host "[STEP] Build target: $Target ($Config)..." -ForegroundColor Cyan
  $buildArgs = @("--build", $BuildDir, "--config", $Config, "--target", $Target)
  & cmake @buildArgs

  if ($Run.IsPresent) {
    Write-Host "[STEP] Run tests via ctest (pattern: $Target)..." -ForegroundColor Cyan
    & ctest "--test-dir" $BuildDir "-C" $Config "-R" $Target "--output-on-failure"
  }

  Write-Host "[DONE] Integration tests build completed." -ForegroundColor Green
}
catch {
  Write-Error "[ERROR] Failed to build integration tests: $_"
  exit 1
}