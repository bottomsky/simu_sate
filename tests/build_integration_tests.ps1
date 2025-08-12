param(
  [string]$Config = "Release",
  [string]$BuildDir = "",
  [string]$Generator = "",
  [switch]$Run,
  [Alias('jt')]
  [switch]$JustTest,
  [string]$Target = "integration_tests"
)

$ErrorActionPreference = "Stop"

# 脚本所在目录（tests）
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path

# 根据 Target 推断默认构建目录
if (-not $BuildDir -or $BuildDir -eq "") {
  if ($Target -match '^unit') {
    $BuildDir = Join-Path $scriptDir "build-unit"
  }
  elseif ($Target -match '^integration') {
    $BuildDir = Join-Path $scriptDir "build-integration"
  }
  else {
    $BuildDir = Join-Path $scriptDir "build"
  }
}

Write-Host "[INFO] Using build directory: $BuildDir"
if (-not (Test-Path -Path $BuildDir)) {
  if ($JustTest.IsPresent) {
    Write-Error "[ERROR] Build directory '$BuildDir' does not exist. Run a build first."
    exit 1
  }
  New-Item -ItemType Directory -Force -Path $BuildDir | Out-Null
}

# 生成 CMake 配置参数（以 tests 为顶层）
$cmakeArgs = @("-S", $scriptDir, "-B", $BuildDir, "-DBUILD_TESTING=ON")
if ($Generator -and $Generator.Trim().Length -gt 0) {
  $cmakeArgs = @("-G", $Generator) + $cmakeArgs
}

try {
  if (-not $JustTest.IsPresent) {
    $what = if ($Target -match '^unit') { 'unit tests' } elseif ($Target -match '^integration') { 'integration tests' } else { "tests for target '$Target'" }
    Write-Host "[STEP] Configure CMake for $what..." -ForegroundColor Cyan
    & cmake @cmakeArgs

    Write-Host "[STEP] Build target: $Target ($Config)..." -ForegroundColor Cyan
    $buildArgs = @("--build", $BuildDir, "--config", $Config, "--target", $Target)
    & cmake @buildArgs
  }

  if ($Run.IsPresent -or $JustTest.IsPresent) {
    Write-Host "[STEP] Run tests via ctest (pattern: $Target)..." -ForegroundColor Cyan
    & ctest "--test-dir" $BuildDir "-C" $Config "-R" $Target "--output-on-failure"

    # 如果运行的是单元测试，则列出生成的数据文件
    if ($Target -eq "unit_tests") {
      Write-Host "[INFO] Unit test execution completed. Check test data files:" -ForegroundColor Yellow
      $dataDir = Join-Path (Split-Path -Parent $scriptDir) "tests\data"
      if (Test-Path $dataDir) {
        $files = Get-ChildItem -Path $dataDir -Filter "multi_simulation_results_step_*.json" -ErrorAction SilentlyContinue
        if ($files) {
          $files | ForEach-Object { Write-Host "  - $($_.FullName)" -ForegroundColor Gray }
        } else {
          Write-Host "  (no multi_simulation_results_step_*.json files found)" -ForegroundColor DarkGray
        }
      } else {
        Write-Host "  (data directory not found: $dataDir)" -ForegroundColor DarkGray
      }
    }
  }

  Write-Host "[DONE] Script completed successfully." -ForegroundColor Green
}
catch {
  Write-Error "[ERROR] Failed to build or run tests: $_"
  exit 1
}