param(
  [Alias('c')]
  [string]$Config = "Release",

  [Alias('b','bd')]
  [string]$BuildDir = "",

  [Alias('g')]
  [string]$Generator = "",

  [Alias('r')]
  [switch]$Run,

  [Alias('jt','j')]
  [switch]$JustTest,

  [Alias('t')]
  [string]$Target = "integration_tests",

  # 便捷开关：快速指定 Target
  [Alias('u')]
  [switch]$Unit,

  [Alias('it')]
  [switch]$Integration,

  [Alias('p')]
  [switch]$Performance,

  [Alias('all')]
  [switch]$AllTests
)

$ErrorActionPreference = "Stop"

# 脚本所在目录（tests）
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path

# 如果指定了便捷开关，覆盖 Target
if ($Unit.IsPresent) { $Target = "unit_tests" }
elseif ($Integration.IsPresent) { $Target = "integration_tests" }
elseif ($Performance.IsPresent) { $Target = "performance_tests" }
elseif ($AllTests.IsPresent) { $Target = "all" }

# 根据 Target 推断默认构建目录
if (-not $BuildDir -or $BuildDir -eq "") {
  if ($Target -match '^unit') {
    $BuildDir = Join-Path $scriptDir "build-unit"
  }
  elseif ($Target -match '^integration') {
    $BuildDir = Join-Path $scriptDir "build-integration"
  }
  elseif ($Target -match '^performance') {
    $BuildDir = Join-Path $scriptDir "build-performance"
  }
  elseif ($Target -eq "all") {
    $BuildDir = Join-Path $scriptDir "build-all"
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
    if ($Target -eq "all") {
      $what = 'all tests'
    } else {
      $what = if ($Target -match '^unit') { 'unit tests' } elseif ($Target -match '^integration') { 'integration tests' } elseif ($Target -match '^performance') { 'performance tests' } else { "tests for target '$Target'" }
    }
    Write-Host "[STEP] Configure CMake for $what..." -ForegroundColor Cyan
    & cmake @cmakeArgs

    if ($Target -eq "all") {
      Write-Host "[STEP] Build all test targets ($Config)..." -ForegroundColor Cyan
      $targets = @("unit_tests", "integration_tests", "performance_tests")
      foreach ($t in $targets) {
        Write-Host "  - Building $t" -ForegroundColor DarkCyan
        $buildArgs = @("--build", $BuildDir, "--config", $Config, "--target", $t)
        & cmake @buildArgs
      }
    } else {
      Write-Host "[STEP] Build target: $Target ($Config)..." -ForegroundColor Cyan
      $buildArgs = @("--build", $BuildDir, "--config", $Config, "--target", $Target)
      & cmake @buildArgs
    }
  }

  if ($Run.IsPresent -or $JustTest.IsPresent) {
    if ($Target -eq "all") {
      Write-Host "[STEP] Run all tests via ctest..." -ForegroundColor Cyan
      & ctest "--test-dir" $BuildDir "-C" $Config "--output-on-failure"
    } else {
      Write-Host "[STEP] Run tests via ctest (pattern: $Target)..." -ForegroundColor Cyan
      & ctest "--test-dir" $BuildDir "-C" $Config "-R" $Target "--output-on-failure"
    }

    if ($Target -eq "unit_tests" -or $Target -eq "all") {
      Write-Host "[INFO] Unit test execution completed. Check test data files:" -ForegroundColor Yellow
      $dataDir = Join-Path (Split-Path -Parent $scriptDir) "tests\data"
      if (Test-Path $dataDir) {
        $files = Get-ChildItem -Path $dataDir -Filter "multi_simulation_results_step_*.json" -ErrorAction SilentlyContinue
        if ($files) {
          $files | ForEach-Object { Write-Host "  - $($_.FullName)" -ForegroundColor Gray }
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