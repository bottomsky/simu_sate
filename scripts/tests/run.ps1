param(
  [switch]$PersistOutput,
  [string]$Filter,
  [string]$GTestFilter,
  [switch]$Rebuild
)
$ErrorActionPreference = 'Stop'
$root = (Resolve-Path "$PSScriptRoot/../..").Path
$build = Join-Path $root 'build'
if ($PersistOutput) {
  $env:TEST_OUTPUT_DIR = Join-Path $root 'simu_sate/tests/data'
  New-Item -ItemType Directory -Force -Path $env:TEST_OUTPUT_DIR | Out-Null
}
# 1) Python tests
Push-Location $root
if ($Filter) { pytest -q -k $Filter } else { pytest -q }
Pop-Location
# 2) C++ tests
New-Item -ItemType Directory -Force -Path $build | Out-Null
if ($Rebuild -or -not (Test-Path (Join-Path $build 'CMakeCache.txt'))) {
  cmake -S (Join-Path $root 'simu_sate') -B $build -DCMAKE_BUILD_TYPE=Release
}
cmake --build $build --target unit_tests
ctest --test-dir $build --output-on-failure -R unit_tests