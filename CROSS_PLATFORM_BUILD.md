# Cross-Platform Build Guide

This guide explains how to build the J2 Perturbation Orbit Propagator across Windows, Linux, and macOS, including native CMake builds, Docker-based builds, and .NET interop.

## Prerequisites
- C++17-compatible compiler: MSVC 2019+, GCC 8+, or Clang 7+
- CMake 3.10+
- Eigen dependency is vendored under `lib/eigen`
- Optional: CUDA Toolkit 11.0+ for GPU acceleration
- Optional: Docker (Linux, Windows containers)

## Directory Layout
```
bin/                 # Unified output for all build artifacts
build/               # CMake build tree (cache, intermediates)
src/, include/       # C++ source and headers
scripts/             # Build automation scripts (PowerShell)
docker/              # Containerized build support
tests/               # Unit, integration, performance tests
example/csharp/      # .NET bindings and examples
```

## Unified PowerShell Build Script
Use `scripts/build.ps1` for a consistent native build experience on Windows (and PowerShell Core on Linux/macOS):

```powershell
# Default build (Release)
./scripts/build.ps1

# Clean build cache (preserves build/CMakeLists.txt)
./scripts/build.ps1 -Clean -Config Release

# Force CMake reconfiguration
./scripts/build.ps1 -Reconfigure -Config Debug

# Enable CUDA explicitly
./scripts/build.ps1 -EnableCuda -Config Release

# Specify generator
./scripts/build.ps1 -Generator "Ninja"
```

Parameters:
- `-Config` (Release|Debug|RelWithDebInfo|MinSizeRel)
- `-Clean` cleans build directory but preserves `build/CMakeLists.txt`
- `-Reconfigure` removes `CMakeCache.txt` and `CMakeFiles/` to force a fresh configure
- `-EnableCuda` toggles CUDA build
- `-Generator` specifies CMake generator (e.g., Ninja, Visual Studio)

Output policy:
- All dynamic/static libraries and executables are placed in the repository-level `bin/` folder.
- The script ensures `bin/` exists.

## Manual CMake Build

```bash
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . --config Release
```

Artifacts appear under `../bin/`.

## Docker Builds

Recommended to produce clean Linux/Windows artifacts without polluting your host environment.

```bash
# Linux container build (outputs to ./bin via volume mount)
./docker/build.sh --target linux --build-type Release

# Windows container build (PowerShell)
./docker/build.ps1 -Target windows -BuildType Release

# Multi-target via docker-compose
docker-compose up
```

Notes:
- Container builds also publish artifacts into host `./bin`.
- See `docker/README-动态库路径说明.md` for shared library naming and paths.

## Test Execution

Use the unified test runner script:

```powershell
# Run all tests
./tests/build_and_run_tests.ps1

# With CUDA
./tests/build_and_run_tests.ps1 -EnableCuda

# Specific targets and verbosity
./tests/build_and_run_tests.ps1 -TestTargets "unit,integration" -Verbose
```

Test binaries and any generated outputs are placed under `bin/`.

## .NET / C# Integration

The C# script orchestrates native and managed builds:

```powershell
# Release build
./example/csharp/build_and_test_csharp.ps1 -BuildType Release

# Clean native cache and force reconfigure
./example/csharp/build_and_test_csharp.ps1 -CleanNative -NativeReconfigure

# Override native configuration
./example/csharp/build_and_test_csharp.ps1 -NativeConfig Debug
```

Behavior:
1. Invokes `scripts/build.ps1` to build native libs into `bin/`.
2. Builds `J2Orbit.Library` and `MemoryLayoutTest`.
3. Copies native libraries from `bin/` into test output folders to satisfy runtime loading.

## Cleaning Strategy

- `-Clean` in `scripts/build.ps1` removes build intermediates but preserves `build/CMakeLists.txt`.
- This provides fast recovery without losing the generated CMakeLists if your workflow depends on it.
- `-Reconfigure` wipes CMake cache files to ensure a fresh configure step.

## Output Artifacts and Names

- Windows: `j2_orbit_propagator.dll`, `j2_orbit_propagator.lib`
- Linux: `libj2_orbit_propagator.so`, `libj2_orbit_propagator.a`
- macOS: `libj2_orbit_propagator.dylib`, `libj2_orbit_propagator.a`

All above are deposited into `bin/` for consistency across build methods.

## Troubleshooting

- If native library fails to load in C#, confirm the DLL/SO/DYLIB is present next to the executable or included in the system search path.
- If switching generators or toolchains, prefer `-Clean -Reconfigure` to avoid stale cache issues.
- CUDA builds require compatible GPU drivers; ensure `nvcc`/toolkit is installed and discoverable.