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
build/<config>/      # Primary build artifacts (build/Debug, build/Release, etc.)
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
- Primary build artifacts are located in `build/<Config>/` directories (e.g., `build/Release/`, `build/Debug/`).
- 所有构建产物统一输出到 `build/<配置>/` 目录，确保跨平台一致性。

## Manual CMake Build

```bash
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . --config Release
```

Artifacts appear under `build/<Config>/`.

## Docker Builds

Recommended to produce clean Linux/Windows artifacts without polluting your host environment.

```bash
# Linux container build (outputs to ./build/Release via volume mount)
./scripts/docker/build.sh --target linux --build-type Release

# Windows container build (PowerShell)
./scripts/docker/build.ps1 -Target windows -BuildType Release

# Multi-target via docker-compose
docker-compose up
```

Notes:
- Container builds publish artifacts into host `./build/Release/`.
- See `docker/README-动态库路径说明.md` for shared library naming and paths.

## Test Execution

Use the unified test runner script:

```powershell
# Run all tests
./scripts/tests/build_and_run_tests.ps1

# With CUDA
./scripts/tests/build_and_run_tests.ps1 -EnableCuda

# Specific targets and verbosity
./scripts/tests/build_and_run_tests.ps1 -TestTargets "unit,integration" -Verbose
```

Test binaries and any generated outputs are placed under `build/<Config>/`.

## .NET / C# Integration

The C# script orchestrates native and managed builds:

```powershell
# Release build
./scripts/example/csharp/build_and_test_csharp.ps1 -BuildType Release

# Clean native cache and force reconfigure
./scripts/example/csharp/build_and_test_csharp.ps1 -CleanNative -NativeReconfigure

# Override native configuration
./scripts/example/csharp/build_and_test_csharp.ps1 -NativeConfig Debug

# Select which C# app to run (default: MemoryLayoutTest)
./scripts/example/csharp/build_and_test_csharp.ps1 -Run MemoryLayoutTest
./scripts/example/csharp/build_and_test_csharp.ps1 -Run TestApp
./scripts/example/csharp/build_and_test_csharp.ps1 -Run TestProject
./scripts/example/csharp/build_and_test_csharp.ps1 -Run None   # build only, do not run
```

Behavior:
1. Invokes `scripts/build.ps1` to build native libs into `build/<Config>/`.
2. Builds `J2Orbit.Library` and, based on `-Run`, optionally builds one of `MemoryLayoutTest`, `TestApp`, or `TestProject`.
3. If `-Run` is not `None`, copies native libraries from `build/<Config>/` into the selected app's output directory and runs it (default target is `MemoryLayoutTest`).

## Cleaning Strategy

- `-Clean` in `scripts/build.ps1` removes build intermediates but preserves `build/CMakeLists.txt`.
- This provides fast recovery without losing the generated CMakeLists if your workflow depends on it.
- `-Reconfigure` wipes CMake cache files to ensure a fresh configure step.

## Output Artifacts and Names

- Windows: `j2_orbit_propagator.dll`, `j2_orbit_propagator.lib`
- Linux: `libj2_orbit_propagator.so`, `libj2_orbit_propagator.a`
- macOS: `libj2_orbit_propagator.dylib`, `libj2_orbit_propagator.a`

All above are deposited into `build/<Config>/` for consistency across build methods.

## Troubleshooting

- If native library fails to load in C#, confirm the DLL/SO/DYLIB is present next to the executable or included in the system search path.
- If switching generators or toolchains, prefer `-Clean -Reconfigure` to avoid stale cache issues.
- CUDA builds require compatible GPU drivers; ensure `nvcc`/toolkit is installed and discoverable.

<a id="validation-checklist"></a>
## 验证清单（快速自检）

以下命令默认在仓库根目录的 PowerShell 中执行（Windows 环境）。每条命令后列出关键"期望输出"片段，出现即视为通过。

- C# 默认（不带 -Run，默认运行 MemoryLayoutTest）
  - 命令：
    ```powershell
    ./example/csharp/build_and_test_csharp.ps1
    ```
  - 期望输出片段：
    ```
    All tests passed successfully!
    ```

- 指定运行 MemoryLayoutTest
  - 命令：
    ```powershell
    ./example/csharp/build_and_test_csharp.ps1 -Run MemoryLayoutTest
    ```
  - 期望输出片段：
    ```
    All tests passed successfully!
    ```

- 指定运行 TestApp
  - 命令：
    ```powershell
    ./example/csharp/build_and_test_csharp.ps1 -Run TestApp
    ```
  - 期望输出片段：
    ```
    [TestApp] All checks passed.
    ```

- 指定运行 TestProject
  - 命令：
    ```powershell
    ./example/csharp/build_and_test_csharp.ps1 -Run TestProject
    ```
  - 期望输出片段：
    ```
    Hello, World!
    ```

- 仅构建，不运行 C# 应用
  - 命令：
    ```powershell
    ./example/csharp/build_and_test_csharp.ps1 -Run None
    ```
  - 期望输出片段：
    ```
    Skipped running any C# app (-Run None)
    ```
  - 进一步检查（可选）：构建产物位于 build/<配置> 目录（例如 Release）：
    ```powershell
    Test-Path -LiteralPath ./build/Release/j2_orbit_propagator.dll
    ```
    期望：返回 True。

- 清理并重新配置原生构建（Debug）且不运行 C# 应用
  - 命令：
    ```powershell
    ./example/csharp/build_and_test_csharp.ps1 -CleanNative -NativeReconfigure -NativeConfig Debug -Run None
    ```
  - 期望输出片段：
    ```
    Skipped running any C# app (-Run None)
    ```

- 运行过的 C# 应用输出目录包含原生库（自动复制）
  - 命令：
    ```powershell
    $paths = @(
      'example/csharp/J2Orbit.TestApp/build/Release/net8.0/j2_orbit_propagator.dll',
'example/csharp/MemoryLayoutTest/build/Release/net8.0/j2_orbit_propagator.dll',
'example/csharp/TestProject/build/Release/net9.0/j2_orbit_propagator.dll'
    )
    $paths | ForEach-Object { "$_ => " + (Test-Path -LiteralPath $_) }
    ```
  - 期望输出片段：三条路径均为：
    ```
    => True
    ```

- C++ 统一构建脚本（主产物在 build/<配置>，支持清理与重配）
  - 命令：
    ```powershell
    ./scripts/build.ps1 -Clean -Reconfigure -Config Release
    ```
  - 期望输出片段（收尾）：
    ```
    Build completed. Primary artifacts: ...\build\Release
    ```