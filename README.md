# J2 Perturbation Orbit Propagator

## Overview
This project implements high-performance orbit propagators for satellite trajectory simulation in C++, with support for both single satellites and large-scale constellations (up to 20,000+ satellites). The project provides cross-platform support for Windows, Linux, and macOS, with optional GPU acceleration via CUDA.

## Features
- **Memory Optimized**: Separated epoch time from orbital elements, reducing memory usage by ~14%
- **SIMD Acceleration**: AVX2 vectorization for processing 4 satellites simultaneously  
- **GPU Computing**: CUDA kernels for massive parallel processing
- **Cross-Platform**: Native support for Windows, Linux, and macOS
- **Multiple Precision**: Configurable precision vs performance tradeoffs
- **Scalable Architecture**: Structure-of-Arrays (SoA) for optimal cache performance
- **Language Bindings**: C and C# interfaces for easy integration
- **Docker Support**: Containerized builds for consistent environments

## Project Structure
```
j2-perturbation-orbit-propagator/
├── bin/                        # Build output directory (all compiled artifacts)
│   ├── *.dll/*.so/*.dylib         # Dynamic libraries
│   ├── *.lib/*.a                  # Static libraries
│   └── *.exe                      # Executable files
├── include/                    # Header files
│   ├── j2_orbit_propagator.h      # Single satellite propagator
│   ├── constellation_propagator.h  # Large-scale constellation propagator
│   └── j2_cuda_kernels.h          # CUDA kernel declarations
├── src/                        # Source files
│   ├── j2_orbit_propagator.cpp     # Original implementation
│   ├── constellation_propagator.cpp # Optimized constellation implementation
│   ├── constellation_demo.cpp      # Performance testing and demonstration
│   └── j2_cuda_kernels.cu          # CUDA GPU acceleration kernels
├── tests/                      # Test suite
│   ├── unit/                      # Unit tests
│   ├── integration/               # Integration tests
│   ├── performance/               # Performance benchmarks
│   └── build_and_run_tests.ps1    # Test execution script
├── example/                    # Usage examples
│   ├── c_example.c                # C language example
│   ├── csharp/                    # C# .NET bindings and examples
│   │   ├── J2Orbit.Library/        # .NET wrapper library
│   │   ├── J2Orbit.TestApp/        # Simple C# demo
│   │   ├── MemoryLayoutTest/       # Memory layout tests
│   │   └── build_and_test_csharp.ps1 # C# build and test script
│   └── python/                    # Python bindings (future)
├── scripts/                    # Build automation scripts
│   └── build.ps1                  # Unified C++ build script (with clean/reconfigure support)
├── docker/                     # Docker build support
│   ├── Dockerfile.linux           # Linux build environment
│   ├── Dockerfile.windows         # Windows build environment
│   ├── build.sh                   # Docker build script (Linux/macOS)
│   └── build.ps1                  # Docker build script (Windows)
├── supabase/migrations/        # Database migrations (if applicable)
├── lib/eigen/                  # Eigen linear algebra library
├── build_dynamic_library.ps1   # Legacy Windows build script
├── build_dynamic_library.sh    # Legacy Linux/macOS build script
├── docker-compose.yml          # Multi-platform Docker builds
└── CROSS_PLATFORM_BUILD.md     # Detailed cross-platform build guide
```

## Performance Comparison

### Memory Usage (20,000 satellites)
- **Original**: 1.12 MB (with epoch time per satellite)  
- **Optimized**: 0.96 MB (shared epoch time)
- **Savings**: 14.3% memory reduction

### Computational Performance (estimated)
- **CPU Scalar**: 1x baseline
- **CPU SIMD (AVX2)**: 3-4x speedup
- **GPU CUDA**: 10-50x speedup (depending on constellation size)

## Building the Project

### Prerequisites
- C++17 compatible compiler (MSVC 2019+, GCC 8+, Clang 7+)
- CMake 3.10+
- Eigen3 library (included)
- Optional: CUDA Toolkit 11.0+ for GPU acceleration
- Optional: Docker for containerized builds

### Quick Start - C++ Native Build

#### Unified Build Script (Recommended)
The project provides a unified build script with advanced cleaning and configuration options:

```powershell
# Build with default settings (Release mode)
.\scripts\build.ps1

# Build with cleaning build cache (preserves build/CMakeLists.txt)
.\scripts\build.ps1 -Clean -Config Release

# Build with full reconfiguration (removes CMakeCache.txt)
.\scripts\build.ps1 -Clean -Config Release -Reconfigure

# Debug build with CUDA enabled
.\scripts\build.ps1 -Config Debug -EnableCuda
```

**Available Parameters:**
- `-Config`: Build configuration (Release, Debug, RelWithDebInfo, MinSizeRel)
- `-Clean`: Clean build directory while preserving `build/CMakeLists.txt`
- `-Reconfigure`: Force CMake reconfiguration (removes CMakeCache.txt and CMakeFiles)
- `-EnableCuda`: Force enable CUDA support
- `-Generator`: Specify CMake generator (default: Visual Studio 17 2022)

#### Legacy Build Scripts
```powershell
# Windows (PowerShell) - Legacy
.\scripts\build_dynamic_library.ps1 -BuildType Release -Generator "Visual Studio 17 2022" -EnableCuda

# Linux/macOS (Bash) - Legacy
./scripts/build_dynamic_library.sh --build-type Release --generator Ninja --enable-cuda
```

#### Docker Build (Recommended for Cross-Platform)
```bash
# Build for Linux
./scripts/docker/build.sh --target linux --build-type Release

# Build for Windows (requires Windows containers)
./scripts/docker/build.ps1 -Target windows -BuildType Release

# Build for both platforms using Docker Compose
docker-compose up
```

### C# Example and Test

The C# example provides an integrated workflow that automatically handles native library building:

```powershell
# Build C++ native library and run C# tests (Release mode)
.\scripts\example\csharp\build_and_test_csharp.ps1 -BuildType Release

# Clean native build and reconfigure before running C# tests
.\scripts\example\csharp\build_and_test_csharp.ps1 -BuildType Release -CleanNative -NativeReconfigure

# Use specific native build configuration
.\scripts\example\csharp\build_and_test_csharp.ps1 -BuildType Release -NativeConfig Debug
```

**Available C# Script Parameters:**
- `-BuildType`: C# build configuration (Release, Debug)
- `-CleanNative`: Clean native build directory (preserves build/CMakeLists.txt)
- `-NativeReconfigure`: Force native CMake reconfiguration
- `-NativeConfig`: Native build configuration (inherits from BuildType if not specified)

The C# script automatically:
1. Calls the unified `scripts/build.ps1` to build native libraries
2. Builds the C# projects (`J2Orbit.Library`, `MemoryLayoutTest`)
3. Copies native libraries from `bin/` to test output directories
4. Runs comprehensive memory layout and functionality tests

### Manual CMake Build
```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . --config Release
```

### Build Targets
- `j2_orbit_propagator_static`: Static library
- `j2_orbit_propagator_shared`: Dynamic library (.dll/.so/.dylib)
- `j2_example`: C example executable
- `unit_tests`: Unit test suite
- `integration_tests`: Integration test suite
- `performance_tests`: Performance benchmarks

For detailed cross-platform build instructions and Docker support, see [CROSS_PLATFORM_BUILD.md](CROSS_PLATFORM_BUILD.md).

### Build Output Directory

All compiled artifacts (dynamic libraries, static libraries, and executables) are unified into the `bin/` directory:

- **Dynamic Libraries**: `j2_orbit_propagator.dll` (Windows), `libj2_orbit_propagator.so` (Linux), `libj2_orbit_propagator.dylib` (macOS)
- **Static Libraries**: `j2_orbit_propagator.lib` (Windows), `libj2_orbit_propagator.a` (Linux/macOS)
- **Executables**: Test programs, examples, and utilities

This unified output structure is consistent across all build methods:
- CMake builds (via CMakeLists.txt configuration)
- Unified build script (`scripts/build.ps1`)
- Legacy build scripts (`build_dynamic_library.ps1`, `build_dynamic_library.sh`)
- Docker builds (via volume mounting to `./bin`)
- C# integrated builds (copies from `bin/` to runtime directories)

The `bin/` directory is automatically created if it doesn't exist during the build process.

### Build Cache Management

The unified build script provides intelligent cache management:

- **`-Clean`**: Removes build artifacts but preserves `build/CMakeLists.txt` for faster subsequent builds
- **`-Reconfigure`**: Performs full CMake reconfiguration by removing `CMakeCache.txt` and `CMakeFiles/`
- **Automatic Detection**: The script automatically detects when reconfiguration is needed

### Running Examples

#### C Example
```bash
# Compile and run (Linux/macOS)
gcc -o c_example example/c_example.c -L./bin -lj2_orbit_propagator_shared -lm
./c_example

# Compile and run (Windows)
cl example/c_example.c /I include /link bin/j2_orbit_propagator_shared.lib
c_example.exe
```

#### C# Example
```bash
# Build and run C# example
cd example/csharp
./build_and_test_csharp.ps1

# Or manually
dotnet build J2Orbit.Library/J2Orbit.Library.csproj -c Release
dotnet run --project J2Orbit.TestApp -c Release
```

## Testing

### Test Suite Overview
The project includes comprehensive testing with three main categories:
- **Unit Tests**: Test individual components and functions
- **Integration Tests**: Test component interactions and workflows
- **Performance Tests**: Benchmark and validate performance characteristics

### Running Tests

#### Quick Test Execution (Windows)
```powershell
# Run all tests with default configuration
.\scripts\tests\build_and_run_tests.ps1

# Run specific test types
.\scripts\tests\build_and_run_tests.ps1 -TestTargets "unit,integration"

# Run with specific build configuration
.\scripts\tests\build_and_run_tests.ps1 -BuildType Release -BuildDir "build"

# Run with CUDA enabled
.\scripts\tests\build_and_run_tests.ps1 -EnableCuda

# Run with verbose output
.\scripts\tests\build_and_run_tests.ps1 -Verbose
```

#### Manual Test Execution
```bash
# Build the project first
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . --config Release

# Run all tests using CTest
ctest --output-on-failure

# Run specific test categories
ctest -R "unit_tests" --output-on-failure
ctest -R "integration_tests" --output-on-failure
ctest -R "performance_tests" --output-on-failure

# Run tests in parallel
ctest -j 4 --output-on-failure
```

### Test Categories

#### Unit Tests (`tests/unit/`)
- **Core Algorithm Tests**: Validate J2 perturbation calculations
- **Memory Layout Tests**: Verify SoA vs AoS performance and correctness
- **SIMD Tests**: Test AVX2 vectorization accuracy
- **CUDA Tests**: Validate GPU kernel correctness (if CUDA available)
- **API Tests**: Test C and C++ interface functions

#### Integration Tests (`tests/integration/`)
- **End-to-End Workflows**: Complete propagation scenarios
- **Multi-Satellite Tests**: Large constellation handling
- **Cross-Platform Tests**: Verify consistent results across platforms
- **Parameter Sweep Tests**: Validate across different orbital parameters
- **Memory Management Tests**: Test resource allocation and cleanup

#### Performance Tests (`tests/performance/`)
- **Scalability Benchmarks**: Performance vs constellation size
- **SIMD vs Scalar Comparison**: Vectorization performance gains
- **GPU vs CPU Comparison**: CUDA acceleration benchmarks
- **Memory Usage Profiling**: Memory efficiency validation
- **Throughput Tests**: Satellites processed per second

### Test Results Interpretation
```
# Example test output
Test project d:/code/j2-perturbation-orbit-propagator/build
    Start 1: unit_tests
1/4 Test #1: unit_tests ..................   Passed    2.34 sec
    Start 2: integration_tests
2/4 Test #2: integration_tests ............   Passed   15.67 sec
    Start 3: integration_param_sweep
3/4 Test #3: integration_param_sweep ......   Passed   25.43 sec
    Start 4: performance_tests
4/4 Test #4: performance_tests ............   Passed   29.78 sec

100% tests passed, 0 tests failed out of 4

Total Test time (real) =   73.22 sec
```

### Continuous Integration
For CI/CD pipelines, use:
```yaml
# Example GitHub Actions step
- name: Run Tests
  run: |
    ./scripts/build_dynamic_library.ps1 -BuildType Release
    ./scripts/tests/build_and_run_tests.ps1 -BuildType Release -Quiet
```

## Algorithm Optimizations

### 1. Memory Layout
- **AoS to SoA**: Changed from Array-of-Structures to Structure-of-Arrays
- **Shared Epoch**: Single epoch time for entire constellation
- **Aligned Memory**: 32-byte alignment for SIMD operations

### 2. SIMD Vectorization  
- **AVX2 Instructions**: Process 4 satellites per instruction
- **Vectorized Math**: Parallel trigonometric and arithmetic operations
- **Batch Processing**: Minimize scalar fallback code

### 3. GPU Computing
- **CUDA Kernels**: Massively parallel J2 propagation
- **Memory Coalescing**: Optimized memory access patterns
- **Occupancy Tuning**: Block size optimization for target GPU

## Performance Recommendations

### For Small Constellations (<1,000 satellites)
- Use `CPU_SCALAR` mode for simplicity
- Memory overhead is negligible

### For Medium Constellations (1,000-10,000 satellites)  
- Use `CPU_SIMD` mode for best CPU performance
- Significant speedup with manageable complexity

### For Large Constellations (>10,000 satellites)
- Use `GPU_CUDA` mode if available
- Consider distributed computing for >100,000 satellites

## C#/.NET Integration

### Overview
The project provides comprehensive .NET bindings for easy integration with C# applications. The bindings support all core functionality including CUDA acceleration detection and cross-platform deployment.

### Project Structure (`example/csharp/`)
- **`J2Orbit.Library/`**: .NET Standard 2.1 wrapper library (P/Invoke bindings)
- **`J2Orbit.TestApp/`**: .NET 8.0 console application (simple demonstration)
- **`MemoryLayoutTest/`**: .NET 8.0 console application (memory layout and regression tests)
- **`build_and_test_csharp.ps1`**: One-click build and test script

### Prerequisites
- .NET SDK 8.0+
- Native dynamic library (built using project build scripts)
- Optional: CUDA Toolkit 11.0+ (for GPU acceleration)

### Quick Start

#### One-Click Build and Test (Recommended)
```powershell
# Navigate to C# example directory
cd example/csharp

# Build native library and run default C# target (MemoryLayoutTest)
./build_and_test_csharp.ps1
```


## 验证清单（快速自检）

为保持文档单一来源，本章节已迁移至 CROSS_PLATFORM_BUILD.md。请参见此处获取最新清单：
- [CROSS_PLATFORM_BUILD.md — 验证清单（快速自检）](./CROSS_PLATFORM_BUILD.md#validation-checklist)
