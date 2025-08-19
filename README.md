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
│   │   └── MemoryLayoutTest/       # Memory layout tests
│   └── python/                    # Python bindings (future)
├── docker/                     # Docker build support
│   ├── Dockerfile.linux           # Linux build environment
│   ├── Dockerfile.windows         # Windows build environment
│   ├── build.sh                   # Docker build script (Linux/macOS)
│   └── build.ps1                  # Docker build script (Windows)
├── supabase/migrations/        # Database migrations (if applicable)
├── lib/eigen/                  # Eigen linear algebra library
├── build_dynamic_library.ps1   # Windows build script
├── build_dynamic_library.sh    # Linux/macOS build script
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

### Quick Start

#### Windows (PowerShell)
```powershell
# Build dynamic library with default settings
.\build_dynamic_library.ps1

# Build with specific configuration
.\build_dynamic_library.ps1 -BuildType Release -Generator "Visual Studio 17 2022" -EnableCuda

# Clean build
.\build_dynamic_library.ps1 -Clean
```

#### Linux/macOS (Bash)
```bash
# Build dynamic library with default settings
./build_dynamic_library.sh

# Build with specific configuration
./build_dynamic_library.sh --build-type Release --generator Ninja --enable-cuda

# Clean build
./build_dynamic_library.sh --clean
```

#### Docker Build (Recommended for Cross-Platform)
```bash
# Build for Linux
./docker/build.sh --target linux --build-type Release

# Build for Windows (requires Windows containers)
./docker/build.ps1 -Target windows -BuildType Release

# Build for both platforms using Docker Compose
docker-compose up
```

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

- **Dynamic Libraries**: `j2_orbit_propagator_shared.dll` (Windows), `libj2_orbit_propagator_shared.so` (Linux), `libj2_orbit_propagator_shared.dylib` (macOS)
- **Static Libraries**: `j2_orbit_propagator_static.lib` (Windows), `libj2_orbit_propagator_static.a` (Linux/macOS)
- **Executables**: Test programs, examples, and utilities

This unified output structure is consistent across all build methods:
- CMake builds (via CMakeLists.txt configuration)
- Local build scripts (`build_dynamic_library.ps1`, `build_dynamic_library.sh`)
- Docker builds (via volume mounting to `./bin`)

The `bin/` directory is automatically created if it doesn't exist during the build process.

## Usage Examples

### C Language Example
```c
#include "j2_orbit_propagator.h"
#include <stdio.h>

int main() {
    // Create constellation propagator with J2000 epoch
    ConstellationPropagatorHandle constellation = constellation_propagator_create(0.0);
    
    // Add a satellite with compact orbital elements
    CCompactOrbitalElements elements = {
        .a = 7000e3,                    // Semi-major axis (m)
        .e = 0.001,                     // Eccentricity
        .i = 98.0 * M_PI / 180.0,      // Inclination (rad)
        .O = 0.0,                       // RAAN (rad)
        .w = 0.0,                       // Argument of perigee (rad)
        .M = 0.0                        // Mean anomaly (rad)
    };
    
    int sat_id = constellation_propagator_add_satellite(constellation, &elements);
    
    // Propagate for 1 hour (3600 seconds)
    constellation_propagator_propagate(constellation, 3600.0);
    
    // Get satellite state
    CSatelliteState state;
    constellation_propagator_get_satellite_state(constellation, sat_id, &state);
    
    printf("Position: [%.3f, %.3f, %.3f] km\n", 
           state.r[0]/1000, state.r[1]/1000, state.r[2]/1000);
    printf("Velocity: [%.3f, %.3f, %.3f] km/s\n", 
           state.v[0]/1000, state.v[1]/1000, state.v[2]/1000);
    
    // Clean up
    constellation_propagator_destroy(constellation);
    return 0;
}
```

### C# Example
```csharp
using J2.Propagator;
using System;

class Program
{
    static void Main()
    {
        // Create constellation propagator with J2000 epoch
        var constellation = new ConstellationPropagator(0.0);
        
        // Check if CUDA is available
        if (ConstellationPropagator.IsCudaAvailable())
        {
            Console.WriteLine("CUDA acceleration is available");
        }
        
        // Add a satellite with compact orbital elements
        var elements = new CCompactOrbitalElements
        {
            a = 7000e3,                           // Semi-major axis (m)
            e = 0.001,                            // Eccentricity
            i = 98.0 * Math.PI / 180.0,          // Inclination (rad)
            O = 0.0,                              // RAAN (rad)
            w = 0.0,                              // Argument of perigee (rad)
            M = 0.0                               // Mean anomaly (rad)
        };
        
        int satId = constellation.AddSatellite(elements);
        
        // Propagate for 1 hour (3600 seconds)
        constellation.Propagate(3600.0);
        
        // Get satellite state
        var state = constellation.GetSatelliteState(satId);
        
        Console.WriteLine($"Position: [{state.r[0]/1000:F3}, {state.r[1]/1000:F3}, {state.r[2]/1000:F3}] km");
        Console.WriteLine($"Velocity: [{state.v[0]/1000:F3}, {state.v[1]/1000:F3}, {state.v[2]/1000:F3}] km/s");
        
        // Clean up
        constellation.Dispose();
    }
}
```

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
.\tests\build_and_run_tests.ps1

# Run specific test types
.\tests\build_and_run_tests.ps1 -TestTargets "unit,integration"

# Run with specific build configuration
.\tests\build_and_run_tests.ps1 -BuildType Release -BuildDir "build"

# Run with CUDA enabled
.\tests\build_and_run_tests.ps1 -EnableCuda

# Run with verbose output
.\tests\build_and_run_tests.ps1 -Verbose
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
    ./build_dynamic_library.ps1 -BuildType Release
    ./tests/build_and_run_tests.ps1 -BuildType Release -Quiet
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

# Build native library and run all C# tests
./build_and_test_csharp.ps1
```

This script will:
1. Build the native dynamic library (`j2_orbit_propagator.dll`)
2. Build the `J2Orbit.Library` wrapper
3. Run `MemoryLayoutTest` for regression testing
4. Run `J2Orbit.TestApp` for functionality demonstration

#### Manual Build Process

1. **Build Native Library**:
```powershell
# Using project build script (Windows)
./build_dynamic_library.ps1 -BuildType Release

# Or using CMake directly
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . --config Release
```

2. **Build and Run .NET Projects**:
```powershell
# Build wrapper library
dotnet build example/csharp/J2Orbit.Library/J2Orbit.Library.csproj -c Release

# Build and run tests
dotnet build example/csharp/MemoryLayoutTest/MemoryLayoutTest.csproj -c Release
dotnet run --project example/csharp/MemoryLayoutTest -c Release

# Build and run demo application
dotnet build example/csharp/J2Orbit.TestApp/J2Orbit.TestApp.csproj -c Release
dotnet run --project example/csharp/J2Orbit.TestApp -c Release
```

### Cross-Platform Library Loading
The P/Invoke bindings use the base name `j2_orbit_propagator`, which automatically maps to:
- **Windows**: `j2_orbit_propagator.dll`
- **Linux**: `libj2_orbit_propagator.so`
- **macOS**: `libj2_orbit_propagator.dylib`

Ensure the native library is in the executable output directory or system search path (`PATH`).

### Advanced Usage Example
```csharp
using J2.Propagator;
using System;
using System.Collections.Generic;

class AdvancedExample
{
    static void Main()
    {
        // Check CUDA availability
        bool cudaAvailable = ConstellationPropagator.IsCudaAvailable();
        Console.WriteLine($"CUDA Available: {cudaAvailable}");
        
        // Create constellation with automatic CUDA detection
        using var constellation = new ConstellationPropagator(0.0);
        
        // Add multiple satellites
        var elements = new CCompactOrbitalElements
        {
            a = 7000e3,
            e = 0.001,
            i = 98.0 * Math.PI / 180.0,
            O = 0.0,
            w = 0.0,
            M = 0.0
        };
        
        // Add constellation of satellites
        var satelliteIds = new List<int>();
        for (int i = 0; i < 100; i++)
        {
            elements.M = i * 2.0 * Math.PI / 100.0; // Distribute in orbit
            satelliteIds.Add(constellation.AddSatellite(elements));
        }
        
        // Propagate constellation
        constellation.Propagate(3600.0); // 1 hour
        
        // Get states for all satellites
        foreach (int satId in satelliteIds)
        {
            var state = constellation.GetSatelliteState(satId);
            double posMagnitude = Math.Sqrt(state.r[0]*state.r[0] + state.r[1]*state.r[1] + state.r[2]*state.r[2]);
            Console.WriteLine($"Satellite {satId}: Position magnitude = {posMagnitude/1000:F3} km");
        }
    }
}
```

### Troubleshooting

#### Common Issues
- **"Unable to load DLL 'j2_orbit_propagator' (0x8007007E)"**:
  - Verify DLL exists in output directory and matches process architecture (x64)
  - Add DLL directory to `PATH` or place in same directory as executable
  - Ensure all dependencies (like CUDA runtime) are available

- **"Entry point not found"**:
  - Ensure native library is built with latest code
  - Run `./build_and_test_csharp.ps1` for complete rebuild and test

- **CUDA Not Available**:
  - `ConstellationPropagator.IsCudaAvailable()` returning `false` is expected without proper CUDA installation
  - Install CUDA Toolkit 11.0+ and ensure GPU drivers are up to date

#### Performance Tips
- Use `using` statements or explicit `Dispose()` calls for proper resource cleanup
- For large constellations (>1000 satellites), CUDA acceleration provides significant performance benefits
- Consider batch operations when adding multiple satellites

## Additional Documentation

- **[Cross-Platform Build Guide](CROSS_PLATFORM_BUILD.md)**: Comprehensive guide for building on Linux, Windows, and using Docker
- **[Test Script Usage](tests/TEST_SCRIPT_USAGE.md)**: Detailed documentation for the PowerShell test execution script
- **[C Example](example/c_example.c)**: Complete C language usage example
- **[C# Integration](example/csharp/)**: .NET bindings and examples

## License
MIT License