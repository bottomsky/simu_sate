# J2 Perturbation Orbit Propagator

## Overview
This project implements high-performance orbit propagators for satellite trajectory simulation in C++, with support for both single satellites and large-scale constellations (up to 20,000+ satellites).

## Features
- **Memory Optimized**: Separated epoch time from orbital elements, reducing memory usage by ~14%
- **SIMD Acceleration**: AVX2 vectorization for processing 4 satellites simultaneously  
- **GPU Computing**: CUDA kernels for massive parallel processing
- **Multiple Precision**: Configurable precision vs performance tradeoffs
- **Scalable Architecture**: Structure-of-Arrays (SoA) for optimal cache performance

## Project Structure
- `include/`: Header files
  - `j2_orbit_propagator.h`: Single satellite propagator
  - `constellation_propagator.h`: Large-scale constellation propagator
- `src/`: Source files  
  - `j2_orbit_propagator.cpp`: Original implementation
  - `constellation_propagator.cpp`: Optimized constellation implementation
  - `constellation_demo.cpp`: Performance testing and demonstration
  - `j2_cuda_kernels.cu`: CUDA GPU acceleration kernels
- `build/`: Build configuration
- `lib/eigen/`: Eigen linear algebra library

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

### Build Steps
```bash
cd build
cmake ..
make # or cmake --build . on Windows
```

### Build Targets
- `j2_propagator`: Original single satellite propagator
- `constellation_propagator`: CPU-optimized constellation propagator  
- `constellation_propagator_cuda`: GPU-accelerated version (if CUDA available)

## Usage

### Single Satellite Propagation
```bash
./j2_propagator
```

### Large Constellation Testing
```bash
./constellation_propagator
```

### GPU Acceleration (if available)
```bash
./constellation_propagator_cuda
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

## License
MIT License