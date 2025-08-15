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

## C#/.NET 绑定（example/csharp）

本仓库提供了 .NET 包装库与示例，便于在 C# 中调用原生 J2 轨道传播能力。

- 位置：`example/csharp/`
- 项目：
  - `J2Orbit.Library`：.NET Standard 2.1 类库（P/Invoke 封装）
  - `MemoryLayoutTest`：.NET 8.0 控制台程序（内存布局与功能回归测试）
  - `J2Orbit.TestApp`：.NET 8.0 控制台程序（简单功能演示）

### 先决条件
- .NET SDK 8.0+
- Windows 环境建议安装 Visual C++ 工具链用于构建原生 DLL（或直接使用 `example/csharp/j2_orbit_propagator.dll` 预编译文件）
- 可选：CUDA Toolkit 11.0+（若需 GPU 模式）

### 一键构建与测试（推荐）
在 PowerShell 中执行：
```powershell
# 路径：example/csharp
./build_and_test_csharp.ps1
```
脚本将：
- 构建原生动态库（Windows: `j2_orbit_propagator.dll`）
- 构建 `J2Orbit.Library`、运行 `MemoryLayoutTest` 回归测试
- 运行 `J2Orbit.TestApp` 简单验证

### 手动构建与运行
1) 构建原生库（任选其一）：
```powershell
# 使用仓库脚本（Windows）
./build_dynamic_library.ps1

# 或使用 CMake（示例）
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . --config Release
```
2) 构建并运行 .NET 项目：
```powershell
# 构建包装库
 dotnet build example/csharp/J2Orbit.Library/J2Orbit.Library.csproj -c Release
 
# 构建并运行测试/示例
 dotnet build example/csharp/MemoryLayoutTest/MemoryLayoutTest.csproj -c Release
 dotnet run --project example/csharp/MemoryLayoutTest -c Release
 
 dotnet build example/csharp/J2Orbit.TestApp/J2Orbit.TestApp.csproj -c Release
 dotnet run --project example/csharp/J2Orbit.TestApp -c Release
```

### 原生 DLL 加载说明
- P/Invoke 使用库“基名”`j2_orbit_propagator`，运行时会在不同平台自动映射：
  - Windows：`j2_orbit_propagator.dll`
  - Linux：`libj2_orbit_propagator.so`
  - macOS：`libj2_orbit_propagator.dylib`
- 请确保原生库位于可执行输出目录或系统可搜索路径（如 `PATH`）下。示例项目已配置将 `example/csharp/j2_orbit_propagator.dll` 复制到输出目录。

### 简单示例（C#）
```csharp
using J2.Propagator;

// 以 J2000 历元秒为 0 创建星座传播器
default:
var constellation = new ConstellationPropagator(0.0);

// 添加一颗卫星（紧凑根数）
constellation.AddSatellite(new CCompactOrbitalElements
{
    a = 7000e3, e = 0.001, i = 98 * Math.PI / 180.0, O = 0, w = 0, M = 0
});

// 推进 3600 秒并读取状态
constellation.Propagate(3600.0);
var state = constellation.GetSatelliteState(0);
Console.WriteLine($"pos: [{state.r[0]}, {state.r[1]}, {state.r[2]}]");
```

### 常见问题排查
- “Unable to load DLL 'j2_orbit_propagator' (0x8007007E)”：
  - 确认 DLL 存在于输出目录并与进程架构匹配（x64）
  - 将 DLL 所在目录加入 `PATH` 或与可执行文件放在同一目录
- “Entry point not found”/找不到入口点：
  - 确保原生库已按本仓库最新代码构建
  - 运行 `example/csharp/build_and_test_csharp.ps1` 进行全量构建与测试
- CUDA 不可用：`ConstellationPropagator.IsCudaAvailable()` 返回 `false` 为预期；需要正确安装/配置 CUDA 才能启用 GPU 模式

## License
MIT License