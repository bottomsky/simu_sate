# Dynamic Library Summary

This document summarizes how the dynamic libraries are produced, named, and consumed across platforms. It also clarifies the output policy and how the C# example/test workflow finds the native binaries.

## Output Location

Primary build artifacts are located under `build/<Config>/` (e.g., `build/Release/`, `build/Debug/`), regardless of whether you build via:
- CMake directly
- Unified script `scripts/build.ps1`
- Docker builds (mount or extraction)
- C# integrated script (`scripts/example/csharp/build_and_test_csharp.ps1`)

所有构建产物统一输出到 `build/<配置>/` 目录，确保跨平台一致性。

## Library Names

- Windows: `j2_orbit_propagator.dll`, import library `j2_orbit_propagator.lib`
- Linux: `libj2_orbit_propagator.so`, static `libj2_orbit_propagator.a`
- macOS: `libj2_orbit_propagator.dylib`, static `libj2_orbit_propagator.a`

These names match the P/Invoke base name `j2_orbit_propagator` used by .NET bindings, enabling automatic platform mapping.

## Consumers

- C Example: Links against artifacts under `build/<Config>/j2_orbit_propagator.*`
- C#/.NET: P/Invoke (`DllImport("j2_orbit_propagator")`) expects the native library to be present in the runtime directory or on the system search path.
- Tests: Binaries and dependencies are discoverable under `build/<Config>/`.

## Build Variants

- Build types: Release, Debug, RelWithDebInfo, MinSizeRel
- CUDA-enabled variants are produced when CUDA is detected or explicitly forced via `-EnableCuda`.

## Clean and Reconfigure

- `scripts/build.ps1 -Clean` removes build intermediates but keeps `build/CMakeLists.txt`.
- `scripts/build.ps1 -Reconfigure` wipes CMake cache (`CMakeCache.txt` and `CMakeFiles/`) to force a fresh configure.

## Docker Notes

- Container builds publish artifacts into host `./build/Release/` via bind mounts, or can be exported there after the build.
- See `docker/README-动态库路径说明.md` for additional path mapping and extraction options.

## C# Integration Notes

- `scripts/example/csharp/build_and_test_csharp.ps1` 会从 `build/<配置>` 复制原生库到 C# 项目的输出目录。
- The script supports `-CleanNative`, `-NativeReconfigure`, `-NativeConfig` for controlling native build behavior from the managed side.

## Versioning

- Library versioning (if any) should be handled by CMake project version and optionally embedded into filenames as needed by downstream packaging systems.

## 概述

本项目已成功为 `j2_orbit_propagator.cpp` 构建了C格式的动态库，支持Python、C#等外部语言调用。

## 已完成的工作

### 1. C接口封装

- **头文件**: `j2_orbit_propagator_c.h`
  - 定义了C格式的数据结构 (`COrbitalElements`, `CStateVector`)
  - 声明了所有C接口函数
  - 提供了完整的API文档

- **实现文件**: `j2_orbit_propagator_c.cpp`
  - 实现了C++到C的接口转换
  - 提供了错误处理和内存管理
  - 支持所有核心功能：轨道传播、坐标转换、工具函数

### 2. 构建系统

- **CMake配置**: `CMakeLists.txt`
  - 支持静态库和动态库构建
  - 自动处理Eigen依赖
  - 配置了Windows DLL导出
  - 包含示例程序和测试

- **构建脚本**: `scripts/build_dynamic_library.ps1`
  - 自动化构建流程
  - 支持不同构建类型 (Debug/Release)
  - 自动复制动态库到示例目录
  - 提供详细的构建信息

### 3. 语言绑定示例

#### Python绑定
- **文件**: `example/python_binding_example.py`
- **功能**: 通过ctypes调用动态库
- **特性**: 
  - 面向对象的Python接口
  - 完整的错误处理
  - 轨道传播和坐标转换示例
  - 跨平台支持

#### C#绑定
- **文件**: `example/CSharpBindingExample.cs`
- **功能**: 通过P/Invoke调用动态库
- **特性**:
  - .NET兼容的接口设计
  - 内存安全的数据传递
  - 完整的示例代码
  - 自动构建脚本 (`build_and_run_csharp.ps1`)

#### C语言示例
- **文件**: `example/c_example.c`
- **功能**: 直接使用C接口
- **特性**:
  - 纯C语言实现
  - 完整的功能演示
  - 编译脚本 (`build_and_run_c.ps1`)

### 4. 文档和说明

- **API文档**: `C_API_README.md`
  - 详细的API参考
  - 多平台构建指南
  - 各种语言绑定说明
  - 故障排除指南

## 构建结果

### 生成的文件

构建完成后，动态库文件统一位于 `build/<配置>/` 目录：

| 平台 | 文件类型 | 位置 |
|------|----------|------|
| Linux | 共享库 | `build/Release/libj2_orbit_propagator.so` |
| Windows | 动态库 | `build/Release/j2_orbit_propagator.dll` |
| Windows | 导入库 | `build/Release/j2_orbit_propagator.lib` |
| macOS | 动态库 | `build/Release/libj2_orbit_propagator.dylib` |

```
build/Release/
├── j2_orbit_propagator.dll          # Windows动态库
├── j2_orbit_propagator.lib          # Windows导入库
├── j2_orbit_propagator_static.lib   # 静态库
└── j2_example.exe                   # 示例程序
```

### 测试结果

✅ **Python绑定测试**: 成功
- 动态库加载正常
- 轨道传播功能正确
- 坐标转换功能正确
- 工具函数正常工作

✅ **C#绑定测试**: 成功
- P/Invoke调用正常
- 数据结构映射正确
- 所有功能测试通过
- .NET 8.0兼容

✅ **构建系统测试**: 成功
- CMake配置正确
- 动态库生成成功
- 依赖处理正常
- 跨平台兼容

## 核心功能

### 1. 轨道传播
- J2摄动模型
- 自适应步长RK4积分
- 高精度轨道预测

### 2. 坐标转换
- ECI ↔ ECEF坐标转换
- 轨道要素 ↔ 状态向量转换
- GMST计算
- 角度归一化

### 3. 参数配置
- 积分步长设置
- 自适应步长控制
- 精度参数调整

## 使用方法

### 快速开始

1. **构建动态库**:
   ```powershell
   ./scripts/build_dynamic_library.ps1 -BuildType Release
   ```

2. **Python使用**:
   ```bash
   python example/python_binding_example.py
   ```

3. **C#使用**:
   ```powershell
   ./scripts/example/csharp/build_and_test_csharp.ps1
   ```

### 集成到其他项目

1. 复制动态库文件 (`j2_orbit_propagator.dll`)
2. 复制头文件 (`j2_orbit_propagator_c.h`)
3. 根据目标语言选择合适的绑定方式
4. 参考示例代码进行集成

## 技术特点

### 1. 高性能
- 优化的数值算法
- 高效的内存管理
- 最小化函数调用开销

### 2. 跨平台
- Windows/Linux/macOS支持
- 标准C接口
- CMake构建系统

### 3. 易用性
- 清晰的API设计
- 完整的文档
- 丰富的示例代码

### 4. 可扩展性
- 模块化设计
- 标准化接口
- 易于添加新功能

## 后续扩展建议

1. **更多语言绑定**:
   - Julia绑定
   - MATLAB MEX接口
   - Rust FFI绑定
   - JavaScript/Node.js绑定

2. **功能增强**:
   - 更多摄动模型 (J3, J4, 大气阻力)
   - 多体问题支持
   - 轨道机动建模

3. **性能优化**:
   - GPU加速支持
   - 并行计算
   - 内存池管理

4. **工具集成**:
   - Python包发布 (PyPI)
   - NuGet包发布
   - Docker容器化

## 总结

J2轨道传播器的C格式动态库构建已经完成，提供了完整的外部语言调用支持。该解决方案具有高性能、跨平台、易用等特点，可以满足各种应用场景的需求。通过标准化的C接口，用户可以轻松地在Python、C#、C等多种语言中使用该库的功能。