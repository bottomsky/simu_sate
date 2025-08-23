# J2 轨道传播器测试脚本使用文档

## 概述

`build_and_run_tests.ps1` 是一个 PowerShell 脚本，用于自动化构建和运行 J2 轨道传播器项目的各种测试。支持单元测试、集成测试和性能测试，并可选择性地启用或禁用 CUDA 支持。

## 脚本位置

```
tests\build_and_run_tests.ps1
```

## 基本语法

```powershell
.\tests\build_and_run_tests.ps1 [参数选项]
```

## 参数详解

### 配置参数

| 参数 | 别名 | 类型 | 默认值 | 说明 |
|------|------|------|--------|------|
| `-Config` | `-c` | string | "Release" | 构建配置（Release/Debug） |
| `-BuildDir` | `-b`, `-bd` | string | 自动推断 | 构建目录路径 |
| `-Generator` | `-g` | string | 默认 | CMake 生成器 |

### 测试目标选择

| 参数 | 别名 | 类型 | 说明 |
|------|------|------|------|
| `-Target` | `-t` | string | 指定具体测试目标 |
| `-Unit` | `-u` | switch | 快速选择单元测试 |
| `-Integration` | `-it` | switch | 快速选择集成测试 |
| `-Performance` | `-p` | switch | 快速选择性能测试 |
| `-AllTests` | `-all` | switch | 运行所有测试 |

### 执行控制

| 参数 | 别名 | 类型 | 说明 |
|------|------|------|------|
| `-Run` | `-r` | switch | 构建后立即运行测试 |
| `-JustTest` | `-jt`, `-j` | switch | 仅运行测试（不重新构建） |

### 输出控制

| 参数 | 别名 | 类型 | 说明 |
|------|------|------|------|
| `-Verbose` | `-v` | switch | 显示详细测试输出 |
| `-ShowCUDA` | `-scuda` | switch | 直接展示 CUDA 一致性测试输出 |

### CUDA 控制

| 参数 | 别名 | 类型 | 说明 |
|------|------|------|------|
| `-NoCuda` | `-nc` | switch | 禁用 CUDA 编译 |

## 构建目录规则

脚本会根据测试目标自动推断构建目录：

- 单元测试: `tests\build-unit`
- 集成测试: `tests\build-integration`  
- 性能测试: `tests\build-performance`
- 所有测试: `tests\build-all`
- 其他目标: `tests\build`

## 与统一构建脚本/输出目录协作

- 统一输出策略：项目所有动态/静态库与可执行文件统一收集到仓库根目录 `bin/`。
- 原生构建入口：建议通过 `scripts/build.ps1` 进行 C++/CUDA 原生构建与清理；该脚本提供：
  - `-Clean`：清理构建缓存，但保留 `build/CMakeLists.txt`
  - `-Reconfigure`：移除 `CMakeCache.txt` 与 `CMakeFiles/` 以强制全量配置
  - `-EnableCuda`：启用 CUDA 构建
- 测试脚本关系：测试脚本会在 `tests\build-*` 目录下单独进行 CMake 配置与构建；它不会清理根构建目录。
  - 若需彻底刷新根构建，再运行测试，建议先执行：
    ```powershell
    .\scripts\build.ps1 -Clean -Reconfigure -Config Release
    ```
  - Docker 流程：推荐使用挂载方式将容器 `/output` 直接同步到主机 `./bin`，详见 `docker/README-动态库路径说明.md`
  - 端到端验证清单：参见 [CROSS_PLATFORM_BUILD.md — 验证清单（快速自检）](../CROSS_PLATFORM_BUILD.md#validation-checklist)

## 常用使用示例

### 1. 基础用法

```powershell
# 构建并运行集成测试（启用 CUDA，Release 模式）
.\tests\build_and_run_tests.ps1 -Integration -Run

# 构建并运行单元测试
.\tests\build_and_run_tests.ps1 -Unit -Run

# 构建并运行性能测试
.\tests\build_and_run_tests.ps1 -Performance -Run
```

### 2. 配置选择

```powershell
# Debug 模式运行集成测试
.\tests\build_and_run_tests.ps1 -Integration -Run -Config Debug

# Release 模式运行所有测试
.\tests\build_and_run_tests.ps1 -AllTests -Run -Config Release
```

### 3. CUDA 控制

```powershell
# 禁用 CUDA 运行集成测试
.\tests\build_and_run_tests.ps1 -Integration -Run -NoCuda

# 启用 CUDA 并查看详细 CUDA 测试输出
.\tests\build_and_run_tests.ps1 -Integration -Run -ShowCUDA
```

### 4. 详细输出

```powershell
# 显示详细测试输出
.\tests\build_and_run_tests.ps1 -Integration -Run -Verbose

# 仅运行测试（不重新构建）
.\tests\build_and_run_tests.ps1 -Integration -JustTest
```

### 5. 自定义构建目录

```powershell
# 指定自定义构建目录
.\tests\build_and_run_tests.ps1 -Integration -Run -BuildDir "custom-build"
```

## 工作流程说明

### 1. 标准构建和测试流程

```powershell
.\tests\build_and_run_tests.ps1 -Integration -Run
```

执行步骤：
1. 创建/检查构建目录 `tests\build-integration`
2. 配置 CMake（启用 CUDA，配置测试构建）
3. 构建 `integration_tests` 目标
4. 通过 ctest 运行集成测试
5. 显示测试结果

### 2. 仅测试流程

```powershell
.\tests\build_and_run_tests.ps1 -Integration -JustTest
```

执行步骤：
1. 检查构建目录是否存在
2. 直接通过 ctest 运行测试（跳过构建）

## CUDA 支持说明

### 启用 CUDA（默认）
- 脚本会自动检测 CUDA Toolkit
- 支持的架构：75, 86, 89, 90（Turing, Ampere, Ada Lovelace, Hopper）
- 编译 `j2_cuda_kernels.cu` 源文件
- 链接 CUDA 运行时库

### 禁用 CUDA
- 使用 `-NoCuda` 参数
- 跳过 CUDA 相关代码编译
- CUDA 测试用例会显示"编译期禁用CUDA，跳过测试"

## 测试类型详解

### 单元测试 (Unit Tests)
- **文件**: `tests/unit/single_satellite_propagation_test.cpp`
- **功能**: 测试单颗卫星的轨道传播算法
- **构建目标**: `unit_tests`
- **数据输出**: 生成 JSON 格式的仿真结果文件

### 集成测试 (Integration Tests)
- **文件**: 
  - `cuda_consistency_test.cpp` - CUDA 一致性测试
  - `mode_consistency_regression_test.cpp` - 模式一致性回归测试
  - `simd_consistency_test.cpp` - SIMD 一致性测试
  - `parameter_sweep_test.cpp` - 参数扫描测试
- **功能**: 验证不同计算模式（CPU/SIMD/CUDA）的结果一致性
- **构建目标**: `integration_tests`

### 性能测试 (Performance Tests)
- **文件**: `tests/performance/constellation_benchmark_test.cpp`
- **功能**: 星座传播性能基准测试，比较不同计算模式的性能
- **构建目标**: `performance_tests`
- **特性**: 启用 AVX2/FMA 优化

## 与 C# 示例协作

- C# 脚本：`example/csharp/build_and_test_csharp.ps1`
  - 统一调用 `scripts/build.ps1` 构建原生库
  - 参数：`-CleanNative`、`-NativeReconfigure`、`-NativeConfig Release|Debug|RelWithDebInfo|MinSizeRel`
  - 从项目根 `bin/` 复制 `j2_orbit_propagator` 动态库到 C# 运行目录，完成 P/Invoke 运行时加载

## 输出文件位置

### 可执行文件
- 单元测试: `tests\build-unit\unit\Release\unit_tests.exe`
- 集成测试: `tests\build-integration\integration\Release\integration_tests.exe`
- 性能测试: `tests\build-performance\performance\Release\performance_tests.exe`

### 测试数据
- 单元测试数据: `tests\data\multi_simulation_results_step_*.json`

## 注意事项

1. 管理员权限：通常不需要，除非 CUDA 安装需要特殊权限
2. 路径要求：脚本必须从项目根目录调用
3. 依赖检查：确保 CMake、编译器和必要的库已安装
4. 并行构建：脚本默认使用系统可用的 CPU 核心数进行并行构建
5. 清理建议：如需完全重新配置根构建，先执行 `scripts/build.ps1 -Clean -Reconfigure`；如需刷新测试构建，可删除对应的 `tests\build-*` 目录

## 联系与支持

如遇到问题，请检查：
1. 系统环境是否满足要求
2. 依赖库是否正确安装
3. 构建日志中的具体错误信息

### 1. CUDA 编译错误
```
error C1083: 无法打开包含文件: "cuda_runtime.h"
```
**解决方案**: 使用 `-NoCuda` 参数禁用 CUDA 或安装 CUDA Toolkit

### 2. 构建目录不存在
```
[ERROR] Build directory does not exist. Run a build first.
```
**解决方案**: 移除 `-JustTest` 参数，让脚本重新构建

### 3. CMake 配置失败
**检查项**:
- CMake 版本 ≥ 3.10
- Visual Studio 2019/2022 已安装
- CUDA Toolkit 已正确安装（如需要）

## 高级用法

### 1. 组合多个参数
```powershell
# Debug 模式，禁用 CUDA，显示详细输出的集成测试
.\tests\build_and_run_tests.ps1 -Integration -Run -Config Debug -NoCuda -Verbose
```

### 2. 脚本链式调用
```powershell
# 先运行单元测试，再运行集成测试
.\tests\build_and_run_tests.ps1 -Unit -Run
.\tests\build_and_run_tests.ps1 -Integration -Run
```

### 3. 性能对比测试
```powershell
# 启用 CUDA 的性能测试
.\tests\build_and_run_tests.ps1 -Performance -Run

# 禁用 CUDA 的性能测试（仅 CPU/SIMD）
.\tests\build_and_run_tests.ps1 -Performance -Run -NoCuda
```

## 输出文件位置

### 可执行文件
- 单元测试: `tests\build-unit\unit\Release\unit_tests.exe`
- 集成测试: `tests\build-integration\integration\Release\integration_tests.exe`
- 性能测试: `tests\build-performance\performance\Release\performance_tests.exe`

### 测试数据
- 单元测试数据: `tests\data\multi_simulation_results_step_*.json`

## 注意事项

1. **管理员权限**: 通常不需要，除非 CUDA 安装需要特殊权限
2. **路径要求**: 脚本必须从项目根目录调用
3. **依赖检查**: 确保 CMake、编译器和必要的库已安装
4. **并行构建**: 脚本默认使用系统可用的 CPU 核心数进行并行构建
5. **清理构建**: 如需完全重新构建，可手动删除对应的 `build-*` 目录

## 联系与支持

如遇到问题，请检查：
1. 系统环境是否满足要求
2. 依赖库是否正确安装
3. 构建日志中的具体错误信息

更多技术细节请参考项目的 CMakeLists.txt 文件和源代码注释。