# J2 轨道传播器跨平台构建指南

本文档提供了在不同平台上构建 J2 轨道传播器 C++ 动态库的详细说明。

## 目录

- [系统要求](#系统要求)
- [Linux 构建](#linux-构建)
- [Windows 构建](#windows-构建)
- [构建选项说明](#构建选项说明)
- [Docker 构建](#docker-构建)
- [故障排除](#故障排除)
- [输出文件说明](#输出文件说明)

## 系统要求

### 通用要求

- **CMake**: 3.12 或更高版本
- **C++ 编译器**: 支持 C++17 标准
- **Eigen3**: 已包含在项目中 (`lib/eigen`)

### Linux 特定要求

- **编译器**: GCC 7+ 或 Clang 6+
- **构建工具**: Make 或 Ninja
- **可选**: CUDA Toolkit 11.0+ (用于 GPU 加速)

### Windows 特定要求

- **编译器**: Visual Studio 2019/2022 或 MinGW-w64
- **构建工具**: MSBuild (Visual Studio) 或 Ninja
- **可选**: CUDA Toolkit 11.0+ (用于 GPU 加速)

## Linux 构建

### 快速开始

```bash
# 基本构建
./build_dynamic_library.sh

# 清理构建
./build_dynamic_library.sh --clean

# 启用 CUDA 支持
./build_dynamic_library.sh --enable-cuda

# 自定义构建目录和类型
./build_dynamic_library.sh -t Debug -b /tmp/j2_build
```

### 详细选项

```bash
./build_dynamic_library.sh [OPTIONS]

选项:
  -t, --type TYPE         构建类型 (Debug, Release, RelWithDebInfo, MinSizeRel)
                         默认: Release
  -b, --build-dir DIR    构建目录路径
                         默认: build
  -c, --clean            构建前清理构建目录
  -i, --install          构建后安装
  -g, --generator GEN    CMake 生成器
                         默认: Unix Makefiles
  -j, --jobs NUM         并行作业数
                         默认: $(nproc)
  --enable-cuda          强制启用 CUDA 支持
  --disable-tests        禁用测试构建
  --disable-examples     禁用示例构建
  -h, --help             显示帮助信息
```

### 示例命令

```bash
# 发布版本构建，启用 CUDA，使用 8 个并行作业
./build_dynamic_library.sh -t Release --enable-cuda -j 8

# 调试版本构建，禁用测试
./build_dynamic_library.sh -t Debug --disable-tests

# 使用 Ninja 生成器
./build_dynamic_library.sh -g Ninja -c

# 构建并安装到自定义目录
./build_dynamic_library.sh -i -b /opt/j2_build
```

## Windows 构建

### 快速开始

```powershell
# 基本构建
.\build_dynamic_library.ps1

# 清理构建
.\build_dynamic_library.ps1 -Clean

# 启用 CUDA 支持
.\build_dynamic_library.ps1 -EnableCuda

# 自定义构建目录和类型
.\build_dynamic_library.ps1 -BuildType Debug -BuildDir "C:\temp\j2_build"
```

### 详细选项

```powershell
.\build_dynamic_library.ps1 [参数]

参数:
  -BuildType TYPE        构建类型 (Debug, Release, RelWithDebInfo, MinSizeRel)
                        别名: -t, -config
                        默认: Release
  -Generator GEN         CMake 生成器
                        别名: -g
                        默认: "Visual Studio 17 2022"
  -BuildDir DIR          构建目录
                        别名: -b
                        默认: build
  -Clean                 构建前清理构建目录
                        别名: -c
  -Install               构建后安装
                        别名: -i
  -InstallPrefix DIR     安装前缀目录
                        别名: -p, -prefix
                        默认: .\install
  -EnableCuda            强制启用 CUDA 支持
  -DisableTests          禁用测试构建
  -DisableExamples       禁用示例构建
  -Jobs NUM              并行作业数
                        别名: -j
                        默认: 自动检测处理器核心数
```

### 示例命令

```powershell
# 发布版本构建，启用 CUDA，使用 8 个并行作业
.\build_dynamic_library.ps1 -t Release -EnableCuda -j 8

# 调试版本构建，禁用测试
.\build_dynamic_library.ps1 -t Debug -DisableTests

# 使用 Ninja 生成器
.\build_dynamic_library.ps1 -g Ninja -c

# 使用 MinGW 生成器
.\build_dynamic_library.ps1 -g "MinGW Makefiles"

# 构建并安装到自定义目录
.\build_dynamic_library.ps1 -Install -InstallPrefix "C:\J2OrbitPropagator"
```

## 构建选项说明

### 构建类型

- **Debug**: 包含调试信息，未优化，适用于开发和调试
- **Release**: 完全优化，无调试信息，适用于生产环境
- **RelWithDebInfo**: 优化构建但包含调试信息
- **MinSizeRel**: 针对最小二进制大小优化

### CMake 生成器

#### Linux
- **Unix Makefiles**: 默认，使用 Make
- **Ninja**: 更快的构建系统
- **CodeBlocks - Unix Makefiles**: IDE 支持

#### Windows
- **Visual Studio 17 2022**: 默认，VS2022 支持
- **Visual Studio 16 2019**: VS2019 支持
- **Ninja**: 跨平台快速构建
- **MinGW Makefiles**: MinGW-w64 支持

### CUDA 支持

- **自动检测**: 默认行为，如果检测到 CUDA Toolkit 则启用
- **强制启用**: 使用 `--enable-cuda` (Linux) 或 `-EnableCuda` (Windows)
- **支持的架构**: 75, 86, 89, 90 (可在 CMakeLists.txt 中修改)

## Docker 构建（推荐）

### 前提条件
- Docker Desktop
- 对于 Windows 构建：需要 Windows 容器支持
- 对于 CUDA 支持：需要 NVIDIA Docker 运行时

### 快速开始

#### 使用构建脚本（推荐）

**Linux/macOS:**
```bash
cd docker

# 构建 Linux 版本
./build.sh

# 构建 Windows 版本
./build.sh -p windows

# 构建两个平台
./build.sh -p both

# 启用 CUDA 支持
./build.sh --enable-cuda

# 自定义构建选项
./build.sh -t Debug -j 8 --disable-tests
```

**Windows PowerShell:**
```powershell
cd docker

# 构建 Linux 版本
.\build.ps1

# 构建 Windows 版本
.\build.ps1 -Platform windows

# 构建两个平台
.\build.ps1 -Platform both

# 启用 CUDA 支持
.\build.ps1 -EnableCuda

# 自定义构建选项
.\build.ps1 -BuildType Debug -Jobs 8 -DisableTests
```

#### 使用 Docker Compose

```bash
cd docker

# 构建 Linux 版本
docker-compose up linux-build

# 构建 Windows 版本（需要 Windows 容器）
docker-compose up windows-build

# 构建两个平台
docker-compose up
```

#### 直接使用 Docker 命令

```bash
# 构建 Linux 版本
docker build -t j2-orbit-propagator:linux -f docker/Dockerfile.linux .

# 构建 Windows 版本（需要 Windows 容器）
docker build -t j2-orbit-propagator:windows -f docker/Dockerfile.windows .

# 运行容器并提取构建产物
docker run --rm -v $(pwd)/output:/host_output j2-orbit-propagator:linux
```

## 故障排除

### 常见问题

#### 1. CMake 版本过低
```
CMake Error: CMake 3.12 or higher is required
```
**解决方案**: 升级 CMake 到 3.12 或更高版本

#### 2. C++17 编译器不支持
```
CMake Error: The compiler does not support C++17
```
**解决方案**: 升级编译器或使用支持 C++17 的编译器

#### 3. CUDA 编译错误
```
nvcc fatal: Unsupported gpu architecture 'compute_XX'
```
**解决方案**: 在 CMakeLists.txt 中调整 `CMAKE_CUDA_ARCHITECTURES` 设置

#### 4. Windows 上的 DLL 导入错误
```
error LNK2019: unresolved external symbol __declspec(dllimport)
```
**解决方案**: 确保正确设置了 `J2_BUILD_DLL` 或 `J2_BUILD_STATIC` 宏

### 调试构建问题

```bash
# Linux: 详细构建输出
./build_dynamic_library.sh -t Debug --verbose

# Windows: 详细构建输出
.\build_dynamic_library.ps1 -t Debug -Verbose
```

## 输出文件说明

### Linux 输出文件

```
build/
├── libj2_orbit_propagator.so      # 动态库
├── libj2_orbit_propagator_static.a # 静态库
├── j2_example                      # 示例程序
├── unit_tests                      # 单元测试
├── integration_tests               # 集成测试
└── performance_tests               # 性能测试
```

### Windows 输出文件

```
build/
├── Release/
│   ├── j2_orbit_propagator.dll     # 动态库
│   ├── j2_orbit_propagator.lib     # 导入库
│   ├── j2_orbit_propagator_static.lib # 静态库
│   ├── j2_example.exe              # 示例程序
│   ├── unit_tests.exe              # 单元测试
│   ├── integration_tests.exe       # 集成测试
│   └── performance_tests.exe       # 性能测试
```

## 运行测试

### 所有平台

```bash
# 进入构建目录
cd build

# 运行所有测试
ctest --output-on-failure

# 运行特定测试
ctest -R unit_tests --output-on-failure
```

### 手动运行测试

```bash
# Linux
./unit_tests
./integration_tests
./performance_tests

# Windows
.\Release\unit_tests.exe
.\Release\integration_tests.exe
.\Release\performance_tests.exe
```

## 集成到其他项目

### CMake 集成

```cmake
# 查找已安装的包
find_package(J2OrbitPropagator REQUIRED)

# 链接到你的目标
target_link_libraries(your_target J2OrbitPropagator::j2_orbit_propagator_shared)
```

### 手动集成

```cmake
# 包含头文件目录
include_directories(/path/to/j2-orbit-propagator/include)

# 链接库文件
target_link_libraries(your_target /path/to/libj2_orbit_propagator.so)  # Linux
target_link_libraries(your_target /path/to/j2_orbit_propagator.lib)    # Windows
```

## 许可证

请参阅项目根目录中的 LICENSE 文件了解许可证信息。

## 支持

如果遇到构建问题，请：

1. 检查系统要求是否满足
2. 查看故障排除部分
3. 在项目 GitHub 仓库中提交 Issue
4. 提供详细的错误信息和系统环境信息