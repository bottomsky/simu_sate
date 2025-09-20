# 统一构建脚本使用说明（scripts 目录）

本目录提供跨平台统一的构建入口脚本：
- Windows: build.ps1（PowerShell）
- Linux/macOS: build.sh（Bash）

两个脚本支持一致的核心参数，用于配置 CMake、控制是否构建测试/示例/可视化模块，以及选择生成器、并行度、工具链等。构建完成后，会将主要产物（动态库、静态库、示例/测试可执行文件）统一归集到 build/Release 目录，便于后续 Python/示例脚本查找和分发。

快速开始
- Windows（PowerShell）
  - 基本构建（Ninja）：
    .\build.ps1 -Config Release -Generator "Ninja" -Parallel 8
  - 关闭测试与示例、仅构建核心库：
    .\build.ps1 -Config Release -DisableTests -DisableExamples
  - 启用可视化模块：
    .\build.ps1 -Config Release -Visualization
  - 通过 WSL 同时产出 Linux .so（需启用 -AlsoLinux）：
    .\build.ps1 -Config Release -Generator "Ninja" -Parallel 8 -AlsoLinux
  - 指定工具链（例如 vcpkg 工具链）：
    .\build.ps1 -Config Release -Generator "Ninja" -Toolchain "C:\vcpkg\scripts\buildsystems\vcpkg.cmake"

- Linux/macOS（Bash）
  - 基本构建（Ninja）：
    bash build.sh --release --generator "Ninja" --jobs 8
  - 关闭测试与示例、仅构建核心库：
    bash build.sh --release --no-tests --no-examples
  - 启用可视化模块：
    bash build.sh --release --visualization
  - 指定工具链（关键字或绝对路径）：
    bash build.sh --release --toolchain aarch64
    bash build.sh --release --toolchain mingw
    bash build.sh --release --toolchain /abs/path/to/toolchain.cmake
  - 同时编译 Windows 目标（交叉编译，需安装 mingw-w64）：
    bash build.sh --release --also-windows

可用参数
- Windows: build.ps1
  - -Clean / -CleanCache: 清理 build 目录缓存（保留 build/CMakeLists.txt）
  - -Config <Debug|Release>: 构建类型（默认 Release）
  - -Parallel <N>: 并行构建任务数（默认 CPU 逻辑核数）
  - -Reconfigure: 强制删除 CMakeCache 后重新配置
  - -Generator <Name>: 指定 CMake 生成器（如 Ninja、Visual Studio 17 2022）
  - -Toolchain <path>: 指定 CMake 工具链文件
  - -EnableCuda: 启用 CUDA 支持
  - -DisableTests: 不构建测试
  - -DisableExamples: 不构建示例
  - -Visualization: 构建可视化模块
  - -AlsoLinux: 在 Windows 上通过 WSL 同时构建 Linux 目标并归集产物

- Linux/macOS: build.sh
  - --clean: 清理 build 目录缓存（保留 build/CMakeLists.txt）
  - --debug / --release: 构建类型（默认 Release）
  - --jobs <N>: 并行构建任务数
  - --generator <Name>: 指定 CMake 生成器（如 Ninja、Unix Makefiles）
  - --toolchain <path|mingw|aarch64>: 指定工具链文件或关键字
  - --cuda: 启用 CUDA 支持
  - --no-tests: 不构建测试
  - --no-examples: 不构建示例
  - --visualization: 构建可视化模块
  - --also-windows: 在 Linux 上额外交叉编译 Windows 目标并归集产物

构建产物归集
- Windows: 构建完成后将 .dll/.exe/.lib/.pdb/.exp 等复制到 build/Release
- Linux/macOS: 构建完成后将 .so/.a 以及示例/测试可执行文件复制到 build/Release
- 示例与测试二进制也会同步复制到该目录，便于一次性打包与验证

依赖提示
- CMake 3.15+
- 编译器：MSVC / GCC / Clang（支持 C++17）
- 可视化模块：需安装 Vulkan SDK、GLFW、GLM
- CUDA（可选）：需安装 NVIDIA CUDA Toolkit
- 交叉编译（可选）：
  - Linux -> Windows: 安装 mingw-w64 工具链
  - x86_64 -> aarch64: 使用提供的 cmake/toolchains/linux-aarch64.cmake
  - Windows -> Linux: 需已安装并启用 WSL（Windows Subsystem for Linux），并在 WSL 环境中具备 cmake、编译器和 Ninja

常见问题
- 缺少 Vulkan/CUDA/工具链：请根据提示安装对应依赖或移除相关选项
- 交叉编译失败：确认工具链是否可用、生成器是否兼容（建议 Ninja）
- Python 调用动态库路径：默认查找 build/Release 下的库文件

备注
- 旧版 build_visualization.* 和 build_dynamic_library.* 已合并进统一脚本，不再使用。
- Windows 下已移除 .bat 构建脚本，统一使用 PowerShell（build.ps1）。