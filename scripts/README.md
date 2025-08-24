# J2 轨道传播器可视化构建脚本

本目录包含用于构建和运行 J2 轨道传播器 Vulkan 可视化演示程序的脚本。

## 脚本说明

### Windows 版本 (PowerShell)

**文件**: `build_visualization.ps1`

**功能**:
- 检查 Vulkan SDK 是否已安装
- 配置 CMake 并启用可视化模块
- 编译项目
- 运行可视化演示程序
- 包含完整的错误处理和用户友好的提示信息

**使用方法**:
```powershell
# 显示帮助信息
.\build_visualization.ps1 -Help

# 基本构建和运行
.\build_visualization.ps1

# 清理缓存并使用 Release 模式构建
.\build_visualization.ps1 -Clean -BuildType Release

# 跳过编译，直接运行演示程序
.\build_visualization.ps1 -SkipBuild
```

**参数说明**:
- `-Clean`: 清理构建缓存（保留 CMakeLists.txt）
- `-BuildType`: 构建类型，可选 Debug 或 Release（默认：Debug）
- `-SkipBuild`: 跳过编译步骤，直接运行演示程序
- `-Help`: 显示帮助信息

### Linux 版本 (Bash)

**文件**: `build_visualization.sh`

**功能**:
- 检查 Vulkan SDK 是否已安装
- 检查必要的编译工具（GCC/Clang, CMake, Make/Ninja）
- 配置 CMake 并启用可视化模块
- 编译项目
- 运行可视化演示程序
- 包含完整的错误处理和用户友好的提示信息

**使用方法**:
```bash
# 添加可执行权限
chmod +x build_visualization.sh

# 显示帮助信息
./build_visualization.sh --help

# 基本构建和运行
./build_visualization.sh

# 清理缓存并使用 Release 模式构建
./build_visualization.sh --clean --release

# 跳过编译，直接运行演示程序
./build_visualization.sh --skip-build
```

**参数说明**:
- `--clean`: 清理构建缓存（保留 CMakeLists.txt）
- `--release`: 使用 Release 构建类型（默认：Debug）
- `--skip-build`: 跳过编译步骤，直接运行演示程序
- `--help`: 显示帮助信息

## 依赖要求

### 通用要求
- CMake 3.15 或更高版本
- Vulkan SDK
- C++17 兼容的编译器

### Windows 特定要求
- Visual Studio 2019 或更高版本（或 Visual Studio Build Tools）
- PowerShell 5.0 或更高版本

### Linux 特定要求
- GCC 7+ 或 Clang 6+
- Make 或 Ninja 构建系统
- 开发包：
  - Ubuntu/Debian: `sudo apt install vulkan-tools libvulkan-dev build-essential cmake`
  - Fedora/RHEL: `sudo dnf install vulkan-tools vulkan-devel gcc-c++ cmake make`
  - Arch: `sudo pacman -S vulkan-tools vulkan-headers base-devel cmake`

## 安装 Vulkan SDK

### Windows
1. 访问 [Vulkan SDK 官网](https://vulkan.lunarg.com/)
2. 下载适用于 Windows 的最新版本
3. 运行安装程序并按照提示完成安装
4. 重启命令行或 PowerShell 以确保环境变量生效

### Linux

**方法 1: 包管理器安装（推荐）**
```bash
# Ubuntu/Debian
sudo apt update
sudo apt install vulkan-tools libvulkan-dev

# Fedora/RHEL
sudo dnf install vulkan-tools vulkan-devel

# Arch Linux
sudo pacman -S vulkan-tools vulkan-headers
```

**方法 2: 官方 SDK 安装**
1. 访问 [Vulkan SDK 官网](https://vulkan.lunarg.com/)
2. 下载适用于 Linux 的最新版本
3. 按照官方文档进行安装和配置

## 故障排除

### 常见问题

1. **找不到 Vulkan SDK**
   - 确保已正确安装 Vulkan SDK
   - 检查 `VULKAN_SDK` 环境变量是否设置
   - 重启终端或命令行

2. **CMake 配置失败**
   - 确保 CMake 版本 ≥ 3.15
   - 检查编译器是否支持 C++17
   - 尝试使用 `-Clean` 参数清理缓存

3. **编译失败**
   - 检查是否安装了所有必要的开发工具
   - 确保有足够的磁盘空间
   - 查看详细的编译错误信息

4. **演示程序无法启动**
   - 确保显卡支持 Vulkan
   - 检查显卡驱动是否为最新版本
   - 尝试运行 `vulkaninfo` 命令验证 Vulkan 安装

### 获取帮助

如果遇到问题，请：
1. 首先查看脚本输出的错误信息
2. 确认所有依赖项都已正确安装
3. 尝试使用 `-Help` 或 `--help` 参数查看详细用法
4. 检查项目的主要文档和 README 文件

## 注意事项

- 脚本会自动检测和验证所有必要的依赖项
- 构建过程可能需要几分钟时间，请耐心等待
- 首次运行时，CMake 可能需要下载一些依赖库
- 演示程序运行时，按 ESC 键可以退出
- 脚本包含完整的错误处理，会在遇到问题时给出明确的提示信息