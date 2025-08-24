# Vulkan SDK 安装指南

本文档提供了在 Windows 系统上安装 Vulkan SDK 的详细步骤，以便能够编译和运行 J2 轨道传播器的可视化模块。

## 问题诊断

如果您遇到以下错误信息：
```
错误: 需要完整的 Vulkan SDK 进行开发。请从 https://vulkan.lunarg.com/ 下载并安装 Vulkan SDK，确保选择包含开发组件的完整安装。
```

这表明您的系统中只安装了 Vulkan 运行时（通常由显卡驱动程序提供），但缺少开发所需的头文件和库。

## 安装步骤

### 1. 下载 Vulkan SDK

1. 访问 [Vulkan SDK 官方网站](https://vulkan.lunarg.com/)
2. 点击 "Download" 按钮
3. 选择适合您系统的版本（通常选择最新的稳定版本）
4. 下载 Windows 版本的安装程序

### 2. 安装 Vulkan SDK

1. 运行下载的安装程序
2. **重要**：在安装选项中，确保选择以下组件：
   - Vulkan SDK Core
   - Vulkan SDK Headers
   - Vulkan SDK Libraries
   - Vulkan SDK Tools
   - Vulkan SDK Documentation（可选）

3. 选择安装路径（建议使用默认路径）
4. 完成安装

### 3. 验证安装

安装完成后，打开新的 PowerShell 窗口并运行：

```powershell
# 检查环境变量
echo $env:VULKAN_SDK

# 验证 vulkaninfo 工具
vulkaninfo --summary

# 检查头文件是否存在
Test-Path "$env:VULKAN_SDK\Include\vulkan\vulkan.h"
```

如果所有命令都成功执行，说明 Vulkan SDK 已正确安装。

### 4. 运行可视化构建脚本

现在您可以运行可视化构建脚本：

```powershell
.\scripts\build_visualization.ps1
```

## 常见问题

### Q: 安装后仍然提示找不到 Vulkan SDK

**A**: 请尝试以下解决方案：

1. **重启计算机**：环境变量可能需要重启后才能生效
2. **检查环境变量**：确保 `VULKAN_SDK` 环境变量已正确设置
3. **重新安装**：卸载当前安装，然后重新安装 Vulkan SDK

### Q: CMake 仍然找不到 Vulkan

**A**: 如果环境变量设置正确但 CMake 仍然找不到 Vulkan，可以尝试：

1. 清理 CMake 缓存：
   ```powershell
   .\scripts\build_visualization.ps1 -Clean
   ```

2. 手动指定 Vulkan 路径：
   ```powershell
   cmake -DVULKAN_SDK="$env:VULKAN_SDK" -B build
   ```

### Q: 显卡不支持 Vulkan

**A**: 确保您的显卡支持 Vulkan API：

1. 更新显卡驱动程序到最新版本
2. 检查显卡是否支持 Vulkan（大多数 2016 年后的显卡都支持）
3. 运行 `vulkaninfo` 查看支持的设备列表

## 系统要求

- **操作系统**：Windows 10/11 (64-bit)
- **显卡**：支持 Vulkan 1.0 或更高版本的显卡
- **驱动程序**：最新的显卡驱动程序
- **内存**：至少 4GB RAM
- **存储空间**：至少 2GB 可用空间用于 Vulkan SDK

## 其他依赖项

除了 Vulkan SDK，可视化模块还需要以下依赖项：

- **GLFW3**：窗口管理库
- **GLM**：数学库
- **CMake**：构建系统（3.15 或更高版本）
- **Visual Studio Build Tools**：C++ 编译器

这些依赖项可以通过 vcpkg 安装：

```powershell
# 安装 vcpkg（如果尚未安装）
git clone https://github.com/Microsoft/vcpkg.git
cd vcpkg
.\bootstrap-vcpkg.bat

# 安装依赖项
.\vcpkg install glfw3:x64-windows
.\vcpkg install glm:x64-windows
```

## 联系支持

如果您在安装过程中遇到问题，请：

1. 检查本文档的常见问题部分
2. 查看 Vulkan SDK 官方文档
3. 在项目 GitHub 仓库中创建 Issue

---

**注意**：本指南适用于 Windows 系统。Linux 和 macOS 用户请参考相应的安装文档。