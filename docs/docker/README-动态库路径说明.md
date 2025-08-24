# J2 Orbit Propagator 动态库路径说明

## 构建产物位置（统一策略）

本项目的主产物（DLL/EXE/SO/DYLIB）统一生成于 `build/<配置>` 目录，例如 `build/Debug` 或 `build/Release`。所有构建脚本都已更新为输出到此统一位置。

### 1. 容器内路径
- 运行时库路径: `/usr/local/lib/`
  - `libj2_orbit_propagator.so` - 主要动态库
  - `libj2_orbit_propagator_static.a` - 静态库
  - `libgtest.so`, `libgmock.so` - 测试库

- 构建输出路径: `/output/`
  - 容器内用于和主机目录进行挂载同步的输出目录

### 2. 主机路径（挂载模式，推荐）
- 默认挂载路径（从 scripts/docker 目录调用脚本时）: `../build/Release`
- 默认挂载路径（从项目根目录调用脚本时）: `./build/Release`
- 示例绝对路径: `D:\code\j2-perturbation-orbit-propagator\build\Release\`

## 使用方法

### 方法1：挂载模式构建（推荐）
```powershell
# 在 scripts\docker 目录下（输出到项目根 build/Release）
.\scripts\docker\build.ps1 -WithMount

# 自定义挂载路径（例如输出到 build/<配置>）
.\scripts\docker\build.ps1 -WithMount -MountPath "../build/Release"   # 在 scripts\docker 目录下
.\scripts\docker\build.ps1 -WithMount -MountPath "./build/Release"    # 在项目根目录下
.\scripts\docker\build.ps1 -WithMount -MountPath "./my-output"        # 自定义目录
```

### 方法2：从容器中提取（不挂载）
```powershell
# Windows PowerShell（自动提取到 ./build/Release）
.\scripts\docker\build.ps1

# 提示：Linux Shell 脚本（scripts/docker/build.sh）也可将提取路径调整为 ./build/<配置>
# 同样也可使用“方法1 挂载模式”或手动 docker run -v 将容器 /output 直接挂载到 ./build/<配置> 以加速同步
```

### 方法3：直接运行容器并手动挂载到 build/<配置>
```bash
# 运行容器并将主机 ./build/Release 挂载到容器 /output
# PowerShell/Windows
docker run --rm -v "${PWD}\build\Release:/output" j2-orbit-propagator-alpine:latest

# Bash/Linux 或 WSL
docker run --rm -v "$(pwd)/build/Release:/output" j2-orbit-propagator-alpine:latest

# 库文件仍位于容器内 /usr/local/lib/，构建/提取时同步到 /output（即主机 ./build/<配置>）
```

## 构建过程中的路径信息

构建过程会自动打印以下信息：
- 容器内构建目录和文件列表
- 输出目录（/output）的完整路径及其对应的主机挂载路径
- 挂载参数与主机路径
- 库文件的大小和修改时间

## 库文件说明（主机侧 build/<配置>/）

| 文件名 | 类型 | 用途 |
|--------|------|------|
| `libj2_orbit_propagator.so` | 动态库 | Linux 运行时主要功能库 |
| `libj2_orbit_propagator.dylib` | 动态库 | macOS 运行时主要功能库 |
| `j2_orbit_propagator.dll` | 动态库 | Windows 运行时主要功能库 |
| `j2_orbit_propagator.lib` | 导入库 | Windows 导入库 |
| `libj2_orbit_propagator.a` | 静态库 | Linux/macOS 静态链接版本 |

说明：以上库文件统一位于 `build/<配置>` 目录。所有示例脚本和构建工具都已更新为使用此统一输出位置。

## 与 C# 示例/测试的协作

- C# 构建脚本：`scripts/example/csharp/build_and_test_csharp.ps1`
  - 先调用顶层 `scripts/build.ps1` 进行原生构建
  - 支持参数：`-CleanNative`、`-NativeReconfigure`、`-NativeConfig Release|Debug|...`
  - 构建完成后，从 `build/<配置>` 复制本地库到 C# 工程输出目录，满足 P/Invoke 运行时加载

- 端到端验证清单：参见 [CROSS_PLATFORM_BUILD.md — 验证清单（快速自检）](../CROSS_PLATFORM_BUILD.md#validation-checklist)

## 注意事项

1. 挂载模式：构建产物将直接输出到主机 `./build/<配置>`，无需额外提取步骤
2. 路径格式：Windows 环境下使用反斜杠路径分隔符
3. 权限：挂载目录会自动创建，确保有足够的磁盘空间
4. 清理：建议使用 `scripts/build.ps1 -Clean` 清理构建缓存（会保留 `build/CMakeLists.txt`），并可结合 `-Reconfigure` 触发全量配置
5. Linux Shell 脚本（scripts/docker/build.sh）：如需统一到 `build/<配置>`，请优先使用挂载方式或调整提取路径