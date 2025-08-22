# J2 Orbit Propagator 动态库路径说明

## 构建产物位置（统一策略）

本项目将所有构建产物统一输出到仓库根目录的 `bin/` 目录。推荐通过 Docker 的挂载方式或 PowerShell 提取方式将容器内产物直接同步到主机的 `./bin`。

### 1. 容器内路径
- 运行时库路径: `/usr/local/lib/`
  - `libj2_orbit_propagator.so` - 主要动态库
  - `libj2_orbit_propagator_static.a` - 静态库
  - `libgtest.so`, `libgmock.so` - 测试库

- 构建输出路径: `/output/`
  - 容器内用于和主机目录进行挂载同步的输出目录

### 2. 主机路径（挂载模式，推荐）
- 默认挂载路径（从 docker 目录调用脚本时）: `../bin`
- 默认挂载路径（从项目根目录调用脚本时）: `./bin`
- 示例绝对路径: `D:\code\j2-perturbation-orbit-propagator\bin\`

## 使用方法

### 方法1：挂载模式构建（推荐）
```powershell
# 在 docker 目录下（输出到项目根 bin）
.\docker\build.ps1 -WithMount

# 自定义挂载路径（例如仍然输出到 bin）
.\docker\build.ps1 -WithMount -MountPath "../bin"   # 在 docker 目录下
.\docker\build.ps1 -WithMount -MountPath "./bin"    # 在项目根目录下
.\docker\build.ps1 -WithMount -MountPath "./my-output"  # 自定义目录
```

### 方法2：从容器中提取（不挂载）
```powershell
# Windows PowerShell（自动提取到 ./bin）
.\docker\build.ps1

# 提示：Linux Shell 脚本（build.sh）当前默认提取到 ./build/alpine-artifacts/
# 如需统一到 bin，建议使用“方法1 挂载模式”或手动 docker run -v 挂载到 ./bin
```

### 方法3：直接运行容器并手动挂载到 bin
```bash
# 运行容器并将主机 ./bin 挂载到容器 /output
# PowerShell/Windows
docker run --rm -v "${PWD}\bin:/output" j2-orbit-propagator-alpine:latest

# Bash/Linux 或 WSL
docker run --rm -v "$(pwd)/bin:/output" j2-orbit-propagator-alpine:latest

# 库文件仍位于容器内 /usr/local/lib/，并会复制到 /output（即主机 ./bin）
```

## 构建过程中的路径信息

构建过程会自动打印以下信息：
- 容器内构建目录和文件列表
- 输出目录（/output）的完整路径及其对应的主机挂载路径
- 挂载参数与主机路径
- 库文件的大小和修改时间

## 库文件说明（主机侧 bin/）

| 文件名 | 类型 | 用途 |
|--------|------|------|
| `libj2_orbit_propagator.so` | 动态库 | Linux 运行时主要功能库 |
| `libj2_orbit_propagator.dylib` | 动态库 | macOS 运行时主要功能库 |
| `j2_orbit_propagator.dll` | 动态库 | Windows 运行时主要功能库 |
| `j2_orbit_propagator.lib` | 导入库 | Windows 导入库 |
| `libj2_orbit_propagator.a` | 静态库 | Linux/macOS 静态链接版本 |

说明：以上库文件均推荐统一收集到仓库根目录 `bin/`。

## 与 C# 示例/测试的协作

- C# 构建脚本：`example/csharp/build_and_test_csharp.ps1`
  - 先调用顶层 `scripts/build.ps1` 进行原生构建
  - 支持参数：`-CleanNative`、`-NativeReconfigure`、`-NativeConfig Release|Debug|...`
  - 构建完成后，从仓库根 `bin/` 复制本地库到 C# 工程输出目录，满足 P/Invoke 运行时加载

## 注意事项

1. 挂载模式：构建产物将直接输出到主机 `./bin`，无需额外提取步骤
2. 路径格式：Windows 环境下使用反斜杠路径分隔符
3. 权限：挂载目录会自动创建，确保有足够的磁盘空间
4. 清理：建议使用 `scripts/build.ps1 -Clean` 清理构建缓存（会保留 `build/CMakeLists.txt`），并可结合 `-Reconfigure` 触发全量配置
5. Linux Shell 脚本（build.sh）：若需统一到 `bin/`，请优先使用挂载方式或后续将提取路径调整为 `./bin`