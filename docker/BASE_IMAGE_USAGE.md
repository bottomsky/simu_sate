# C++ 开发基础镜像使用指南

## 概述

本项目现在使用两阶段构建方式：
1. **基础镜像** (`cpp-dev-base:latest`) - 包含常用的 C++ 开发依赖
2. **项目镜像** (`j2-orbit-propagator-alpine:latest`) - 基于基础镜像构建的项目特定镜像

## 基础镜像信息

### 镜像名称
- `cpp-dev-base:latest`

### 包含的依赖
- **cmake** - 构建系统
- **git** - 版本控制
- **pkgconfig** - 包配置工具
- **eigen-dev** - Eigen 线性代数库
- **linux-headers** - Linux 内核头文件
- **build-base** - 基础构建工具（gcc, g++, make 等）
- **gcompat** 和 **libc6-compat** - 兼容性库

### 基础镜像特点
- 基于 Alpine Linux，轻量级
- 包含完整的 C++ 开发环境
- 适合作为其他 C++ 项目的基础镜像
- 预装常用开发依赖，减少重复安装时间

## 使用方法

### 1. 仅构建基础镜像

```powershell
# 在 docker 目录下运行
.\build.ps1 -BuildBaseOnly
```

这将构建 `cpp-dev-base:latest` 基础镜像，包含所有 C++ 开发依赖。

### 2. 完整构建流程（推荐）

```powershell
# 在 docker 目录下运行
.\build.ps1
```

这将：
1. 检查基础镜像是否存在，如不存在则自动构建
2. 基于基础镜像构建项目镜像
3. 提取构建产物到本地

### 3. 跳过基础镜像构建

```powershell
# 假设基础镜像已存在
.\build.ps1 -SkipBase
```

### 4. 使用挂载模式

```powershell
# 直接输出到主机目录
.\build.ps1 -WithMount
.\build.ps1 -WithMount -MountPath ./my-libs
```

## 构建脚本参数说明

| 参数 | 说明 |
|------|------|
| `-BuildBaseOnly` | 仅构建基础镜像 |
| `-SkipBase` | 跳过基础镜像构建（假设已存在） |
| `-NoCleanup` | 跳过清理步骤 |
| `-NoExtract` | 跳过提取构建产物 |
| `-WithMount` | 使用挂载方式直接输出到主机目录 |
| `-MountPath` | 指定挂载的主机路径（默认: ./build/docker-output） |
| `-Help` | 显示帮助信息 |

## 文件结构

```
docker/
├── Dockerfile.base          # 基础镜像 Dockerfile
├── Dockerfile               # 项目镜像 Dockerfile（基于基础镜像）
├── build-base.ps1          # 基础镜像独立构建脚本
├── build.ps1               # 主构建脚本（支持基础镜像）
├── build.sh                # Linux 构建脚本
├── docker-compose.yml      # Docker Compose 配置
└── BUILD_USAGE.md          # 构建使用说明
```

## 为其他项目使用基础镜像

### 1. 构建基础镜像

```powershell
# 在本项目的 docker 目录下
.\build.ps1 -BuildBaseOnly
```

### 2. 在其他项目中使用

在其他 C++ 项目的 Dockerfile 中：

```dockerfile
# 使用 C++ 开发基础镜像
FROM cpp-dev-base:latest as builder

# 设置工作目录
WORKDIR /workspace

# 复制项目源代码
COPY . .

# 构建项目
RUN mkdir -p build && \
    cd build && \
    cmake .. && \
    make -j$(nproc)

# 运行时镜像
FROM alpine:latest as runtime
RUN apk add --no-cache libstdc++
COPY --from=builder /workspace/build/your-app /usr/local/bin/
CMD ["your-app"]
```

## 镜像管理

### 查看镜像

```powershell
# 查看基础镜像
docker images cpp-dev-base

# 查看项目镜像
docker images j2-orbit-propagator-alpine
```

### 清理镜像

```powershell
# 删除基础镜像
docker rmi cpp-dev-base:latest

# 删除项目镜像
docker rmi j2-orbit-propagator-alpine:latest
```

### 推送到镜像仓库

```powershell
# 标记镜像
docker tag cpp-dev-base:latest your-registry/cpp-dev-base:latest

# 推送镜像
docker push your-registry/cpp-dev-base:latest
```

## 优势

1. **减少构建时间** - 基础依赖只需构建一次
2. **提高复用性** - 多个 C++ 项目可共享同一基础镜像
3. **简化维护** - 依赖更新只需修改基础镜像
4. **标准化环境** - 确保所有项目使用相同的开发环境
5. **减少镜像大小** - 避免重复安装相同依赖

## 故障排除

### 基础镜像构建失败

1. 检查 Docker 是否正常运行
2. 确保网络连接正常（需要下载 Alpine 包）
3. 检查 `Dockerfile.base` 文件是否存在

### 项目构建失败

1. 确保基础镜像存在：`docker images cpp-dev-base`
2. 如果基础镜像不存在，先运行 `.\build.ps1 -BuildBaseOnly`
3. 检查项目源代码是否完整

### 权限问题

在 Windows 上，确保 PowerShell 执行策略允许运行脚本：

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```