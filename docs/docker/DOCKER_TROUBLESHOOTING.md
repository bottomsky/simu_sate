# Docker 构建问题修复指南

## 问题概述

本文档记录了 Docker 构建失败问题的修复过程和解决方案。

## 修复的问题

### 1. Dockerfile 语法警告

**问题**: FROM 和 as 关键字大小写不匹配
```
WARN: FromAsCasing: 'as' and 'FROM' keywords' casing do not match (line 3)
WARN: FromAsCasing: 'as' and 'FROM' keywords' casing do not match (line 125)
```

**修复**: 统一使用大写 AS 关键字
```dockerfile
# 修复前
FROM cpp-dev-base:latest as builder
FROM alpine:latest as runtime

# 修复后
FROM cpp-dev-base:latest AS builder
FROM alpine:latest AS runtime
```

### 2. 网络连接问题

**问题**: 无法连接到 Docker Hub
```
ERROR: failed to do request: Head "https://registry-1.docker.io/v2/library/alpine/manifests/latest": 
dialing registry-1.docker.io:443 container via direct connection because static system has no HTTPS proxy: 
connecting to registry-1.docker.io:443: dial tcp 157.240.17.35:443: connectex: A connection attempt failed
```

**修复方案**:

#### A. 配置国内镜像源

1. **自动配置** (推荐):
   ```powershell
   .\configure-docker-mirror.ps1 -Apply
   ```

2. **手动配置**:
   - 打开 Docker Desktop
   - 点击设置图标 (齿轮图标)
   - 选择 'Docker Engine'
   - 在 JSON 配置中添加:
   ```json
   {
     "registry-mirrors": [
       "https://docker.mirrors.ustc.edu.cn",
       "https://hub-mirror.c.163.com",
       "https://mirror.baidubce.com",
       "https://ccr.ccs.tencentyun.com"
     ]
   }
   ```
   - 点击 'Apply & Restart'

#### B. 网络重试机制

构建脚本现在包含自动重试机制:
- 最多重试 3 次
- 每次重试间隔 10 秒
- 自动检测网络相关错误
- 提供镜像源配置建议

## 新增功能

### 1. 网络连接检测

构建前自动检测 Docker Hub 连接状态:
```powershell
[INFO] 检查 Docker 网络连接...
[INFO] 测试 Docker Hub 连接...
```

### 2. 智能错误处理

- 自动识别网络相关错误
- 提供针对性的解决建议
- 显示重试进度和状态

### 3. 镜像源配置工具

新增 `configure-docker-mirror.ps1` 脚本:
- 自动检测 Docker Desktop 配置路径
- 一键应用镜像源配置
- 提供详细的手动配置说明

## 使用方法

### 基本构建
```powershell
# 完整构建流程（包含网络检测和重试）
.\build.ps1

# 仅构建基础镜像
.\build.ps1 -BuildBaseOnly

# 跳过基础镜像构建
.\build.ps1 -SkipBase
```

### 网络问题解决
```powershell
# 配置镜像源
.\configure-docker-mirror.ps1 -Apply

# 测试网络连接
docker pull hello-world:latest
```

### 挂载模式构建
```powershell
# 直接输出到 build/Release 目录
.\build.ps1 -WithMount

# 输出到自定义目录
.\build.ps1 -WithMount -MountPath ./custom-output
```

## 故障排除

### 如果仍然遇到网络问题

1. **检查网络连接**:
   ```powershell
   ping registry-1.docker.io
   ```

2. **验证镜像源配置**:
   ```powershell
   docker info | Select-String "Registry Mirrors"
   ```

3. **重启 Docker Desktop**:
   - 完全退出 Docker Desktop
   - 重新启动应用程序

4. **清理 Docker 缓存**:
   ```powershell
   docker system prune -a
   ```

### 如果构建仍然失败

1. **检查基础镜像**:
   ```powershell
   docker images cpp-dev-base:latest
   ```

2. **重新构建基础镜像**:
   ```powershell
   .\build.ps1 -BuildBaseOnly
   ```

3. **查看详细错误信息**:
   ```powershell
   docker build --no-cache -f Dockerfile -t test . --progress=plain
   ```

## 文件说明

- `Dockerfile`: 主构建文件（已修复大小写问题）
- `Dockerfile.base`: 基础镜像构建文件
- `build.ps1`: 增强的构建脚本（包含重试机制）
- `daemon.json`: Docker 镜像源配置文件
- `configure-docker-mirror.ps1`: 镜像源配置工具
- `DOCKER_TROUBLESHOOTING.md`: 本故障排除指南

## 总结

通过以上修复，Docker 构建环境现在具备:
- ✅ 语法规范的 Dockerfile
- ✅ 网络连接检测和重试机制
- ✅ 国内镜像源配置支持
- ✅ 智能错误处理和建议
- ✅ 完整的故障排除工具

如果遇到其他问题，请参考本文档或查看构建脚本的详细输出信息。