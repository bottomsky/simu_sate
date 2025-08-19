# J2 Orbit Propagator 动态库路径说明

## 构建产物位置

### 1. 容器内路径
- **运行时库路径**: `/usr/local/lib/`
  - `libj2_orbit_propagator.so` - 主要动态库
  - `libj2_orbit_propagator_static.a` - 静态库
  - `libgtest.so`, `libgmock.so` - 测试库

- **构建输出路径**: `/output/`
  - 构建过程中的临时输出目录
  - 可通过挂载方式直接获取构建产物

### 2. 主机路径（挂载模式）
- **默认挂载路径**: `./build/docker-output/`
- **绝对路径**: `D:\code\j2-perturbation-orbit-propagator\build\docker-output\`

## 使用方法

### 方法1：挂载模式构建（推荐）
```powershell
# 使用默认挂载路径
.\docker\build.ps1 -WithMount

# 使用自定义挂载路径
.\docker\build.ps1 -WithMount -MountPath "./my-output"
```

### 方法2：从容器中提取
```powershell
# 标准构建模式
.\docker\build.ps1

# 构建产物会提取到 ./build/alpine-artifacts/
```

### 方法3：直接从容器运行时使用
```bash
# 运行容器并挂载输出目录
docker run --rm -v "${PWD}\build\docker-output:/output" j2-orbit-propagator-alpine:latest

# 库文件位于容器内 /usr/local/lib/ 目录
```

## 构建过程中的路径信息

构建过程会自动打印以下信息：
- 容器内构建目录和文件列表
- 输出目录的完整路径
- 挂载参数和主机路径
- 库文件的大小和修改时间

## 库文件说明

| 文件名 | 类型 | 大小 | 用途 |
|--------|------|------|------|
| `libj2_orbit_propagator.so` | 动态库 | ~76KB | 主要功能库 |
| `libj2_orbit_propagator_static.a` | 静态库 | ~102KB | 静态链接版本 |
| `libgtest.so` | 动态库 | ~658KB | Google Test 框架 |
| `libgmock.so` | 动态库 | ~156KB | Google Mock 框架 |
| `libgtest_main.so` | 动态库 | ~17KB | Google Test 主函数 |
| `libgmock_main.so` | 动态库 | ~17KB | Google Mock 主函数 |

## 注意事项

1. **挂载模式**：构建产物直接输出到主机目录，无需额外提取步骤
2. **路径格式**：Windows 环境下使用反斜杠路径分隔符
3. **权限**：挂载目录会自动创建，确保有足够的磁盘空间
4. **清理**：旧的构建产物会在新构建开始前自动清理