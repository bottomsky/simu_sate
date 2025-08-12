# J2轨道传播器 C API 使用指南

本文档介绍如何使用J2轨道传播器的C语言接口，以便在Python、C#等外部语言中调用。

## 目录

- [概述](#概述)
- [构建动态库](#构建动态库)
- [C API 接口](#c-api-接口)
- [Python 绑定](#python-绑定)
- [C# 绑定](#c-绑定)
- [其他语言绑定](#其他语言绑定)
- [示例代码](#示例代码)
- [故障排除](#故障排除)

## 概述

J2轨道传播器提供了完整的C语言接口，使得外部语言可以通过动态库调用以下功能：

- **轨道传播**: 使用J2摄动模型进行轨道外推
- **坐标转换**: ECI/ECEF坐标系之间的相互转换
- **轨道要素转换**: 轨道要素与状态向量之间的转换
- **参数配置**: 积分步长、自适应步长等参数设置
- **工具函数**: GMST计算、角度归一化等

## 构建动态库

### Windows 平台

使用提供的PowerShell脚本构建：

```powershell
# 基本构建
.\build_dynamic_library.ps1

# 指定构建类型
.\build_dynamic_library.ps1 -BuildType Release

# 清理并重新构建
.\build_dynamic_library.ps1 -Clean

# 构建并安装
.\build_dynamic_library.ps1 -Install -InstallPrefix "C:\Program Files\J2OrbitPropagator"
```

### Linux/macOS 平台

```bash
# 创建构建目录
mkdir build && cd build

# 配置CMake
cmake .. -DCMAKE_BUILD_TYPE=Release -DBUILD_EXAMPLES=ON

# 构建
make -j$(nproc)

# 安装（可选）
sudo make install
```

### 手动构建

如果不使用CMake，也可以手动编译：

```bash
# 编译源文件
g++ -shared -fPIC -O3 -std=c++17 \
    -I./include -I./lib/eigen \
    src/j2_orbit_propagator.cpp src/j2_orbit_propagator_c.cpp \
    -o libj2_orbit_propagator.so
```

## C API 接口

### 数据结构

```c
// 轨道要素结构体
typedef struct {
    double a;   // 半长轴 (m)
    double e;   // 偏心率
    double i;   // 倾角 (rad)
    double O;   // 升交点赤经 (rad)
    double w;   // 近地点幅角 (rad)
    double M;   // 平近点角 (rad)
    double t;   // 历元时间 (s)
} COrbitalElements;

// 状态向量结构体
typedef struct {
    double r[3];  // 位置矢量 (m) [x, y, z]
    double v[3];  // 速度矢量 (m/s) [vx, vy, vz]
} CStateVector;

// 传播器句柄（不透明指针）
typedef void* J2PropagatorHandle;
```

### 核心函数

#### 创建和销毁

```c
// 创建传播器实例
J2PropagatorHandle j2_propagator_create(const COrbitalElements* initial_elements);

// 销毁传播器实例
void j2_propagator_destroy(J2PropagatorHandle handle);
```

#### 轨道传播

```c
// 轨道传播
int j2_propagator_propagate(J2PropagatorHandle handle, double target_time, COrbitalElements* result);

// 轨道要素到状态向量
int j2_propagator_elements_to_state(J2PropagatorHandle handle, const COrbitalElements* elements, CStateVector* state);

// 状态向量到轨道要素
int j2_propagator_state_to_elements(J2PropagatorHandle handle, const CStateVector* state, double time, COrbitalElements* elements);
```

#### 坐标转换

```c
// ECI到ECEF位置转换
int j2_eci_to_ecef_position(const double eci_position[3], double utc_seconds, double ecef_position[3]);

// ECEF到ECI位置转换
int j2_ecef_to_eci_position(const double ecef_position[3], double utc_seconds, double eci_position[3]);

// ECI到ECEF速度转换
int j2_eci_to_ecef_velocity(const double eci_position[3], const double eci_velocity[3], double utc_seconds, double ecef_velocity[3]);

// ECEF到ECI速度转换
int j2_ecef_to_eci_velocity(const double ecef_position[3], const double ecef_velocity[3], double utc_seconds, double eci_velocity[3]);
```

#### 工具函数

```c
// 计算格林威治平恒星时
int j2_compute_gmst(double utc_seconds, double* gmst);

// 角度归一化
double j2_normalize_angle(double angle);
```

### 返回值约定

- `0`: 成功
- `非0`: 失败（具体错误码待定义）
- `NULL`: 创建函数失败

## Python 绑定

### 安装依赖

```bash
pip install ctypes  # 通常已内置
```

### 使用示例

```python
from j2_orbit_propagator_python import J2OrbitPropagator

# 定义初始轨道要素
initial_elements = {
    'a': 6.78e6,        # 半长轴 (m)
    'e': 0.0001,        # 偏心率
    'i': 0.9006,        # 倾角 (rad)
    'O': 0.0,           # 升交点赤经 (rad)
    'w': 0.0,           # 近地点幅角 (rad)
    'M': 0.0,           # 平近点角 (rad)
    't': 0.0            # 历元时间 (s)
}

# 创建传播器
propagator = J2OrbitPropagator(initial_elements)

# 传播轨道
result = propagator.propagate(3600.0)  # 1小时后
print(f"传播后轨道要素: {result}")

# 坐标转换
from j2_orbit_propagator_python import eci_to_ecef_position
ecef_pos = eci_to_ecef_position([7000000, 0, 0], 3600.0)
print(f"ECEF位置: {ecef_pos}")
```

### 完整示例

运行提供的Python示例：

```bash
python example/python_binding_example.py
```

## C# 绑定

### 项目设置

在C#项目中添加以下NuGet包（如果需要）：

```xml
<PackageReference Include="System.Runtime.InteropServices" Version="4.3.0" />
```

### 使用示例

```csharp
using J2OrbitPropagatorBinding;

// 定义初始轨道要素
var initialElements = new COrbitalElements
{
    a = 6.78e6,        // 半长轴 (m)
    e = 0.0001,        // 偏心率
    i = 0.9006,        // 倾角 (rad)
    O = 0.0,           // 升交点赤经 (rad)
    w = 0.0,           // 近地点幅角 (rad)
    M = 0.0,           // 平近点角 (rad)
    t = 0.0            // 历元时间 (s)
};

// 创建传播器
using (var propagator = new J2OrbitPropagator(initialElements))
{
    // 传播轨道
    var result = propagator.Propagate(3600.0);  // 1小时后
    Console.WriteLine($"传播后轨道要素: {result}");
    
    // 坐标转换
    double[] eciPos = {7000000, 0, 0};
    double[] ecefPos = J2Utils.EciToEcefPosition(eciPos, 3600.0);
    Console.WriteLine($"ECEF位置: [{ecefPos[0]}, {ecefPos[1]}, {ecefPos[2]}]");
}
```

### 编译和运行

```bash
# 编译C#示例
csc /reference:System.Runtime.InteropServices.dll example/CSharpBindingExample.cs

# 运行
./CSharpBindingExample.exe
```

## 其他语言绑定

### Julia

```julia
# 加载动态库
const j2lib = Libdl.dlopen("./libj2_orbit_propagator.so")

# 定义函数
j2_propagator_create = Libdl.dlsym(j2lib, :j2_propagator_create)

# 调用函数
# ... (具体实现)
```

### MATLAB

```matlab
% 加载动态库
loadlibrary('j2_orbit_propagator', 'j2_orbit_propagator_c.h');

% 调用函数
% ... (具体实现)
```

### Rust

```rust
// 在Cargo.toml中添加
// [dependencies]
// libc = "0.2"

extern "C" {
    fn j2_propagator_create(elements: *const COrbitalElements) -> *mut c_void;
    // ... 其他函数声明
}

// 使用示例
// ... (具体实现)
```

## 示例代码

### 基本轨道传播示例

```c
#include "j2_orbit_propagator_c.h"
#include <stdio.h>

int main() {
    // 定义初始轨道要素
    COrbitalElements initial = {
        .a = 6.78e6,
        .e = 0.0001,
        .i = 0.9006,
        .O = 0.0,
        .w = 0.0,
        .M = 0.0,
        .t = 0.0
    };
    
    // 创建传播器
    J2PropagatorHandle handle = j2_propagator_create(&initial);
    if (!handle) {
        printf("创建传播器失败\n");
        return 1;
    }
    
    // 传播轨道
    COrbitalElements result;
    int ret = j2_propagator_propagate(handle, 3600.0, &result);
    if (ret != 0) {
        printf("轨道传播失败\n");
        j2_propagator_destroy(handle);
        return 1;
    }
    
    printf("传播后轨道要素:\n");
    printf("  a = %.3f m\n", result.a);
    printf("  e = %.6f\n", result.e);
    printf("  i = %.6f rad\n", result.i);
    
    // 清理资源
    j2_propagator_destroy(handle);
    return 0;
}
```

### 坐标转换示例

```c
#include "j2_orbit_propagator_c.h"
#include <stdio.h>

int main() {
    double eci_pos[3] = {7000000.0, 0.0, 0.0};
    double ecef_pos[3];
    double utc_time = 3600.0;
    
    // ECI到ECEF转换
    int ret = j2_eci_to_ecef_position(eci_pos, utc_time, ecef_pos);
    if (ret != 0) {
        printf("坐标转换失败\n");
        return 1;
    }
    
    printf("ECI位置: [%.3f, %.3f, %.3f] m\n", 
           eci_pos[0], eci_pos[1], eci_pos[2]);
    printf("ECEF位置: [%.3f, %.3f, %.3f] m\n", 
           ecef_pos[0], ecef_pos[1], ecef_pos[2]);
    
    return 0;
}
```

## 故障排除

### 常见问题

1. **动态库加载失败**
   - 确保动态库文件在系统PATH中或与可执行文件同目录
   - 检查动态库的架构（32位/64位）是否与调用程序匹配
   - 在Linux上，可能需要设置`LD_LIBRARY_PATH`

2. **函数调用失败**
   - 检查函数参数是否正确
   - 确保传入的指针不为NULL
   - 验证轨道要素的数值范围是否合理

3. **编译错误**
   - 确保包含了正确的头文件
   - 检查Eigen库是否正确安装
   - 验证C++编译器版本（需要C++17支持）

### 调试技巧

1. **启用详细输出**
   ```bash
   # 在Linux上查看动态库依赖
   ldd libj2_orbit_propagator.so
   
   # 在Windows上使用Dependency Walker
   depends.exe j2_orbit_propagator.dll
   ```

2. **检查函数导出**
   ```bash
   # Linux
   nm -D libj2_orbit_propagator.so | grep j2_
   
   # Windows
   dumpbin /exports j2_orbit_propagator.dll
   ```

3. **内存检查**
   ```bash
   # 使用Valgrind检查内存泄漏
   valgrind --leak-check=full ./your_program
   ```

### 性能优化

1. **编译优化**
   - 使用Release模式编译（`-O3`优化）
   - 启用特定CPU指令集优化

2. **使用建议**
   - 重用传播器实例，避免频繁创建/销毁
   - 对于大量计算，考虑使用自适应步长
   - 批量处理多个轨道时，可以并行化

### 联系支持

如果遇到问题，请提供以下信息：
- 操作系统和版本
- 编译器版本
- 错误消息的完整输出
- 最小可重现的示例代码

## 版本历史

- **v1.0.0**: 初始版本，包含基本的轨道传播和坐标转换功能
- **v1.1.0**: 添加自适应步长支持
- **v1.2.0**: 增加ECI/ECEF坐标转换功能

## 许可证

本项目采用 [MIT许可证](LICENSE)。