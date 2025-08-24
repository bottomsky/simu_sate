# J2 轨道可视化系统

## 概述

本可视化系统使用 Vulkan 图形 API 实现地球和卫星轨道的 3D 实时渲染，集成了 J2 摄动轨道外推算法，为轨道力学研究和教学提供直观的可视化工具。

## 功能特性

### 核心功能
- **地球渲染**：高质量的地球球体渲染，支持纹理映射和光照效果
- **轨道可视化**：实时显示卫星轨道路径和卫星位置
- **J2 摄动计算**：集成 J2 摄动效应的轨道外推算法
- **交互式相机**：支持鼠标和键盘控制的 3D 相机系统
- **多轨道支持**：同时显示多条不同颜色的轨道

### 渲染特性
- **Vulkan 渲染**：使用现代 Vulkan API 实现高性能渲染
- **着色器效果**：自定义顶点和片段着色器实现特殊视觉效果
- **大气散射**：简化的大气散射效果增强视觉真实感
- **发光效果**：轨道和卫星的发光和淡出效果
- **时间动画**：基于时间的动画和淡出效果

## 系统架构

```
visualization/
├── include/                    # 头文件
│   ├── visualization_types.h   # 核心数据类型定义
│   ├── vulkan_renderer.h      # Vulkan 渲染器
│   ├── earth_renderer.h       # 地球渲染器
│   ├── orbit_renderer.h       # 轨道渲染器
│   └── j2_orbit_integration.h # J2 轨道算法集成
├── src/                       # 源文件
│   ├── vulkan_renderer.cpp
│   ├── earth_renderer.cpp
│   ├── orbit_renderer.cpp
│   └── j2_orbit_integration.cpp
├── shaders/                   # 着色器文件
│   ├── earth.vert            # 地球顶点着色器
│   ├── earth.frag            # 地球片段着色器
│   ├── orbit.vert            # 轨道顶点着色器
│   └── orbit.frag            # 轨道片段着色器
├── examples/                  # 示例程序
│   └── orbit_visualization_demo.cpp
├── docs/                      # 文档
│   └── README.md
└── CMakeLists.txt            # 构建配置
```

## 依赖要求

### 必需依赖
- **Vulkan SDK** (>= 1.3.0)
- **GLFW** (>= 3.3)
- **GLM** (>= 0.9.9)
- **CMake** (>= 3.16)
- **C++17** 兼容编译器

### 可选依赖
- **stb_image** - 纹理加载（如果需要地球纹理）
- **tinyobjloader** - 3D 模型加载（如果需要复杂卫星模型）

## 构建说明

### 1. 安装依赖

#### Windows (使用 vcpkg)
```bash
# 安装 vcpkg
git clone https://github.com/Microsoft/vcpkg.git
cd vcpkg
.\bootstrap-vcpkg.bat

# 安装依赖
.\vcpkg install vulkan glfw3 glm
```

#### Linux (Ubuntu/Debian)
```bash
# 安装 Vulkan SDK
wget -qO - https://packages.lunarg.com/lunarg-signing-key-pub.asc | sudo apt-key add -
sudo wget -qO /etc/apt/sources.list.d/lunarg-vulkan-focal.list https://packages.lunarg.com/vulkan/lunarg-vulkan-focal.list
sudo apt update
sudo apt install vulkan-sdk

# 安装其他依赖
sudo apt install libglfw3-dev libglm-dev
```

#### macOS (使用 Homebrew)
```bash
# 安装 Vulkan SDK
brew install --cask vulkan-sdk

# 安装其他依赖
brew install glfw glm
```

### 2. 构建项目

```bash
# 进入项目根目录
cd j2-perturbation-orbit-propagator

# 创建构建目录
mkdir build
cd build

# 配置 CMake
cmake ..

# 编译
cmake --build . --config Release
```

### 3. 运行示例

```bash
# 运行轨道可视化演示
./visualization/examples/orbit_visualization_demo
```

## 使用指南

### 基本使用

```cpp
#include "vulkan_renderer.h"
#include "earth_renderer.h"
#include "orbit_renderer.h"
#include "j2_orbit_integration.h"

using namespace j2_orbit_visualization;

// 1. 初始化渲染系统
VulkanRenderer vulkanRenderer;
vulkanRenderer.initialize(window);

EarthRenderer earthRenderer;
earthRenderer.initialize(vulkanRenderer);

OrbitRenderer orbitRenderer;
orbitRenderer.initialize(vulkanRenderer);

// 2. 创建可视化管理器
OrbitVisualizationManager manager(earthRenderer, orbitRenderer);

// 3. 添加轨道
OrbitalElements elements;
elements.semiMajorAxis = 6778137.0;  // 400 km 高度
elements.eccentricity = 0.001;
elements.inclination = glm::radians(51.6f);
// ... 设置其他轨道参数

PropagationParams params;
params.startTime = 0.0;
params.endTime = 5400.0;  // 1.5 小时
params.timeStep = 60.0;   // 1 分钟步长

uint32_t orbitId = manager.addOrbitPropagation(elements, params,
    glm::vec3(0.0f, 1.0f, 0.0f),  // 绿色轨道
    glm::vec3(1.0f, 1.0f, 0.0f)); // 黄色卫星

// 4. 渲染循环
while (!glfwWindowShouldClose(window)) {
    vulkanRenderer.beginFrame();
    
    // 渲染地球
    EarthRenderParams earthParams;
    // ... 设置渲染参数
    earthRenderer.render(earthParams);
    
    // 渲染轨道
    orbitRenderer.render(view, proj, cameraPos, time);
    
    vulkanRenderer.endFrame();
}
```

### 轨道参数说明

#### 轨道根数 (OrbitalElements)
- `semiMajorAxis`: 半长轴 (米)
- `eccentricity`: 偏心率 (0-1)
- `inclination`: 轨道倾角 (弧度)
- `argumentOfPerigee`: 近地点幅角 (弧度)
- `longitudeOfAscendingNode`: 升交点赤经 (弧度)
- `trueAnomaly`: 真近点角 (弧度)

#### 传播参数 (PropagationParams)
- `startTime`: 开始时间 (秒)
- `endTime`: 结束时间 (秒)
- `timeStep`: 时间步长 (秒)

### 常见轨道类型示例

#### 低地球轨道 (LEO)
```cpp
OrbitalElements leo;
leo.semiMajorAxis = 6778137.0;      // 400 km 高度
leo.eccentricity = 0.001;           // 近圆轨道
leo.inclination = glm::radians(51.6f); // ISS 倾角
```

#### 地球同步轨道 (GEO)
```cpp
OrbitalElements geo;
geo.semiMajorAxis = 42164169.0;     // 地球同步轨道半径
geo.eccentricity = 0.0;             // 圆轨道
geo.inclination = 0.0;              // 赤道轨道
```

#### 极地轨道
```cpp
OrbitalElements polar;
polar.semiMajorAxis = 7178137.0;    // 800 km 高度
polar.eccentricity = 0.01;
polar.inclination = glm::radians(90.0f); // 极地轨道
```

## 控制说明

### 相机控制
- **WASD**: 前后左右移动
- **QE**: 上下移动
- **鼠标移动**: 视角旋转
- **鼠标滚轮**: 缩放
- **ESC**: 退出程序

### 渲染控制
- **F1**: 切换线框模式
- **F2**: 切换轨道可见性
- **F3**: 切换卫星可见性
- **F4**: 重置相机位置

## API 参考

### VulkanRenderer 类

#### 主要方法
- `initialize(GLFWwindow* window)`: 初始化 Vulkan 渲染器
- `beginFrame()`: 开始渲染帧
- `endFrame()`: 结束渲染帧
- `waitIdle()`: 等待设备空闲
- `cleanup()`: 清理资源

### EarthRenderer 类

#### 主要方法
- `initialize(VulkanRenderer& renderer)`: 初始化地球渲染器
- `render(const EarthRenderParams& params)`: 渲染地球
- `updateParams(const EarthRenderParams& params)`: 更新渲染参数

### OrbitRenderer 类

#### 主要方法
- `addOrbit(const std::vector<OrbitPoint>& points, const glm::vec3& color, bool visible)`: 添加轨道
- `addSatellite(const SatelliteState& state, const glm::vec3& color, float size, bool visible)`: 添加卫星
- `updateOrbit(uint32_t orbitId, const std::vector<OrbitPoint>& points)`: 更新轨道
- `removeOrbit(uint32_t orbitId)`: 移除轨道
- `setOrbitVisible(uint32_t orbitId, bool visible)`: 设置轨道可见性

### OrbitVisualizationManager 类

#### 主要方法
- `addOrbitPropagation(const OrbitalElements& elements, const PropagationParams& params, const glm::vec3& orbitColor, const glm::vec3& satelliteColor)`: 添加轨道传播任务
- `updateOrbitPropagation(uint32_t taskId, const PropagationParams& newParams)`: 更新轨道传播
- `removeOrbitPropagation(uint32_t taskId)`: 移除轨道传播任务
- `setOrbitVisible(uint32_t taskId, bool visible)`: 设置轨道可见性

## 性能优化

### 渲染优化
1. **LOD 系统**: 根据距离调整渲染细节
2. **视锥剔除**: 只渲染可见的轨道段
3. **实例化渲染**: 批量渲染相似对象
4. **纹理压缩**: 使用压缩纹理格式

### 计算优化
1. **多线程**: 在后台线程进行轨道计算
2. **缓存**: 缓存计算结果避免重复计算
3. **自适应步长**: 根据轨道特性调整时间步长

## 故障排除

### 常见问题

#### 1. Vulkan 初始化失败
- 确保安装了最新的 Vulkan SDK
- 检查显卡驱动是否支持 Vulkan
- 验证 Vulkan 验证层是否正确安装

#### 2. 着色器编译错误
- 检查着色器文件路径是否正确
- 确保 glslc 编译器在系统路径中
- 验证着色器语法是否正确

#### 3. 纹理加载失败
- 检查纹理文件是否存在
- 验证纹理格式是否支持
- 确保纹理尺寸是 2 的幂次

#### 4. 性能问题
- 降低轨道点密度
- 减少同时显示的轨道数量
- 调整渲染质量设置

### 调试技巧

1. **启用 Vulkan 验证层**
```cpp
#ifdef _DEBUG
    vulkanRenderer.enableValidationLayers(true);
#endif
```

2. **性能分析**
```cpp
// 使用 Vulkan 时间戳查询
auto renderTime = vulkanRenderer.getLastFrameTime();
std::cout << "Frame time: " << renderTime << " ms" << std::endl;
```

3. **内存使用监控**
```cpp
auto memoryUsage = vulkanRenderer.getMemoryUsage();
std::cout << "GPU memory usage: " << memoryUsage << " MB" << std::endl;
```

## 扩展开发

### 添加新的渲染效果

1. **创建新的着色器**
```glsl
// custom_effect.vert
#version 450
// ... 顶点着色器代码

// custom_effect.frag
#version 450
// ... 片段着色器代码
```

2. **扩展渲染器类**
```cpp
class CustomRenderer {
public:
    VisualizationError initialize(VulkanRenderer& renderer);
    void render(const CustomRenderParams& params);
    void cleanup();
};
```

### 集成新的轨道算法

1. **实现算法接口**
```cpp
class CustomOrbitPropagator : public OrbitPropagatorBase {
public:
    PropagationResult propagate(const OrbitalElements& elements,
                               const PropagationParams& params) override;
};
```

2. **注册到可视化管理器**
```cpp
visualizationManager.registerPropagator(std::make_unique<CustomOrbitPropagator>());
```

## 许可证

本项目采用 MIT 许可证。详见 LICENSE 文件。

## 贡献指南

欢迎贡献代码！请遵循以下步骤：

1. Fork 本仓库
2. 创建特性分支 (`git checkout -b feature/amazing-feature`)
3. 提交更改 (`git commit -m 'Add amazing feature'`)
4. 推送到分支 (`git push origin feature/amazing-feature`)
5. 创建 Pull Request

## 联系方式

如有问题或建议，请通过以下方式联系：

- 项目主页：[GitHub Repository]
- 问题报告：[GitHub Issues]
- 邮箱：[project-email]

## 更新日志

### v1.0.0 (2024-01-XX)
- 初始版本发布
- 基础 Vulkan 渲染系统
- 地球和轨道可视化
- J2 摄动算法集成
- 交互式相机控制