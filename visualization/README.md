# J2 轨道可视化模块

基于 Vulkan 的高性能 3D 轨道可视化系统，用于 J2 轨道传播算法的可视化展示。

## 功能特性

- **高性能渲染**: 基于 Vulkan API 的现代图形渲染
- **地球可视化**: 真实感地球球体渲染，支持纹理映射和光照
- **轨道可视化**: 实时卫星轨道路径和位置显示
- **J2 算法集成**: 与 J2 轨道传播算法无缝集成
- **交互式控制**: 支持鼠标和键盘的相机控制
- **多平台支持**: Windows、Linux、macOS

## 系统要求

### 硬件要求
- 支持 Vulkan 1.0+ 的显卡
- 最低 4GB 显存（推荐 8GB+）
- 64位处理器

### 软件依赖
- **Vulkan SDK** (1.3.0+) - [安装指南](docs/VULKAN_SETUP.md)
- **GLFW** (3.3+) - 窗口管理
- **GLM** (0.9.9+) - 数学库
- **CMake** (3.16+)
- **C++17** 兼容编译器

## 安装依赖

### Windows

1. **安装 Vulkan SDK**
   ```bash
   # 下载并安装 Vulkan SDK
   # https://vulkan.lunarg.com/sdk/home
   ```

2. **使用 vcpkg 安装其他依赖**
   ```bash
   vcpkg install glfw3:x64-windows
   vcpkg install glm:x64-windows
   ```

### Linux (Ubuntu/Debian)

```bash
# 安装 Vulkan SDK
sudo apt update
sudo apt install vulkan-sdk vulkan-tools vulkan-validationlayers-dev

# 安装其他依赖
sudo apt install libglfw3-dev libglm-dev

# 验证安装
vulkaninfo
```

### Linux (CentOS/RHEL)

```bash
# 安装 Vulkan SDK
sudo yum install vulkan-devel vulkan-tools vulkan-validation-layers-devel

# 安装其他依赖
sudo yum install glfw-devel glm-devel
```

### macOS

```bash
# 使用 Homebrew 安装
brew install vulkan-sdk glfw glm

# 或使用 MacPorts
sudo port install vulkan-sdk glfw glm
```

## 构建说明

### 快速构建

```bash
# Windows
scripts\build.bat --visualization

# Linux/macOS
./scripts/build.sh --visualization
```

### 手动构建

```bash
# 创建构建目录
mkdir build && cd build

# 配置项目
cmake .. -DBUILD_VISUALIZATION=ON -DCMAKE_BUILD_TYPE=Release

# 编译
cmake --build . --config Release --parallel
```

### 构建选项

| 选项 | 默认值 | 说明 |
|------|--------|------|
| `BUILD_VISUALIZATION` | OFF | 构建可视化模块 |
| `BUILD_EXAMPLES` | ON | 构建示例程序 |
| `BUILD_TESTS` | ON | 构建测试程序 |
| `CMAKE_BUILD_TYPE` | Release | 构建类型 (Debug/Release) |

## 使用指南

### 运行示例

```bash
# 进入构建目录
cd build

# 运行可视化演示
./bin/orbit_visualization_demo
```

### 控制说明

#### 鼠标控制
- **左键拖拽**: 旋转视角
- **右键拖拽**: 平移视角
- **滚轮**: 缩放

#### 键盘控制
- **WASD**: 相机移动
- **QE**: 上下移动
- **空格**: 暂停/继续动画
- **R**: 重置相机
- **ESC**: 退出程序
- **F1**: 显示/隐藏帮助信息
- **F2**: 显示/隐藏性能统计

### 编程接口

#### 基本使用

```cpp
#include "j2_orbit_integration.h"
#include "vulkan_renderer.h"

// 创建可视化管理器
OrbitVisualizationManager manager;

// 初始化
if (manager.initialize(800, 600, "轨道可视化") != VisualizationError::Success) {
    // 处理错误
}

// 添加轨道
OrbitalElements elements;
elements.semi_major_axis = 7000.0;  // km
elements.eccentricity = 0.01;
elements.inclination = 51.6;        // 度
elements.raan = 0.0;
elements.arg_periapsis = 0.0;
elements.true_anomaly = 0.0;

PropagationParams params;
params.duration = 86400.0;           // 1天
params.time_step = 60.0;             // 1分钟
params.j2_enabled = true;

uint32_t orbit_id = manager.addOrbitPropagation(elements, params);

// 主循环
while (!manager.shouldClose()) {
    manager.update();
    manager.render();
}

// 清理
manager.cleanup();
```

#### 高级功能

```cpp
// 设置轨道颜色
manager.setOrbitColor(orbit_id, {1.0f, 0.0f, 0.0f, 1.0f}); // 红色

// 设置轨道可见性
manager.setOrbitVisibility(orbit_id, false);

// 更新轨道参数
manager.updateOrbitPropagation(orbit_id, new_elements, new_params);

// 移除轨道
manager.removeOrbitPropagation(orbit_id);

// 设置进度回调
manager.setProgressCallback([](float progress, const std::string& status) {
    std::cout << "进度: " << (progress * 100) << "% - " << status << std::endl;
});
```

## 性能优化

### 渲染性能
- 使用合适的轨道点密度（推荐每分钟1个点）
- 限制同时显示的轨道数量（推荐 < 50条）
- 启用视锥剔除和距离剔除
- 使用适当的纹理分辨率

### 内存优化
- 定期清理不需要的轨道数据
- 使用流式加载大量轨道数据
- 监控 GPU 内存使用情况

### 配置建议

```cpp
// 性能优化配置
RenderStats stats;
stats.max_orbit_points = 10000;     // 每条轨道最大点数
stats.max_visible_orbits = 20;      // 最大可见轨道数
stats.culling_distance = 100000.0;  // 剔除距离 (km)
stats.target_fps = 60;              // 目标帧率

manager.setRenderStats(stats);
```

## 故障排除

### 常见问题

1. **Vulkan 初始化失败**
   ```
   错误: Failed to create Vulkan instance
   解决: 检查 Vulkan SDK 安装和驱动程序更新
   ```

2. **着色器编译失败**
   ```
   错误: Failed to compile shaders
   解决: 确保 glslc 在 PATH 中，或重新安装 Vulkan SDK
   ```

3. **窗口创建失败**
   ```
   错误: Failed to create GLFW window
   解决: 检查显卡驱动和 GLFW 安装
   ```

4. **性能问题**
   - 检查显卡是否支持 Vulkan
   - 降低轨道点密度
   - 减少同时显示的轨道数量
   - 更新显卡驱动程序

### 调试模式

```bash
# 启用 Vulkan 验证层
export VK_LAYER_PATH=$VULKAN_SDK/etc/vulkan/explicit_layer.d
export VK_INSTANCE_LAYERS=VK_LAYER_KHRONOS_validation

# 运行程序
./orbit_visualization_demo
```

### 日志输出

程序会在以下位置输出日志：
- Windows: `%TEMP%/j2_visualization.log`
- Linux: `/tmp/j2_visualization.log`
- macOS: `/tmp/j2_visualization.log`

## 扩展开发

### 添加新的渲染器

1. 继承 `BaseRenderer` 类
2. 实现必需的虚函数
3. 在 `VulkanRenderer` 中注册

```cpp
class CustomRenderer : public BaseRenderer {
public:
    VisualizationError initialize(VkDevice device, VkRenderPass renderPass) override;
    void render(VkCommandBuffer commandBuffer, const CameraParams& camera) override;
    void cleanup() override;
};
```

### 自定义着色器

1. 在 `shaders/` 目录创建新的着色器文件
2. 更新 `CMakeLists.txt` 添加编译规则
3. 在渲染器中加载和使用

### 添加新的轨道算法

1. 实现 `OrbitPropagator` 接口
2. 在 `OrbitVisualizationManager` 中集成
3. 更新配置和文档

## 许可证

本项目采用 MIT 许可证。详见 [LICENSE](../LICENSE) 文件。

## 贡献指南

1. Fork 项目
2. 创建功能分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 创建 Pull Request

## 联系方式

- 项目主页: [GitHub Repository]
- 问题报告: [GitHub Issues]
- 邮箱: [your-email@example.com]

## 更新日志

### v1.0.0 (2024-01-XX)
- 初始版本发布
- 基础 Vulkan 渲染引擎
- 地球和轨道可视化
- J2 算法集成
- 跨平台支持

---

**注意**: 本模块仍在积极开发中，API 可能会发生变化。建议在生产环境使用前进行充分测试。