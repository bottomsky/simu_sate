# J2 轨道可视化系统 API 参考

## 目录

- [核心类型](#核心类型)
- [VulkanRenderer](#vulkanrenderer)
- [EarthRenderer](#earthrenderer)
- [OrbitRenderer](#orbitrenderer)
- [J2OrbitPropagator](#j2orbitpropagator)
- [OrbitVisualizationManager](#orbitvisualizationmanager)
- [错误处理](#错误处理)
- [回调函数](#回调函数)

## 核心类型

### Vertex

顶点数据结构，用于 Vulkan 渲染管线。

```cpp
struct Vertex {
    glm::vec3 pos;      // 顶点位置
    glm::vec3 normal;   // 法向量
    glm::vec2 texCoord; // 纹理坐标
    
    static VkVertexInputBindingDescription getBindingDescription();
    static std::array<VkVertexInputAttributeDescription, 3> getAttributeDescriptions();
};
```

### UniformBufferObject

统一缓冲区对象，包含渲染所需的变换矩阵和参数。

```cpp
struct UniformBufferObject {
    glm::mat4 model;        // 模型矩阵
    glm::mat4 view;         // 视图矩阵
    glm::mat4 proj;         // 投影矩阵
    glm::mat4 normalMatrix; // 法向量矩阵
    glm::vec3 lightPos;     // 光源位置
    glm::vec3 viewPos;      // 观察者位置
    float time;             // 时间参数
};
```

### OrbitPoint

轨道点数据结构。

```cpp
struct OrbitPoint {
    glm::vec3 position; // 位置 (米)
    glm::vec3 velocity; // 速度 (米/秒)
    double timestamp;   // 时间戳 (秒)
};
```

### SatelliteState

卫星状态数据结构。

```cpp
struct SatelliteState {
    glm::vec3 position; // 位置 (米)
    glm::vec3 velocity; // 速度 (米/秒)
    glm::vec3 attitude; // 姿态 (欧拉角，弧度)
    double timestamp;   // 时间戳 (秒)
};
```

### OrbitalElements

轨道根数结构。

```cpp
struct OrbitalElements {
    double semiMajorAxis;              // 半长轴 (米)
    double eccentricity;               // 偏心率
    double inclination;                // 轨道倾角 (弧度)
    double argumentOfPerigee;          // 近地点幅角 (弧度)
    double longitudeOfAscendingNode;   // 升交点赤经 (弧度)
    double trueAnomaly;                // 真近点角 (弧度)
};
```

### PropagationParams

轨道传播参数。

```cpp
struct PropagationParams {
    double startTime;   // 开始时间 (秒)
    double endTime;     // 结束时间 (秒)
    double timeStep;    // 时间步长 (秒)
};
```

## VulkanRenderer

Vulkan 渲染器主类，负责 Vulkan 系统的初始化和管理。

### 构造函数

```cpp
VulkanRenderer();
```

创建 VulkanRenderer 实例。

### 析构函数

```cpp
~VulkanRenderer();
```

自动清理所有 Vulkan 资源。

### 公共方法

#### initialize

```cpp
VisualizationError initialize(GLFWwindow* window);
```

初始化 Vulkan 渲染系统。

**参数：**
- `window`: GLFW 窗口指针

**返回值：**
- `VisualizationError`: 错误代码，SUCCESS 表示成功

**异常：**
- 如果 Vulkan 初始化失败，返回相应错误代码

#### beginFrame

```cpp
VisualizationError beginFrame();
```

开始渲染帧，准备命令缓冲区。

**返回值：**
- `VisualizationError`: 错误代码

#### endFrame

```cpp
VisualizationError endFrame();
```

结束渲染帧，提交命令缓冲区并呈现图像。

**返回值：**
- `VisualizationError`: 错误代码

#### waitIdle

```cpp
void waitIdle();
```

等待设备完成所有操作。

#### cleanup

```cpp
void cleanup();
```

清理所有 Vulkan 资源。

#### createBuffer

```cpp
VisualizationError createBuffer(VkDeviceSize size, 
                               VkBufferUsageFlags usage,
                               VkMemoryPropertyFlags properties,
                               VulkanBuffer& buffer);
```

创建 Vulkan 缓冲区。

**参数：**
- `size`: 缓冲区大小（字节）
- `usage`: 缓冲区用途标志
- `properties`: 内存属性标志
- `buffer`: 输出的缓冲区对象

**返回值：**
- `VisualizationError`: 错误代码

#### copyBuffer

```cpp
VisualizationError copyBuffer(VkBuffer srcBuffer, 
                             VkBuffer dstBuffer, 
                             VkDeviceSize size);
```

复制缓冲区数据。

**参数：**
- `srcBuffer`: 源缓冲区
- `dstBuffer`: 目标缓冲区
- `size`: 复制大小（字节）

**返回值：**
- `VisualizationError`: 错误代码

### 访问器方法

```cpp
VkDevice getDevice() const;
VkPhysicalDevice getPhysicalDevice() const;
VkCommandPool getCommandPool() const;
VkQueue getGraphicsQueue() const;
VkRenderPass getRenderPass() const;
VkExtent2D getSwapChainExtent() const;
VkCommandBuffer getCurrentCommandBuffer() const;
uint32_t getCurrentFrame() const;
```

## EarthRenderer

地球渲染器，负责地球球体的渲染。

### 构造函数

```cpp
EarthRenderer();
```

### 析构函数

```cpp
~EarthRenderer();
```

### 公共方法

#### initialize

```cpp
VisualizationError initialize(VulkanRenderer& renderer);
```

初始化地球渲染器。

**参数：**
- `renderer`: Vulkan 渲染器引用

**返回值：**
- `VisualizationError`: 错误代码

#### render

```cpp
VisualizationError render(const EarthRenderParams& params);
```

渲染地球。

**参数：**
- `params`: 地球渲染参数

**返回值：**
- `VisualizationError`: 错误代码

#### updateParams

```cpp
VisualizationError updateParams(const EarthRenderParams& params);
```

更新地球渲染参数。

**参数：**
- `params`: 新的渲染参数

**返回值：**
- `VisualizationError`: 错误代码

#### cleanup

```cpp
void cleanup();
```

清理地球渲染器资源。

## OrbitRenderer

轨道渲染器，负责卫星轨道和卫星的渲染。

### 构造函数

```cpp
OrbitRenderer();
```

### 析构函数

```cpp
~OrbitRenderer();
```

### 公共方法

#### initialize

```cpp
VisualizationError initialize(VulkanRenderer& renderer);
```

初始化轨道渲染器。

**参数：**
- `renderer`: Vulkan 渲染器引用

**返回值：**
- `VisualizationError`: 错误代码

#### render

```cpp
VisualizationError render(const glm::mat4& view, 
                         const glm::mat4& proj,
                         const glm::vec3& cameraPos, 
                         float time);
```

渲染所有轨道和卫星。

**参数：**
- `view`: 视图矩阵
- `proj`: 投影矩阵
- `cameraPos`: 相机位置
- `time`: 当前时间

**返回值：**
- `VisualizationError`: 错误代码

#### addOrbit

```cpp
uint32_t addOrbit(const std::vector<OrbitPoint>& points, 
                  const glm::vec3& color, 
                  bool visible = true);
```

添加轨道。

**参数：**
- `points`: 轨道点数组
- `color`: 轨道颜色
- `visible`: 是否可见

**返回值：**
- `uint32_t`: 轨道 ID

#### addSatellite

```cpp
uint32_t addSatellite(const SatelliteState& state, 
                      const glm::vec3& color, 
                      float size = 1.0f, 
                      bool visible = true);
```

添加卫星。

**参数：**
- `state`: 卫星状态
- `color`: 卫星颜色
- `size`: 卫星大小
- `visible`: 是否可见

**返回值：**
- `uint32_t`: 卫星 ID

#### updateOrbit

```cpp
VisualizationError updateOrbit(uint32_t orbitId, 
                              const std::vector<OrbitPoint>& points);
```

更新轨道数据。

**参数：**
- `orbitId`: 轨道 ID
- `points`: 新的轨道点数组

**返回值：**
- `VisualizationError`: 错误代码

#### updateSatellite

```cpp
VisualizationError updateSatellite(uint32_t satelliteId, 
                                  const SatelliteState& state);
```

更新卫星状态。

**参数：**
- `satelliteId`: 卫星 ID
- `state`: 新的卫星状态

**返回值：**
- `VisualizationError`: 错误代码

#### removeOrbit

```cpp
VisualizationError removeOrbit(uint32_t orbitId);
```

移除轨道。

**参数：**
- `orbitId`: 轨道 ID

**返回值：**
- `VisualizationError`: 错误代码

#### removeSatellite

```cpp
VisualizationError removeSatellite(uint32_t satelliteId);
```

移除卫星。

**参数：**
- `satelliteId`: 卫星 ID

**返回值：**
- `VisualizationError`: 错误代码

#### setOrbitVisible

```cpp
VisualizationError setOrbitVisible(uint32_t orbitId, bool visible);
```

设置轨道可见性。

**参数：**
- `orbitId`: 轨道 ID
- `visible`: 是否可见

**返回值：**
- `VisualizationError`: 错误代码

#### setSatelliteVisible

```cpp
VisualizationError setSatelliteVisible(uint32_t satelliteId, bool visible);
```

设置卫星可见性。

**参数：**
- `satelliteId`: 卫星 ID
- `visible`: 是否可见

**返回值：**
- `VisualizationError`: 错误代码

#### setOrbitColor

```cpp
VisualizationError setOrbitColor(uint32_t orbitId, const glm::vec3& color);
```

设置轨道颜色。

**参数：**
- `orbitId`: 轨道 ID
- `color`: 新颜色

**返回值：**
- `VisualizationError`: 错误代码

#### setSatelliteColor

```cpp
VisualizationError setSatelliteColor(uint32_t satelliteId, const glm::vec3& color);
```

设置卫星颜色。

**参数：**
- `satelliteId`: 卫星 ID
- `color`: 新颜色

**返回值：**
- `VisualizationError`: 错误代码

#### clearAll

```cpp
void clearAll();
```

清除所有轨道和卫星。

#### cleanup

```cpp
void cleanup();
```

清理轨道渲染器资源。

## J2OrbitPropagator

J2 摄动轨道传播器。

### 构造函数

```cpp
J2OrbitPropagator();
```

### 公共方法

#### propagateFromElements

```cpp
PropagationResult propagateFromElements(const OrbitalElements& elements,
                                       const PropagationParams& params,
                                       ProgressCallback callback = nullptr);
```

从轨道根数传播轨道。

**参数：**
- `elements`: 初始轨道根数
- `params`: 传播参数
- `callback`: 进度回调函数（可选）

**返回值：**
- `PropagationResult`: 传播结果

#### propagateFromStateVector

```cpp
PropagationResult propagateFromStateVector(const glm::vec3& position,
                                          const glm::vec3& velocity,
                                          const PropagationParams& params,
                                          ProgressCallback callback = nullptr);
```

从状态向量传播轨道。

**参数：**
- `position`: 初始位置 (米)
- `velocity`: 初始速度 (米/秒)
- `params`: 传播参数
- `callback`: 进度回调函数（可选）

**返回值：**
- `PropagationResult`: 传播结果

#### validateElements

```cpp
bool validateElements(const OrbitalElements& elements);
```

验证轨道根数的有效性。

**参数：**
- `elements`: 待验证的轨道根数

**返回值：**
- `bool`: 是否有效

#### validateParams

```cpp
bool validateParams(const PropagationParams& params);
```

验证传播参数的有效性。

**参数：**
- `params`: 待验证的传播参数

**返回值：**
- `bool`: 是否有效

## OrbitVisualizationManager

轨道可视化管理器，整合轨道传播和可视化功能。

### 构造函数

```cpp
OrbitVisualizationManager(EarthRenderer& earthRenderer, 
                         OrbitRenderer& orbitRenderer);
```

**参数：**
- `earthRenderer`: 地球渲染器引用
- `orbitRenderer`: 轨道渲染器引用

### 析构函数

```cpp
~OrbitVisualizationManager();
```

### 公共方法

#### addOrbitPropagation

```cpp
uint32_t addOrbitPropagation(const OrbitalElements& elements,
                            const PropagationParams& params,
                            const glm::vec3& orbitColor,
                            const glm::vec3& satelliteColor,
                            ProgressCallback callback = nullptr);
```

添加轨道传播任务。

**参数：**
- `elements`: 轨道根数
- `params`: 传播参数
- `orbitColor`: 轨道颜色
- `satelliteColor`: 卫星颜色
- `callback`: 进度回调函数（可选）

**返回值：**
- `uint32_t`: 任务 ID

#### updateOrbitPropagation

```cpp
VisualizationError updateOrbitPropagation(uint32_t taskId,
                                         const PropagationParams& newParams,
                                         ProgressCallback callback = nullptr);
```

更新轨道传播任务。

**参数：**
- `taskId`: 任务 ID
- `newParams`: 新的传播参数
- `callback`: 进度回调函数（可选）

**返回值：**
- `VisualizationError`: 错误代码

#### removeOrbitPropagation

```cpp
VisualizationError removeOrbitPropagation(uint32_t taskId);
```

移除轨道传播任务。

**参数：**
- `taskId`: 任务 ID

**返回值：**
- `VisualizationError`: 错误代码

#### setOrbitVisible

```cpp
VisualizationError setOrbitVisible(uint32_t taskId, bool visible);
```

设置轨道可见性。

**参数：**
- `taskId`: 任务 ID
- `visible`: 是否可见

**返回值：**
- `VisualizationError`: 错误代码

#### clearAllOrbits

```cpp
void clearAllOrbits();
```

清除所有轨道。

## 错误处理

### VisualizationError 枚举

```cpp
enum class VisualizationError {
    SUCCESS = 0,
    VULKAN_INIT_FAILED,
    DEVICE_NOT_SUITABLE,
    SWAPCHAIN_CREATION_FAILED,
    SHADER_COMPILATION_FAILED,
    BUFFER_CREATION_FAILED,
    MEMORY_ALLOCATION_FAILED,
    COMMAND_BUFFER_CREATION_FAILED,
    RENDER_PASS_CREATION_FAILED,
    PIPELINE_CREATION_FAILED,
    DESCRIPTOR_SET_CREATION_FAILED,
    TEXTURE_LOADING_FAILED,
    INVALID_PARAMETERS,
    RESOURCE_NOT_FOUND,
    OPERATION_FAILED
};
```

### 错误转换函数

```cpp
std::string errorToString(VisualizationError error);
```

将错误代码转换为可读字符串。

**参数：**
- `error`: 错误代码

**返回值：**
- `std::string`: 错误描述字符串

## 回调函数

### ProgressCallback

```cpp
using ProgressCallback = std::function<void(float progress, const std::string& status)>;
```

进度回调函数类型，用于报告长时间运行操作的进度。

**参数：**
- `progress`: 进度百分比 (0.0 - 1.0)
- `status`: 状态描述字符串

### 使用示例

```cpp
auto progressCallback = [](float progress, const std::string& status) {
    std::cout << "Progress: " << (progress * 100.0f) << "% - " << status << std::endl;
};

uint32_t taskId = manager.addOrbitPropagation(elements, params, 
                                             orbitColor, satelliteColor, 
                                             progressCallback);
```

## 常量定义

### 地球参数

```cpp
namespace EarthConstants {
    constexpr double RADIUS = 6378137.0;           // 地球半径 (米)
    constexpr double FLATTENING = 1.0 / 298.257;   // 地球扁率
    constexpr double J2 = 1.08262668e-3;           // J2 摄动系数
    constexpr double MU = 3.986004418e14;          // 地球引力参数 (m³/s²)
    constexpr double OMEGA = 7.2921159e-5;         // 地球自转角速度 (rad/s)
}
```

### 渲染常量

```cpp
namespace RenderConstants {
    constexpr uint32_t MAX_FRAMES_IN_FLIGHT = 2;   // 最大并行帧数
    constexpr uint32_t MAX_ORBITS = 100;           // 最大轨道数
    constexpr uint32_t MAX_SATELLITES = 1000;      // 最大卫星数
    constexpr float DEFAULT_POINT_SIZE = 2.0f;     // 默认点大小
    constexpr float DEFAULT_LINE_WIDTH = 1.0f;     // 默认线宽
}
```

## 性能提示

### 最佳实践

1. **批量操作**：尽量批量添加/更新轨道，避免频繁的单个操作
2. **内存管理**：及时清理不需要的轨道和卫星
3. **LOD 控制**：根据距离调整轨道点密度
4. **可见性剔除**：隐藏不需要显示的轨道

### 性能监控

```cpp
// 获取渲染统计信息
RenderStats stats = vulkanRenderer.getRenderStats();
std::cout << "Frame time: " << stats.frameTime << " ms" << std::endl;
std::cout << "Draw calls: " << stats.drawCalls << std::endl;
std::cout << "Vertices: " << stats.vertexCount << std::endl;
```

## 线程安全

### 注意事项

- 所有渲染相关操作必须在主线程中执行
- 轨道传播计算可以在后台线程中进行
- 使用进度回调时注意线程同步

### 多线程示例

```cpp
// 在后台线程中进行轨道计算
std::thread propagationThread([&]() {
    auto result = propagator.propagateFromElements(elements, params, 
        [](float progress, const std::string& status) {
            // 线程安全的进度报告
            std::lock_guard<std::mutex> lock(progressMutex);
            currentProgress = progress;
            currentStatus = status;
        });
    
    // 在主线程中更新可视化
    std::lock_guard<std::mutex> lock(resultMutex);
    pendingResults.push_back(result);
});
```