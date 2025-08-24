#pragma once

#include <vulkan/vulkan.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <vector>
#include <memory>
#include <string>

namespace j2_orbit_visualization {

/**
 * @brief 顶点数据结构
 * 包含位置、法线、纹理坐标信息
 */
struct Vertex {
    glm::vec3 position;    ///< 顶点位置
    glm::vec3 normal;      ///< 法线向量
    glm::vec2 texCoord;    ///< 纹理坐标
    
    /**
     * @brief 获取 Vulkan 顶点绑定描述
     * @return VkVertexInputBindingDescription 顶点绑定描述
     */
    static VkVertexInputBindingDescription getBindingDescription();
    
    /**
     * @brief 获取 Vulkan 顶点属性描述
     * @return std::vector<VkVertexInputAttributeDescription> 属性描述数组
     */
    static std::vector<VkVertexInputAttributeDescription> getAttributeDescriptions();
};

/**
 * @brief 地球渲染器的统一缓冲对象
 */
struct EarthUniformBufferObject {
    alignas(16) glm::mat4 model;
    alignas(16) glm::mat4 view;
    alignas(16) glm::mat4 proj;
    alignas(16) glm::mat4 normalMatrix;
    alignas(16) glm::vec3 lightPos;
    alignas(16) glm::vec3 viewPos;
};

/**
 * @brief 轨道渲染器的统一缓冲对象
 */
struct OrbitUniformBufferObject {
    alignas(16) glm::mat4 view;
    alignas(16) glm::mat4 proj;
    alignas(16) glm::vec3 cameraPos;
    alignas(4) float currentTime;
    alignas(4) float orbitAlpha;
    alignas(4) float pointSize;
};

/**
 * @brief [已弃用] 统一缓冲对象数据结构
 * @deprecated 请改用 EarthUniformBufferObject 或 OrbitUniformBufferObject
 */
struct UniformBufferObject {
    glm::mat4 model;       ///< 模型矩阵
    glm::mat4 view;        ///< 视图矩阵
    glm::mat4 proj;        ///< 投影矩阵
    glm::vec3 lightPos;    ///< 光源位置
    float time;            ///< 时间参数
};

/**
 * @brief 轨道点数据结构
 * 用于存储轨道路径上的点
 */
struct OrbitPoint {
    glm::vec3 position;    ///< 轨道点位置
    glm::vec3 color;       ///< 颜色
    float timestamp;       ///< 时间戳

    /**
     * @brief 获取 Vulkan 顶点绑定描述
     * @return VkVertexInputBindingDescription 顶点绑定描述
     */
    static VkVertexInputBindingDescription getBindingDescription();

    /**
     * @brief 获取 Vulkan 顶点属性描述
     * @return std::vector<VkVertexInputAttributeDescription> 属性描述数组
     */
    static std::vector<VkVertexInputAttributeDescription> getAttributeDescriptions();
};

/**
 * @brief 卫星状态数据结构
 * 包含卫星的完整状态信息
 */
struct SatelliteState {
    glm::vec3 position;    ///< 卫星位置 (ECI坐标系)
    glm::vec3 velocity;    ///< 卫星速度
    double timestamp;      ///< 时间戳
    float scale;           ///< 渲染缩放因子
    glm::vec3 color;       ///< 卫星颜色
};

/**
 * @brief 地球渲染参数
 * 控制地球渲染的各种参数
 */
struct EarthRenderParams {
    float radius = 6371.0f;           ///< 地球半径 (km)
    int tessellationLevel = 64;       ///< 细分级别
    bool enableAtmosphere = true;     ///< 是否启用大气效果
    bool enableClouds = true;         ///< 是否启用云层
    float rotationSpeed = 0.1f;       ///< 自转速度
    std::string textureFile;          ///< 地球表面纹理文件路径
    std::string cloudTextureFile;     ///< 云层纹理文件路径
};

/**
 * @brief 相机参数结构
 * 控制相机的位置和行为
 */
struct CameraParams {
    glm::vec3 position = glm::vec3(0.0f, 0.0f, 15000.0f);  ///< 相机位置
    glm::vec3 target = glm::vec3(0.0f, 0.0f, 0.0f);       ///< 目标位置
    glm::vec3 up = glm::vec3(0.0f, 1.0f, 0.0f);           ///< 上方向
    float fov = 45.0f;                                     ///< 视野角度
    float nearPlane = 0.1f;                               ///< 近裁剪面
    float farPlane = 100000.0f;                           ///< 远裁剪面
    float moveSpeed = 1000.0f;                            ///< 移动速度
    float rotateSpeed = 0.5f;                             ///< 旋转速度
};

/**
 * @brief 渲染统计信息
 * 用于性能监控和调试
 */
struct RenderStats {
    uint32_t frameCount = 0;           ///< 帧计数
    float frameTime = 0.0f;            ///< 帧时间 (ms)
    float fps = 0.0f;                  ///< 帧率
    uint32_t vertexCount = 0;          ///< 顶点数量
    uint32_t triangleCount = 0;        ///< 三角形数量
    size_t memoryUsage = 0;            ///< 显存使用量 (bytes)
};

/**
 * @brief Vulkan 缓冲区包装器
 * 简化 Vulkan 缓冲区的管理
 */
struct VulkanBuffer {
    VkBuffer buffer = VK_NULL_HANDLE;           ///< Vulkan 缓冲区句柄
    VkDeviceMemory memory = VK_NULL_HANDLE;     ///< 设备内存句柄
    VkDeviceSize size = 0;                      ///< 缓冲区大小
    void* mapped = nullptr;                     ///< 映射的内存指针
    
    /**
     * @brief 清理缓冲区资源
     * @param device Vulkan 逻辑设备
     */
    void cleanup(VkDevice device);
};

/**
 * @brief Vulkan 图像包装器
 * 简化 Vulkan 图像和纹理的管理
 */
struct VulkanImage {
    VkImage image = VK_NULL_HANDLE;             ///< Vulkan 图像句柄
    VkDeviceMemory memory = VK_NULL_HANDLE;     ///< 设备内存句柄
    VkImageView view = VK_NULL_HANDLE;          ///< 图像视图句柄
    VkSampler sampler = VK_NULL_HANDLE;         ///< 采样器句柄
    uint32_t width = 0;                         ///< 图像宽度
    uint32_t height = 0;                        ///< 图像高度
    VkFormat format = VK_FORMAT_UNDEFINED;      ///< 图像格式
    
    /**
     * @brief 清理图像资源
     * @param device Vulkan 逻辑设备
     */
    void cleanup(VkDevice device);
};

/**
 * @brief 错误代码枚举
 * 定义可视化系统的错误类型
 */
enum class VisualizationError {
    SUCCESS = 0,                    ///< 成功
    VULKAN_INIT_FAILED,            ///< Vulkan 初始化失败
    DEVICE_NOT_SUITABLE,           ///< 设备不合适
    SWAPCHAIN_CREATION_FAILED,     ///< 交换链创建失败
    SHADER_COMPILATION_FAILED,     ///< 着色器编译失败
    BUFFER_CREATION_FAILED,        ///< 缓冲区创建失败
    TEXTURE_LOADING_FAILED,        ///< 纹理加载失败
    MEMORY_ALLOCATION_FAILED,      ///< 内存分配失败
    COMMAND_BUFFER_FAILED,         ///< 命令缓冲区失败
    INITIALIZATION_FAILED,         ///< 初始化失败
    OUT_OF_MEMORY,                 ///< 内存不足
    UNKNOWN_ERROR,                 ///< 未知错误
    // 兼容旧代码的别名（保持与现有源码一致）
    Success = SUCCESS,             ///< 别名：成功
    VulkanError = UNKNOWN_ERROR,   ///< 别名：通用 Vulkan 错误
    InitializationFailed = INITIALIZATION_FAILED,  ///< 别名：初始化失败
    OutOfMemory = OUT_OF_MEMORY    ///< 别名：内存不足
};

/**
 * @brief 将错误代码转换为字符串
 * @param error 错误代码
 * @return const char* 错误描述字符串
 */
const char* errorToString(VisualizationError error);

} // namespace j2_orbit_visualization