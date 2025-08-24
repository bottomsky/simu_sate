#pragma once

#include "visualization_types.h"
#include <GLFW/glfw3.h>
#include <vulkan/vulkan.h>
#include <vector>
#include <optional>
#include <memory>
#include <functional>
#include <chrono>

namespace j2_orbit_visualization {

/**
 * @brief 队列族索引结构
 * 存储不同类型队列的族索引
 */
struct QueueFamilyIndices {
    std::optional<uint32_t> graphicsFamily;  ///< 图形队列族索引
    std::optional<uint32_t> presentFamily;   ///< 呈现队列族索引
    
    /**
     * @brief 检查是否所有必需的队列族都已找到
     * @return bool 如果完整返回 true
     */
    bool isComplete() const {
        return graphicsFamily.has_value() && presentFamily.has_value();
    }
};

/**
 * @brief 交换链支持详情结构
 * 存储交换链的能力和支持信息
 */
struct SwapChainSupportDetails {
    VkSurfaceCapabilitiesKHR capabilities;        ///< 表面能力
    std::vector<VkSurfaceFormatKHR> formats;      ///< 支持的格式
    std::vector<VkPresentModeKHR> presentModes;   ///< 支持的呈现模式
};

/**
 * @brief Vulkan 渲染器主类
 * 负责 Vulkan 的初始化、渲染管线管理和资源管理
 */
class VulkanRenderer {
public:
    /**
     * @brief 构造函数
     * @param width 窗口宽度
     * @param height 窗口高度
     * @param title 窗口标题
     */
    VulkanRenderer(uint32_t width, uint32_t height, const std::string& title);
    
    /**
     * @brief 构造函数（使用现有窗口）
     * @param existingWindow 现有的GLFW窗口句柄
     * @param title 窗口标题（可选，用于调试）
     */
    VulkanRenderer(GLFWwindow* existingWindow, const std::string& title = "J2 Orbit Visualization");
    
    /**
     * @brief 析构函数
     * 清理所有 Vulkan 资源
     */
    ~VulkanRenderer();
    
    /**
     * @brief 初始化 Vulkan 渲染器
     * @return VisualizationError 初始化结果
     */
    VisualizationError initialize();
    
    /**
     * @brief 开始渲染帧
     * @return bool 如果可以继续渲染返回 true
     */
    bool beginFrame();
    
    /**
     * @brief 结束渲染帧并呈现
     * @return VisualizationError 呈现结果
     */
    VisualizationError endFrame();
    
    /**
     * @brief 检查窗口是否应该关闭
     * @return bool 如果应该关闭返回 true
     */
    bool shouldClose() const;
    
    /**
     * @brief 等待设备空闲
     */
    void waitIdle();
    
    /**
     * @brief 获取当前命令缓冲区
     * @return VkCommandBuffer 当前命令缓冲区
     */
    VkCommandBuffer getCurrentCommandBuffer() const;
    
    /**
     * @brief 获取当前帧索引
     * @return uint32_t 当前帧索引
     */
    uint32_t getCurrentFrameIndex() const { return currentFrame; }
    
    /**
     * @brief 获取交换链图像格式
     * @return VkFormat 交换链图像格式
     */
    VkFormat getSwapChainImageFormat() const { return swapChainImageFormat; }
    
    /**
     * @brief 获取交换链范围
     * @return VkExtent2D 交换链范围
     */
    VkExtent2D getSwapChainExtent() const { return swapChainExtent; }
    
    /**
     * @brief 获取渲染通道
     * @return VkRenderPass 渲染通道句柄
     */
    VkRenderPass getRenderPass() const { return renderPass; }
    
    /**
     * @brief 获取逻辑设备
     * @return VkDevice 逻辑设备句柄
     */
    VkDevice getDevice() const { return device; }
    
    /**
     * @brief 获取物理设备
     * @return VkPhysicalDevice 物理设备句柄
     */
    VkPhysicalDevice getPhysicalDevice() const { return physicalDevice; }
    
    /**
     * @brief 获取图形队列
     * @return VkQueue 图形队列句柄
     */
    VkQueue getGraphicsQueue() const { return graphicsQueue; }
    
    /**
     * @brief 获取呈现队列
     * @return VkQueue 呈现队列句柄
     */
    VkQueue getPresentQueue() const { return presentQueue; }
    
    /**
     * @brief 获取命令池
     * @return VkCommandPool 命令池句柄
     */
    VkCommandPool getCommandPool() const { return commandPool; }
    
    /**
     * @brief 设置窗口大小改变回调
     * @param callback 回调函数
     */
    void setResizeCallback(std::function<void(int, int)> callback);
    
    /**
     * @brief 创建缓冲区
     * @param size 缓冲区大小
     * @param usage 使用标志
     * @param properties 内存属性
     * @param buffer 输出缓冲区结构
     * @return VisualizationError 创建结果
     */
    VisualizationError createBuffer(VkDeviceSize size, VkBufferUsageFlags usage,
                                   VkMemoryPropertyFlags properties, VulkanBuffer& buffer);
    
    /**
     * @brief 复制缓冲区数据
     * @param srcBuffer 源缓冲区
     * @param dstBuffer 目标缓冲区
     * @param size 复制大小
     * @return VisualizationError 复制结果
     */
    VisualizationError copyBuffer(VkBuffer srcBuffer, VkBuffer dstBuffer, VkDeviceSize size);
    
    /**
     * @brief 创建图像
     * @param width 图像宽度
     * @param height 图像高度
     * @param format 图像格式
     * @param tiling 平铺模式
     * @param usage 使用标志
     * @param properties 内存属性
     * @param image 输出图像结构
     * @return VisualizationError 创建结果
     */
    VisualizationError createImage(uint32_t width, uint32_t height, VkFormat format,
                                  VkImageTiling tiling, VkImageUsageFlags usage,
                                  VkMemoryPropertyFlags properties, VulkanImage& image);
    
    /**
     * @brief 转换图像布局
     * @param image 图像句柄
     * @param format 图像格式
     * @param oldLayout 旧布局
     * @param newLayout 新布局
     * @return VisualizationError 转换结果
     */
    VisualizationError transitionImageLayout(VkImage image, VkFormat format,
                                            VkImageLayout oldLayout, VkImageLayout newLayout);
    
    /**
     * @brief 复制缓冲区到图像
     * @param buffer 源缓冲区
     * @param image 目标图像
     * @param width 图像宽度
     * @param height 图像高度
     * @return VisualizationError 复制结果
     */
    VisualizationError copyBufferToImage(VkBuffer buffer, VkImage image,
                                        uint32_t width, uint32_t height);
    
    /**
     * @brief 创建图像视图
     * @param image 图像句柄
     * @param format 图像格式
     * @param aspectFlags 方面标志
     * @return VkImageView 图像视图句柄
     */
    VkImageView createImageView(VkImage image, VkFormat format, VkImageAspectFlags aspectFlags);
    
    /**
     * @brief 获取渲染统计信息
     * @return RenderStats 渲染统计
     */
    RenderStats getRenderStats() const { return renderStats; }

    /**
     * @brief 获取最大飞行中帧数
     * @return int 最大飞行中帧数
     */
    int getMaxFramesInFlight() const { return MAX_FRAMES_IN_FLIGHT; }

    /**
     * @brief 获取自渲染开始以来经过的时间（秒）
     * @return float 经过的时间
     */
    float getElapsedTime() const;

    /**
     * @brief 从文件读取二进制数据（用于读取 SPIR-V 着色器）
     * @param filename 文件路径（建议使用绝对路径或相对于工作目录的路径）
     * @return std::vector<char> 文件字节内容；若读取失败返回空向量
     * @note 本函数不抛出异常，调用者可通过返回向量是否为空判断失败
     */
    std::vector<char> readFile(const std::string& filename) const;

    /**
     * @brief 由 SPIR-V 字节码创建 VkShaderModule
     * @param code SPIR-V 二进制字节码
     * @return VkShaderModule 成功返回有效句柄，失败返回 VK_NULL_HANDLE
     * @note 创建成功后，调用者负责使用 vkDestroyShaderModule 释放资源
     */
    VkShaderModule createShaderModule(const std::vector<char>& code) const;

private:
    // 窗口相关
    GLFWwindow* window;                    ///< GLFW 窗口句柄
    bool ownsWindow; // 标识是否拥有窗口的所有权（是否需要清理GLFW）
    uint32_t windowWidth;                  ///< 窗口宽度
    uint32_t windowHeight;                 ///< 窗口高度
    std::string windowTitle;               ///< 窗口标题
    bool framebufferResized = false;       ///< 帧缓冲区是否已调整大小

    // 时间
    std::chrono::steady_clock::time_point startTime; ///< 渲染开始时间
    
    // Vulkan 核心对象
    VkInstance instance;                   ///< Vulkan 实例
    VkDebugUtilsMessengerEXT debugMessenger; ///< 调试信使
    VkSurfaceKHR surface;                  ///< 窗口表面
    VkPhysicalDevice physicalDevice = VK_NULL_HANDLE; ///< 物理设备
    VkDevice device;                       ///< 逻辑设备
    
    // 队列
    VkQueue graphicsQueue;                 ///< 图形队列
    VkQueue presentQueue;                  ///< 呈现队列
    
    // 交换链
    VkSwapchainKHR swapChain;             ///< 交换链
    std::vector<VkImage> swapChainImages; ///< 交换链图像
    VkFormat swapChainImageFormat;        ///< 交换链图像格式
    VkExtent2D swapChainExtent;           ///< 交换链范围
    std::vector<VkImageView> swapChainImageViews; ///< 交换链图像视图
    std::vector<VkFramebuffer> swapChainFramebuffers; ///< 交换链帧缓冲区
    
    // 渲染通道和管线
    VkRenderPass renderPass;              ///< 渲染通道
    
    // 深度缓冲区
    VulkanImage depthImage;               ///< 深度图像
    
    // 命令相关
    VkCommandPool commandPool;            ///< 命令池
    std::vector<VkCommandBuffer> commandBuffers; ///< 命令缓冲区
    
    // 同步对象
    std::vector<VkSemaphore> imageAvailableSemaphores; ///< 图像可用信号量
    std::vector<VkSemaphore> renderFinishedSemaphores; ///< 渲染完成信号量
    std::vector<VkFence> inFlightFences;  ///< 飞行中围栏
    
    // 帧管理
    static const int MAX_FRAMES_IN_FLIGHT = 2; ///< 最大飞行中帧数
    uint32_t currentFrame = 0;            ///< 当前帧索引
    uint32_t imageIndex = 0;              ///< 当前图像索引
    
    // 统计信息
    RenderStats renderStats;              ///< 渲染统计
    
    // 回调函数
    std::function<void(int, int)> resizeCallback; ///< 窗口大小改变回调
    
    // 初始化方法
    VisualizationError initWindow();
    VisualizationError initVulkan();
    VisualizationError createInstance();
    VisualizationError setupDebugMessenger();
    VisualizationError createSurface();
    VisualizationError pickPhysicalDevice();
    VisualizationError createLogicalDevice();
    VisualizationError createSwapChain();
    VisualizationError createImageViews();
    VisualizationError createRenderPass();
    VisualizationError createDepthResources();
    VisualizationError createFramebuffers();
    VisualizationError createCommandPool();
    VisualizationError createCommandBuffers();
    VisualizationError createSyncObjects();
    
    // 辅助方法
    bool checkValidationLayerSupport();
    std::vector<const char*> getRequiredExtensions();
    void populateDebugMessengerCreateInfo(VkDebugUtilsMessengerCreateInfoEXT& createInfo);
    bool isDeviceSuitable(VkPhysicalDevice device);
    bool checkDeviceExtensionSupport(VkPhysicalDevice device);
    QueueFamilyIndices findQueueFamilies(VkPhysicalDevice device);
    SwapChainSupportDetails querySwapChainSupport(VkPhysicalDevice device);
    VkSurfaceFormatKHR chooseSwapSurfaceFormat(const std::vector<VkSurfaceFormatKHR>& availableFormats);
    VkPresentModeKHR chooseSwapPresentMode(const std::vector<VkPresentModeKHR>& availablePresentModes);
    VkExtent2D chooseSwapExtent(const VkSurfaceCapabilitiesKHR& capabilities);
    VkFormat findSupportedFormat(const std::vector<VkFormat>& candidates, VkImageTiling tiling, VkFormatFeatureFlags features);
    VkFormat findDepthFormat();
    uint32_t findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties);
    VkCommandBuffer beginSingleTimeCommands();
    void endSingleTimeCommands(VkCommandBuffer commandBuffer);
    
    // 清理方法
    void cleanup();
    void cleanupSwapChain();
    VisualizationError recreateSwapChain();
    
    // 静态回调函数
    static void framebufferResizeCallback(GLFWwindow* window, int width, int height);
    static VkBool32 debugCallback(VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
                                  VkDebugUtilsMessageTypeFlagsEXT messageType,
                                  const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData,
                                  void* pUserData);
};

} // namespace j2_orbit_visualization