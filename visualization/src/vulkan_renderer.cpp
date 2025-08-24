#include "vulkan_renderer.h"
#include <iostream>
#include <stdexcept>
#include <algorithm>
#include <set>
#include <fstream>
#include <cstring>
#include <array>
#include <chrono>

#ifdef NDEBUG
const bool enableValidationLayers = false;
#else
const bool enableValidationLayers = true;
#endif

const std::vector<const char*> validationLayers = {
    "VK_LAYER_KHRONOS_validation"
};

const std::vector<const char*> deviceExtensions = {
    VK_KHR_SWAPCHAIN_EXTENSION_NAME
};

// Vulkan 扩展函数代理
VkResult CreateDebugUtilsMessengerEXT(VkInstance instance, const VkDebugUtilsMessengerCreateInfoEXT* pCreateInfo,
                                     const VkAllocationCallbacks* pAllocator, VkDebugUtilsMessengerEXT* pDebugMessenger) {
    auto func = (PFN_vkCreateDebugUtilsMessengerEXT) vkGetInstanceProcAddr(instance, "vkCreateDebugUtilsMessengerEXT");
    if (func != nullptr) {
        return func(instance, pCreateInfo, pAllocator, pDebugMessenger);
    } else {
        return VK_ERROR_EXTENSION_NOT_PRESENT;
    }
}

void DestroyDebugUtilsMessengerEXT(VkInstance instance, VkDebugUtilsMessengerEXT debugMessenger,
                                  const VkAllocationCallbacks* pAllocator) {
    auto func = (PFN_vkDestroyDebugUtilsMessengerEXT) vkGetInstanceProcAddr(instance, "vkDestroyDebugUtilsMessengerEXT");
    if (func != nullptr) {
        func(instance, debugMessenger, pAllocator);
    }
}

namespace j2_orbit_visualization {

/**
 * @brief 构造函数
 * @param width 窗口宽度
 * @param height 窗口高度
 * @param title 窗口标题
 */
VulkanRenderer::VulkanRenderer(uint32_t width, uint32_t height, const std::string& title)
    : windowWidth(width), windowHeight(height), windowTitle(title), window(nullptr), ownsWindow(true) {
    renderStats = {};
    startTime = std::chrono::steady_clock::now();
}

/**
 * @brief 构造函数（使用现有窗口）
 * @param existingWindow 现有的GLFW窗口句柄
 * @param title 窗口标题（可选，用于调试）
 */
VulkanRenderer::VulkanRenderer(GLFWwindow* existingWindow, const std::string& title)
    : window(existingWindow), windowTitle(title), ownsWindow(false) {
    if (existingWindow) {
        int width, height;
        glfwGetFramebufferSize(existingWindow, &width, &height);
        windowWidth = static_cast<uint32_t>(width);
        windowHeight = static_cast<uint32_t>(height);
        
        // 设置用户指针和回调
        glfwSetWindowUserPointer(window, this);
        glfwSetFramebufferSizeCallback(window, framebufferResizeCallback);
    } else {
        windowWidth = 800;
        windowHeight = 600;
    }
    renderStats = {};
    startTime = std::chrono::steady_clock::now();
}

/**
 * @brief 析构函数
 * 清理所有 Vulkan 资源
 */
VulkanRenderer::~VulkanRenderer() {
    cleanup();
}

/**
 * @brief 初始化 Vulkan 渲染器
 * @return VisualizationError 初始化结果
 */
VisualizationError VulkanRenderer::initialize() {
    auto result = initWindow();
    if (result != VisualizationError::Success) {
        return result;
    }
    
    result = initVulkan();
    if (result != VisualizationError::Success) {
        return result;
    }
    
    return VisualizationError::Success;
}

/**
 * @brief 开始渲染帧
 * @return bool 如果可以继续渲染返回 true
 */
bool VulkanRenderer::beginFrame() {
    glfwPollEvents();
    
    if (glfwWindowShouldClose(window)) {
        return false;
    }
    
    // 等待上一帧完成
    vkWaitForFences(device, 1, &inFlightFences[currentFrame], VK_TRUE, UINT64_MAX);
    
    // 获取交换链图像
    VkResult result = vkAcquireNextImageKHR(device, swapChain, UINT64_MAX,
                                           imageAvailableSemaphores[currentFrame], VK_NULL_HANDLE, &imageIndex);
    
    if (result == VK_ERROR_OUT_OF_DATE_KHR) {
        recreateSwapChain();
        return true;
    } else if (result != VK_SUCCESS && result != VK_SUBOPTIMAL_KHR) {
        throw std::runtime_error("Failed to acquire swap chain image!");
    }
    
    // 重置围栏
    vkResetFences(device, 1, &inFlightFences[currentFrame]);
    
    // 重置命令缓冲区
    vkResetCommandBuffer(commandBuffers[currentFrame], 0);
    
    // 开始记录命令缓冲区
    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = 0;
    beginInfo.pInheritanceInfo = nullptr;
    
    if (vkBeginCommandBuffer(commandBuffers[currentFrame], &beginInfo) != VK_SUCCESS) {
        throw std::runtime_error("Failed to begin recording command buffer!");
    }
    
    // 开始渲染通道
    VkRenderPassBeginInfo renderPassInfo{};
    renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
    renderPassInfo.renderPass = renderPass;
    renderPassInfo.framebuffer = swapChainFramebuffers[imageIndex];
    renderPassInfo.renderArea.offset = {0, 0};
    renderPassInfo.renderArea.extent = swapChainExtent;
    
    std::array<VkClearValue, 2> clearValues{};
    clearValues[0].color = {{0.0f, 0.0f, 0.0f, 1.0f}}; // 黑色背景
    clearValues[1].depthStencil = {1.0f, 0};
    
    renderPassInfo.clearValueCount = static_cast<uint32_t>(clearValues.size());
    renderPassInfo.pClearValues = clearValues.data();
    
    vkCmdBeginRenderPass(commandBuffers[currentFrame], &renderPassInfo, VK_SUBPASS_CONTENTS_INLINE);
    
    return true;
}

/**
 * @brief 结束渲染帧并呈现
 * @return VisualizationError 呈现结果
 */
VisualizationError VulkanRenderer::endFrame() {
    // 结束渲染通道
    vkCmdEndRenderPass(commandBuffers[currentFrame]);
    
    // 结束命令缓冲区记录
    if (vkEndCommandBuffer(commandBuffers[currentFrame]) != VK_SUCCESS) {
        return VisualizationError::VulkanError;
    }
    
    // 提交命令缓冲区
    VkSubmitInfo submitInfo{};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    
    VkSemaphore waitSemaphores[] = {imageAvailableSemaphores[currentFrame]};
    VkPipelineStageFlags waitStages[] = {VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT};
    submitInfo.waitSemaphoreCount = 1;
    submitInfo.pWaitSemaphores = waitSemaphores;
    submitInfo.pWaitDstStageMask = waitStages;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &commandBuffers[currentFrame];
    
    VkSemaphore signalSemaphores[] = {renderFinishedSemaphores[currentFrame]};
    submitInfo.signalSemaphoreCount = 1;
    submitInfo.pSignalSemaphores = signalSemaphores;
    
    if (vkQueueSubmit(graphicsQueue, 1, &submitInfo, inFlightFences[currentFrame]) != VK_SUCCESS) {
        return VisualizationError::VulkanError;
    }
    
    // 呈现结果
    VkPresentInfoKHR presentInfo{};
    presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
    presentInfo.waitSemaphoreCount = 1;
    presentInfo.pWaitSemaphores = signalSemaphores;
    
    VkSwapchainKHR swapChains[] = {swapChain};
    presentInfo.swapchainCount = 1;
    presentInfo.pSwapchains = swapChains;
    presentInfo.pImageIndices = &imageIndex;
    presentInfo.pResults = nullptr;
    
    VkResult result = vkQueuePresentKHR(presentQueue, &presentInfo);
    
    if (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR || framebufferResized) {
        framebufferResized = false;
        recreateSwapChain();
    } else if (result != VK_SUCCESS) {
        return VisualizationError::VulkanError;
    }
    
    currentFrame = (currentFrame + 1) % MAX_FRAMES_IN_FLIGHT;
    renderStats.frameCount++;
    
    return VisualizationError::Success;
}

/**
 * @brief 检查窗口是否应该关闭
 * @return bool 如果应该关闭返回 true
 */
bool VulkanRenderer::shouldClose() const {
    return glfwWindowShouldClose(window);
}

/**
 * @brief 等待设备空闲
 */
void VulkanRenderer::waitIdle() {
    vkDeviceWaitIdle(device);
}

/**
 * @brief 获取当前命令缓冲区
 * @return VkCommandBuffer 当前命令缓冲区
 */
VkCommandBuffer VulkanRenderer::getCurrentCommandBuffer() const {
    return commandBuffers[currentFrame];
}

/**
 * @brief 设置窗口大小改变回调
 * @param callback 回调函数
 */
void VulkanRenderer::setResizeCallback(std::function<void(int, int)> callback) {
    resizeCallback = callback;
}

/**
 * @brief 获取自渲染器启动以来经过的时间
 * @return float 经过的时间（秒）
 */
float VulkanRenderer::getElapsedTime() const {
    auto currentTime = std::chrono::steady_clock::now();
    std::chrono::duration<float> duration = currentTime - startTime;
    return duration.count();
}

/**
 * @brief 创建缓冲区
 * @param size 缓冲区大小
 * @param usage 使用标志
 * @param properties 内存属性
 * @param buffer 输出缓冲区结构
 * @return VisualizationError 创建结果
 */
VisualizationError VulkanRenderer::createBuffer(VkDeviceSize size, VkBufferUsageFlags usage,
                                               VkMemoryPropertyFlags properties, VulkanBuffer& buffer) {
    VkBufferCreateInfo bufferInfo{};
    bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufferInfo.size = size;
    bufferInfo.usage = usage;
    bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    
    if (vkCreateBuffer(device, &bufferInfo, nullptr, &buffer.buffer) != VK_SUCCESS) {
        return VisualizationError::VulkanError;
    }
    
    VkMemoryRequirements memRequirements;
    vkGetBufferMemoryRequirements(device, buffer.buffer, &memRequirements);
    
    VkMemoryAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfo.allocationSize = memRequirements.size;
    allocInfo.memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits, properties);
    
    if (vkAllocateMemory(device, &allocInfo, nullptr, &buffer.memory) != VK_SUCCESS) {
        vkDestroyBuffer(device, buffer.buffer, nullptr);
        return VisualizationError::OutOfMemory;
    }
    
    vkBindBufferMemory(device, buffer.buffer, buffer.memory, 0);
    buffer.size = size;
    
    return VisualizationError::Success;
}

/**
 * @brief 复制缓冲区数据
 * @param srcBuffer 源缓冲区
 * @param dstBuffer 目标缓冲区
 * @param size 复制大小
 * @return VisualizationError 复制结果
 */
VisualizationError VulkanRenderer::copyBuffer(VkBuffer srcBuffer, VkBuffer dstBuffer, VkDeviceSize size) {
    VkCommandBufferAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocInfo.commandPool = commandPool;
    allocInfo.commandBufferCount = 1;
    
    VkCommandBuffer commandBuffer;
    vkAllocateCommandBuffers(device, &allocInfo, &commandBuffer);
    
    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    
    vkBeginCommandBuffer(commandBuffer, &beginInfo);
    
    VkBufferCopy copyRegion{};
    copyRegion.size = size;
    vkCmdCopyBuffer(commandBuffer, srcBuffer, dstBuffer, 1, &copyRegion);
    
    vkEndCommandBuffer(commandBuffer);
    
    VkSubmitInfo submitInfo{};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &commandBuffer;
    
    vkQueueSubmit(graphicsQueue, 1, &submitInfo, VK_NULL_HANDLE);
    vkQueueWaitIdle(graphicsQueue);
    
    vkFreeCommandBuffers(device, commandPool, 1, &commandBuffer);
    
    return VisualizationError::Success;
}

// 私有方法实现

/**
 * @brief 初始化窗口
 * @return VisualizationError 初始化结果
 */
VisualizationError VulkanRenderer::initWindow() {
    // 如果窗口已存在（通过构造函数传入），则跳过创建步骤
    if (window != nullptr) {
        return VisualizationError::Success;
    }
    
    glfwInit();
    
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    glfwWindowHint(GLFW_RESIZABLE, GLFW_TRUE);
    
    window = glfwCreateWindow(windowWidth, windowHeight, windowTitle.c_str(), nullptr, nullptr);
    if (!window) {
        glfwTerminate();
        return VisualizationError::InitializationFailed;
    }
    
    glfwSetWindowUserPointer(window, this);
    glfwSetFramebufferSizeCallback(window, framebufferResizeCallback);
    
    return VisualizationError::Success;
}

/**
 * @brief 初始化 Vulkan
 * @return VisualizationError 初始化结果
 */
VisualizationError VulkanRenderer::initVulkan() {
    auto result = createInstance();
    if (result != VisualizationError::SUCCESS) return result;
    
    result = setupDebugMessenger();
    if (result != VisualizationError::SUCCESS) return result;
    
    result = createSurface();
    if (result != VisualizationError::SUCCESS) return result;
    
    result = pickPhysicalDevice();
    if (result != VisualizationError::SUCCESS) return result;
    
    result = createLogicalDevice();
    if (result != VisualizationError::SUCCESS) return result;
    
    result = createSwapChain();
    if (result != VisualizationError::SUCCESS) return result;
    
    result = createImageViews();
    if (result != VisualizationError::SUCCESS) return result;
    
    result = createRenderPass();
    if (result != VisualizationError::SUCCESS) return result;
    
    result = createDepthResources();
    if (result != VisualizationError::SUCCESS) return result;
    
    result = createFramebuffers();
    if (result != VisualizationError::SUCCESS) return result;
    
    result = createCommandPool();
    if (result != VisualizationError::SUCCESS) return result;
    
    result = createCommandBuffers();
    if (result != VisualizationError::SUCCESS) return result;
    
    result = createSyncObjects();
    if (result != VisualizationError::SUCCESS) return result;
    
    return VisualizationError::SUCCESS;
}

/**
 * @brief 创建 Vulkan 实例
 * @return VisualizationError 创建结果
 */
VisualizationError VulkanRenderer::createInstance() {
    if (enableValidationLayers && !checkValidationLayerSupport()) {
        return VisualizationError::VULKAN_INIT_FAILED;
    }
    
    VkApplicationInfo appInfo{};
    appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    appInfo.pApplicationName = "J2 Orbit Visualization";
    appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
    appInfo.pEngineName = "J2 Orbit Engine";
    appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
    appInfo.apiVersion = VK_API_VERSION_1_0;
    
    VkInstanceCreateInfo createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    createInfo.pApplicationInfo = &appInfo;
    
    auto extensions = getRequiredExtensions();
    createInfo.enabledExtensionCount = static_cast<uint32_t>(extensions.size());
    createInfo.ppEnabledExtensionNames = extensions.data();
    
    VkDebugUtilsMessengerCreateInfoEXT debugCreateInfo{};
    if (enableValidationLayers) {
        createInfo.enabledLayerCount = static_cast<uint32_t>(validationLayers.size());
        createInfo.ppEnabledLayerNames = validationLayers.data();
        
        populateDebugMessengerCreateInfo(debugCreateInfo);
        createInfo.pNext = (VkDebugUtilsMessengerCreateInfoEXT*) &debugCreateInfo;
    } else {
        createInfo.enabledLayerCount = 0;
        createInfo.pNext = nullptr;
    }
    
    if (vkCreateInstance(&createInfo, nullptr, &instance) != VK_SUCCESS) {
        return VisualizationError::VULKAN_INIT_FAILED;
    }
    
    return VisualizationError::SUCCESS;
}

/**
 * @brief 设置调试信使
 * @return VisualizationError 设置结果
 */
VisualizationError VulkanRenderer::setupDebugMessenger() {
    if (!enableValidationLayers) return VisualizationError::SUCCESS;
    
    VkDebugUtilsMessengerCreateInfoEXT createInfo;
    populateDebugMessengerCreateInfo(createInfo);
    
    if (CreateDebugUtilsMessengerEXT(instance, &createInfo, nullptr, &debugMessenger) != VK_SUCCESS) {
        return VisualizationError::VULKAN_INIT_FAILED;
    }
    
    return VisualizationError::SUCCESS;
}

/**
 * @brief 创建表面
 * @return VisualizationError 创建结果
 */
VisualizationError VulkanRenderer::createSurface() {
    if (glfwCreateWindowSurface(instance, window, nullptr, &surface) != VK_SUCCESS) {
        return VisualizationError::VULKAN_INIT_FAILED;
    }
    return VisualizationError::SUCCESS;
}

/**
 * @brief 选择物理设备
 * @return VisualizationError 选择结果
 */
VisualizationError VulkanRenderer::pickPhysicalDevice() {
    uint32_t deviceCount = 0;
    vkEnumeratePhysicalDevices(instance, &deviceCount, nullptr);
    
    if (deviceCount == 0) {
        return VisualizationError::VULKAN_INIT_FAILED;
    }
    
    std::vector<VkPhysicalDevice> devices(deviceCount);
    vkEnumeratePhysicalDevices(instance, &deviceCount, devices.data());
    
    for (const auto& device : devices) {
        if (isDeviceSuitable(device)) {
            physicalDevice = device;
            break;
        }
    }
    
    if (physicalDevice == VK_NULL_HANDLE) {
        return VisualizationError::VULKAN_INIT_FAILED;
    }
    
    return VisualizationError::SUCCESS;
}

/**
 * @brief 创建逻辑设备
 * @return VisualizationError 创建结果
 */
VisualizationError VulkanRenderer::createLogicalDevice() {
    QueueFamilyIndices indices = findQueueFamilies(physicalDevice);
    
    std::vector<VkDeviceQueueCreateInfo> queueCreateInfos;
    std::set<uint32_t> uniqueQueueFamilies = {indices.graphicsFamily.value(), indices.presentFamily.value()};
    
    float queuePriority = 1.0f;
    for (uint32_t queueFamily : uniqueQueueFamilies) {
        VkDeviceQueueCreateInfo queueCreateInfo{};
        queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
        queueCreateInfo.queueFamilyIndex = queueFamily;
        queueCreateInfo.queueCount = 1;
        queueCreateInfo.pQueuePriorities = &queuePriority;
        queueCreateInfos.push_back(queueCreateInfo);
    }
    
    VkPhysicalDeviceFeatures deviceFeatures{};
    
    VkDeviceCreateInfo createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    createInfo.queueCreateInfoCount = static_cast<uint32_t>(queueCreateInfos.size());
    createInfo.pQueueCreateInfos = queueCreateInfos.data();
    createInfo.pEnabledFeatures = &deviceFeatures;
    createInfo.enabledExtensionCount = static_cast<uint32_t>(deviceExtensions.size());
    createInfo.ppEnabledExtensionNames = deviceExtensions.data();
    
    if (enableValidationLayers) {
        createInfo.enabledLayerCount = static_cast<uint32_t>(validationLayers.size());
        createInfo.ppEnabledLayerNames = validationLayers.data();
    } else {
        createInfo.enabledLayerCount = 0;
    }
    
    if (vkCreateDevice(physicalDevice, &createInfo, nullptr, &device) != VK_SUCCESS) {
        return VisualizationError::VULKAN_INIT_FAILED;
    }
    
    vkGetDeviceQueue(device, indices.graphicsFamily.value(), 0, &graphicsQueue);
    vkGetDeviceQueue(device, indices.presentFamily.value(), 0, &presentQueue);
    
    return VisualizationError::SUCCESS;
}

/**
 * @brief 创建交换链
 * @return VisualizationError 创建结果
 */
VisualizationError VulkanRenderer::createSwapChain() {
    SwapChainSupportDetails swapChainSupport = querySwapChainSupport(physicalDevice);
    
    VkSurfaceFormatKHR surfaceFormat = chooseSwapSurfaceFormat(swapChainSupport.formats);
    VkPresentModeKHR presentMode = chooseSwapPresentMode(swapChainSupport.presentModes);
    VkExtent2D extent = chooseSwapExtent(swapChainSupport.capabilities);
    
    uint32_t imageCount = swapChainSupport.capabilities.minImageCount + 1;
    if (swapChainSupport.capabilities.maxImageCount > 0 && imageCount > swapChainSupport.capabilities.maxImageCount) {
        imageCount = swapChainSupport.capabilities.maxImageCount;
    }
    
    VkSwapchainCreateInfoKHR createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
    createInfo.surface = surface;
    createInfo.minImageCount = imageCount;
    createInfo.imageFormat = surfaceFormat.format;
    createInfo.imageColorSpace = surfaceFormat.colorSpace;
    createInfo.imageExtent = extent;
    createInfo.imageArrayLayers = 1;
    createInfo.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
    
    QueueFamilyIndices indices = findQueueFamilies(physicalDevice);
    uint32_t queueFamilyIndices[] = {indices.graphicsFamily.value(), indices.presentFamily.value()};
    
    if (indices.graphicsFamily != indices.presentFamily) {
        createInfo.imageSharingMode = VK_SHARING_MODE_CONCURRENT;
        createInfo.queueFamilyIndexCount = 2;
        createInfo.pQueueFamilyIndices = queueFamilyIndices;
    } else {
        createInfo.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
        createInfo.queueFamilyIndexCount = 0;
        createInfo.pQueueFamilyIndices = nullptr;
    }
    
    createInfo.preTransform = swapChainSupport.capabilities.currentTransform;
    createInfo.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
    createInfo.presentMode = presentMode;
    createInfo.clipped = VK_TRUE;
    createInfo.oldSwapchain = VK_NULL_HANDLE;
    
    if (vkCreateSwapchainKHR(device, &createInfo, nullptr, &swapChain) != VK_SUCCESS) {
        return VisualizationError::VULKAN_INIT_FAILED;
    }
    
    vkGetSwapchainImagesKHR(device, swapChain, &imageCount, nullptr);
    swapChainImages.resize(imageCount);
    vkGetSwapchainImagesKHR(device, swapChain, &imageCount, swapChainImages.data());
    
    swapChainImageFormat = surfaceFormat.format;
    swapChainExtent = extent;
    
    return VisualizationError::SUCCESS;
}

/**
 * @brief 创建图像视图
 * @return VisualizationError 创建结果
 */
VisualizationError VulkanRenderer::createImageViews() {
    swapChainImageViews.resize(swapChainImages.size());
    
    for (size_t i = 0; i < swapChainImages.size(); i++) {
        swapChainImageViews[i] = createImageView(swapChainImages[i], swapChainImageFormat, VK_IMAGE_ASPECT_COLOR_BIT);
        if (swapChainImageViews[i] == VK_NULL_HANDLE) {
            return VisualizationError::VULKAN_INIT_FAILED;
        }
    }
    
    return VisualizationError::SUCCESS;
}

/**
 * @brief 查询交换链支持详情
 * @param device 物理设备
 * @return SwapChainSupportDetails 交换链支持详情
 */
SwapChainSupportDetails VulkanRenderer::querySwapChainSupport(VkPhysicalDevice device) {
    SwapChainSupportDetails details;
    
    vkGetPhysicalDeviceSurfaceCapabilitiesKHR(device, surface, &details.capabilities);
    
    uint32_t formatCount;
    vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, &formatCount, nullptr);
    if (formatCount != 0) {
        details.formats.resize(formatCount);
        vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, &formatCount, details.formats.data());
    }
    
    uint32_t presentModeCount;
    vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface, &presentModeCount, nullptr);
    if (presentModeCount != 0) {
        details.presentModes.resize(presentModeCount);
        vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface, &presentModeCount, details.presentModes.data());
    }
    
    return details;
}

/**
 * @brief 选择交换链表面格式
 * @param availableFormats 可用格式列表
 * @return VkSurfaceFormatKHR 选择的格式
 */
VkSurfaceFormatKHR VulkanRenderer::chooseSwapSurfaceFormat(const std::vector<VkSurfaceFormatKHR>& availableFormats) {
    for (const auto& availableFormat : availableFormats) {
        if (availableFormat.format == VK_FORMAT_B8G8R8A8_SRGB && availableFormat.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) {
            return availableFormat;
        }
    }
    return availableFormats[0];
}

/**
 * @brief 选择交换链呈现模式
 * @param availablePresentModes 可用呈现模式列表
 * @return VkPresentModeKHR 选择的呈现模式
 */
VkPresentModeKHR VulkanRenderer::chooseSwapPresentMode(const std::vector<VkPresentModeKHR>& availablePresentModes) {
    for (const auto& availablePresentMode : availablePresentModes) {
        if (availablePresentMode == VK_PRESENT_MODE_MAILBOX_KHR) {
            return availablePresentMode;
        }
    }
    return VK_PRESENT_MODE_FIFO_KHR;
}

/**
 * @brief 选择交换链范围
 * @param capabilities 表面能力
 * @return VkExtent2D 选择的范围
 */
VkExtent2D VulkanRenderer::chooseSwapExtent(const VkSurfaceCapabilitiesKHR& capabilities) {
    if (capabilities.currentExtent.width != std::numeric_limits<uint32_t>::max()) {
        return capabilities.currentExtent;
    } else {
        int width, height;
        glfwGetFramebufferSize(window, &width, &height);
        
        VkExtent2D actualExtent = {
            static_cast<uint32_t>(width),
            static_cast<uint32_t>(height)
        };
        
        actualExtent.width = std::clamp(actualExtent.width, capabilities.minImageExtent.width, capabilities.maxImageExtent.width);
        actualExtent.height = std::clamp(actualExtent.height, capabilities.minImageExtent.height, capabilities.maxImageExtent.height);
        
        return actualExtent;
    }
}

/**
 * @brief 创建图像视图
 * @param image 图像
 * @param format 格式
 * @param aspectFlags 方面标志
 * @return VkImageView 创建的图像视图
 */
VkImageView VulkanRenderer::createImageView(VkImage image, VkFormat format, VkImageAspectFlags aspectFlags) {
    VkImageViewCreateInfo viewInfo{};
    viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    viewInfo.image = image;
    viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
    viewInfo.format = format;
    viewInfo.subresourceRange.aspectMask = aspectFlags;
    viewInfo.subresourceRange.baseMipLevel = 0;
    viewInfo.subresourceRange.levelCount = 1;
    viewInfo.subresourceRange.baseArrayLayer = 0;
    viewInfo.subresourceRange.layerCount = 1;
    
    VkImageView imageView;
    if (vkCreateImageView(device, &viewInfo, nullptr, &imageView) != VK_SUCCESS) {
        return VK_NULL_HANDLE;
    }
    
    return imageView;
}

/**
 * @brief 静态回调函数：窗口大小改变
 * @param window GLFW 窗口句柄
 * @param width 新宽度
 * @param height 新高度
 */
void VulkanRenderer::framebufferResizeCallback(GLFWwindow* window, int width, int height) {
    auto renderer = reinterpret_cast<VulkanRenderer*>(glfwGetWindowUserPointer(window));
    renderer->framebufferResized = true;
    
    if (renderer->resizeCallback) {
        renderer->resizeCallback(width, height);
    }
}

/**
 * @brief 清理资源
 * @note 修复 GLFW 错误 65537：确保在销毁 GLFW 之前完全清理所有 Vulkan 资源
 */
void VulkanRenderer::cleanup() {
    // 首先等待设备空闲，确保所有操作完成
    if (device != VK_NULL_HANDLE) {
        vkDeviceWaitIdle(device);
    }
    
    // 清理交换链相关资源
    cleanupSwapChain();
    
    // 清理深度资源
    if (device != VK_NULL_HANDLE && depthImage.image != VK_NULL_HANDLE) {
        if (depthImage.view != VK_NULL_HANDLE) {
            vkDestroyImageView(device, depthImage.view, nullptr);
            depthImage.view = VK_NULL_HANDLE;
        }
        if (depthImage.image != VK_NULL_HANDLE) {
            vkDestroyImage(device, depthImage.image, nullptr);
            depthImage.image = VK_NULL_HANDLE;
        }
        if (depthImage.memory != VK_NULL_HANDLE) {
            vkFreeMemory(device, depthImage.memory, nullptr);
            depthImage.memory = VK_NULL_HANDLE;
        }
    }
    
    // 清理同步对象
    if (device != VK_NULL_HANDLE) {
        for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
            if (renderFinishedSemaphores[i] != VK_NULL_HANDLE) {
                vkDestroySemaphore(device, renderFinishedSemaphores[i], nullptr);
                renderFinishedSemaphores[i] = VK_NULL_HANDLE;
            }
            if (imageAvailableSemaphores[i] != VK_NULL_HANDLE) {
                vkDestroySemaphore(device, imageAvailableSemaphores[i], nullptr);
                imageAvailableSemaphores[i] = VK_NULL_HANDLE;
            }
            if (inFlightFences[i] != VK_NULL_HANDLE) {
                vkDestroyFence(device, inFlightFences[i], nullptr);
                inFlightFences[i] = VK_NULL_HANDLE;
            }
        }
    }
    
    // 清理渲染通道
    if (device != VK_NULL_HANDLE && renderPass != VK_NULL_HANDLE) {
        vkDestroyRenderPass(device, renderPass, nullptr);
        renderPass = VK_NULL_HANDLE;
    }
    
    // 清理命令池
    if (device != VK_NULL_HANDLE && commandPool != VK_NULL_HANDLE) {
        vkDestroyCommandPool(device, commandPool, nullptr);
        commandPool = VK_NULL_HANDLE;
    }
    
    // 清理调试信使（在设备销毁之前，避免验证层回调访问已销毁的资源）
    if (enableValidationLayers && debugMessenger != VK_NULL_HANDLE && instance != VK_NULL_HANDLE) {
        DestroyDebugUtilsMessengerEXT(instance, debugMessenger, nullptr);
        debugMessenger = VK_NULL_HANDLE;
    }
    
    // 清理逻辑设备
    if (device != VK_NULL_HANDLE) {
        vkDestroyDevice(device, nullptr);
        device = VK_NULL_HANDLE;
    }
    
    // 清理表面
    if (surface != VK_NULL_HANDLE && instance != VK_NULL_HANDLE) {
        vkDestroySurfaceKHR(instance, surface, nullptr);
        surface = VK_NULL_HANDLE;
    }
    
    // 清理 Vulkan 实例
    if (instance != VK_NULL_HANDLE) {
        vkDestroyInstance(instance, nullptr);
        instance = VK_NULL_HANDLE;
    }
    
    // 最后清理 GLFW（在所有 Vulkan 资源清理完成后）
    if (window != nullptr && ownsWindow) {
        // 清除所有 GLFW 回调，避免在窗口销毁后被调用
        glfwSetFramebufferSizeCallback(window, nullptr);
        glfwSetWindowUserPointer(window, nullptr);
        
        glfwDestroyWindow(window);
        window = nullptr;
        
        // 只有拥有窗口所有权时才终止 GLFW
        glfwTerminate();
    } else if (window != nullptr) {
        // 如果不拥有窗口所有权，只清除回调但不销毁窗口
        glfwSetFramebufferSizeCallback(window, nullptr);
        glfwSetWindowUserPointer(window, nullptr);
        window = nullptr;
    }
}

/**
 * @brief 从文件读取二进制数据（用于读取 SPIR-V 着色器）
 * @param filename 文件路径（建议使用绝对路径或相对于工作目录的路径）
 * @return std::vector<char> 文件字节内容；若读取失败返回空向量
 * @note 本函数不抛出异常，调用者可通过返回向量是否为空判断失败
 */
std::vector<char> VulkanRenderer::readFile(const std::string& filename) const {
    std::ifstream file(filename, std::ios::ate | std::ios::binary);
    if (!file.is_open()) {
        return {};
    }

    size_t fileSize = static_cast<size_t>(file.tellg());
    std::vector<char> buffer(fileSize);
    file.seekg(0);
    file.read(buffer.data(), fileSize);
    file.close();
    return buffer;
}

/**
 * @brief 由 SPIR-V 字节码创建 VkShaderModule
 * @param code SPIR-V 二进制字节码
 * @return VkShaderModule 成功返回有效句柄，失败返回 VK_NULL_HANDLE
 * @note 创建成功后，调用者负责使用 vkDestroyShaderModule 释放资源
 */
VkShaderModule VulkanRenderer::createShaderModule(const std::vector<char>& code) const {
    VkShaderModuleCreateInfo createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    createInfo.codeSize = code.size();
    createInfo.pCode = reinterpret_cast<const uint32_t*>(code.data());

    VkShaderModule shaderModule = VK_NULL_HANDLE;
    if (vkCreateShaderModule(device, &createInfo, nullptr, &shaderModule) != VK_SUCCESS) {
        return VK_NULL_HANDLE;
    }
    return shaderModule;
}

/**
 * @brief 转换图像布局
 * @param image 图像句柄
 * @param format 图像格式
 * @param oldLayout 旧布局
 * @param newLayout 新布局
 * @return VisualizationError 转换结果
 */
VisualizationError VulkanRenderer::transitionImageLayout(VkImage image, VkFormat format,
                                                        VkImageLayout oldLayout, VkImageLayout newLayout) {
    VkCommandBuffer commandBuffer = beginSingleTimeCommands();
    if (commandBuffer == VK_NULL_HANDLE) {
        return VisualizationError::VulkanError;
    }

    VkImageMemoryBarrier barrier{};
    barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    barrier.oldLayout = oldLayout;
    barrier.newLayout = newLayout;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.image = image;
    barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    barrier.subresourceRange.baseMipLevel = 0;
    barrier.subresourceRange.levelCount = 1;
    barrier.subresourceRange.baseArrayLayer = 0;
    barrier.subresourceRange.layerCount = 1;

    VkPipelineStageFlags sourceStage;
    VkPipelineStageFlags destinationStage;

    if (oldLayout == VK_IMAGE_LAYOUT_UNDEFINED && newLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL) {
        barrier.srcAccessMask = 0;
        barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        sourceStage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
        destinationStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
    } else if (oldLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL && newLayout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL) {
        barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        sourceStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
        destinationStage = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
    } else {
        return VisualizationError::VulkanError;
    }

    vkCmdPipelineBarrier(commandBuffer, sourceStage, destinationStage, 0, 0, nullptr, 0, nullptr, 1, &barrier);

    endSingleTimeCommands(commandBuffer);
    return VisualizationError::Success;
}

/**
 * @brief 复制缓冲区到图像
 * @param buffer 源缓冲区
 * @param image 目标图像
 * @param width 图像宽度
 * @param height 图像高度
 * @return VisualizationError 复制结果
 */
VisualizationError VulkanRenderer::copyBufferToImage(VkBuffer buffer, VkImage image,
                                                    uint32_t width, uint32_t height) {
    VkCommandBuffer commandBuffer = beginSingleTimeCommands();
    if (commandBuffer == VK_NULL_HANDLE) {
        return VisualizationError::VulkanError;
    }

    VkBufferImageCopy region{};
    region.bufferOffset = 0;
    region.bufferRowLength = 0;
    region.bufferImageHeight = 0;
    region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    region.imageSubresource.mipLevel = 0;
    region.imageSubresource.baseArrayLayer = 0;
    region.imageSubresource.layerCount = 1;
    region.imageOffset = {0, 0, 0};
    region.imageExtent = {width, height, 1};

    vkCmdCopyBufferToImage(commandBuffer, buffer, image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);

    endSingleTimeCommands(commandBuffer);
    return VisualizationError::Success;
}

/**
 * @brief 开始单次命令缓冲区
 * @return VkCommandBuffer 命令缓冲区句柄
 */
VkCommandBuffer VulkanRenderer::beginSingleTimeCommands() {
    VkCommandBufferAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocInfo.commandPool = commandPool;
    allocInfo.commandBufferCount = 1;

    VkCommandBuffer commandBuffer;
    if (vkAllocateCommandBuffers(device, &allocInfo, &commandBuffer) != VK_SUCCESS) {
        return VK_NULL_HANDLE;
    }

    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

    if (vkBeginCommandBuffer(commandBuffer, &beginInfo) != VK_SUCCESS) {
        vkFreeCommandBuffers(device, commandPool, 1, &commandBuffer);
        return VK_NULL_HANDLE;
    }

    return commandBuffer;
}

/**
 * @brief 结束单次命令缓冲区
 * @param commandBuffer 命令缓冲区句柄
 */
void VulkanRenderer::endSingleTimeCommands(VkCommandBuffer commandBuffer) {
    vkEndCommandBuffer(commandBuffer);

    VkSubmitInfo submitInfo{};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &commandBuffer;

    vkQueueSubmit(graphicsQueue, 1, &submitInfo, VK_NULL_HANDLE);
    vkQueueWaitIdle(graphicsQueue);

    vkFreeCommandBuffers(device, commandPool, 1, &commandBuffer);
}

/**
 * @brief 重新创建交换链
 * @return VisualizationError 创建结果
 */
VisualizationError VulkanRenderer::recreateSwapChain() {
    int width = 0, height = 0;
    glfwGetFramebufferSize(window, &width, &height);
    while (width == 0 || height == 0) {
        glfwGetFramebufferSize(window, &width, &height);
        glfwWaitEvents();
    }

    vkDeviceWaitIdle(device);

    cleanupSwapChain();

    auto result = createSwapChain();
    if (result != VisualizationError::SUCCESS) return result;
    
    result = createImageViews();
    if (result != VisualizationError::SUCCESS) return result;
    
    result = createDepthResources();
    if (result != VisualizationError::SUCCESS) return result;
    
    result = createFramebuffers();
    if (result != VisualizationError::SUCCESS) return result;

    return VisualizationError::SUCCESS;
}

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
VisualizationError VulkanRenderer::createImage(uint32_t width, uint32_t height, VkFormat format,
                                              VkImageTiling tiling, VkImageUsageFlags usage,
                                              VkMemoryPropertyFlags properties, VulkanImage& image) {
    VkImageCreateInfo imageInfo{};
    imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    imageInfo.imageType = VK_IMAGE_TYPE_2D;
    imageInfo.extent.width = width;
    imageInfo.extent.height = height;
    imageInfo.extent.depth = 1;
    imageInfo.mipLevels = 1;
    imageInfo.arrayLayers = 1;
    imageInfo.format = format;
    imageInfo.tiling = tiling;
    imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    imageInfo.usage = usage;
    imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
    imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    if (vkCreateImage(device, &imageInfo, nullptr, &image.image) != VK_SUCCESS) {
        return VisualizationError::VULKAN_INIT_FAILED;
    }

    VkMemoryRequirements memRequirements;
    vkGetImageMemoryRequirements(device, image.image, &memRequirements);

    VkMemoryAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfo.allocationSize = memRequirements.size;
    allocInfo.memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits, properties);

    if (vkAllocateMemory(device, &allocInfo, nullptr, &image.memory) != VK_SUCCESS) {
        vkDestroyImage(device, image.image, nullptr);
        return VisualizationError::VULKAN_INIT_FAILED;
    }

    vkBindImageMemory(device, image.image, image.memory, 0);
    return VisualizationError::SUCCESS;
}

/**
 * @brief 检查验证层支持
 * @return bool 是否支持所需的验证层
 */
bool VulkanRenderer::checkValidationLayerSupport() {
    uint32_t layerCount;
    vkEnumerateInstanceLayerProperties(&layerCount, nullptr);

    std::vector<VkLayerProperties> availableLayers(layerCount);
    vkEnumerateInstanceLayerProperties(&layerCount, availableLayers.data());

    for (const char* layerName : validationLayers) {
        bool layerFound = false;

        for (const auto& layerProperties : availableLayers) {
            if (strcmp(layerName, layerProperties.layerName) == 0) {
                layerFound = true;
                break;
            }
        }

        if (!layerFound) {
            return false;
        }
    }

    return true;
}

/**
 * @brief 获取所需的扩展
 * @return std::vector<const char*> 扩展名称列表
 */
std::vector<const char*> VulkanRenderer::getRequiredExtensions() {
    uint32_t glfwExtensionCount = 0;
    const char** glfwExtensions;
    glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);

    std::vector<const char*> extensions(glfwExtensions, glfwExtensions + glfwExtensionCount);

    if (enableValidationLayers) {
        extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
    }

    return extensions;
}

/**
 * @brief 检查设备是否适合
 * @param device 物理设备
 * @return bool 设备是否适合
 */
bool VulkanRenderer::isDeviceSuitable(VkPhysicalDevice device) {
    QueueFamilyIndices indices = findQueueFamilies(device);
    
    bool extensionsSupported = checkDeviceExtensionSupport(device);
    
    bool swapChainAdequate = false;
    if (extensionsSupported) {
        SwapChainSupportDetails swapChainSupport = querySwapChainSupport(device);
        swapChainAdequate = !swapChainSupport.formats.empty() && !swapChainSupport.presentModes.empty();
    }
    
    VkPhysicalDeviceFeatures supportedFeatures;
    vkGetPhysicalDeviceFeatures(device, &supportedFeatures);
    
    return indices.isComplete() && extensionsSupported && swapChainAdequate && supportedFeatures.samplerAnisotropy;
}

/**
 * @brief 检查设备扩展支持
 * @param device 物理设备
 * @return bool 是否支持所需扩展
 */
bool VulkanRenderer::checkDeviceExtensionSupport(VkPhysicalDevice device) {
    uint32_t extensionCount;
    vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount, nullptr);
    
    std::vector<VkExtensionProperties> availableExtensions(extensionCount);
    vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount, availableExtensions.data());
    
    std::set<std::string> requiredExtensions(deviceExtensions.begin(), deviceExtensions.end());
    
    for (const auto& extension : availableExtensions) {
        requiredExtensions.erase(extension.extensionName);
    }
    
    return requiredExtensions.empty();
}

/**
 * @brief 填充调试信使创建信息
 * @param createInfo 调试信使创建信息结构
 */
void VulkanRenderer::populateDebugMessengerCreateInfo(VkDebugUtilsMessengerCreateInfoEXT& createInfo) {
    createInfo = {};
    createInfo.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
    createInfo.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT |
                                VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
                                VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
    createInfo.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
                            VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
                            VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
    createInfo.pfnUserCallback = debugCallback;
}

/**
 * @brief 查找合适的内存类型
 * @param typeFilter 内存类型过滤器
 * @param properties 内存属性
 * @return uint32_t 内存类型索引
 */
uint32_t VulkanRenderer::findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties) {
    VkPhysicalDeviceMemoryProperties memProperties;
    vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memProperties);

    for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++) {
        if ((typeFilter & (1 << i)) && (memProperties.memoryTypes[i].propertyFlags & properties) == properties) {
            return i;
        }
    }

    throw std::runtime_error("failed to find suitable memory type!");
}

/**
 * @brief 查找队列族
 * @param physicalDev 物理设备
 * @return QueueFamilyIndices 队列族索引
 */
QueueFamilyIndices VulkanRenderer::findQueueFamilies(VkPhysicalDevice physicalDev) {
    QueueFamilyIndices indices;
    
    uint32_t queueFamilyCount = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(physicalDev, &queueFamilyCount, nullptr);
    
    std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
    vkGetPhysicalDeviceQueueFamilyProperties(physicalDev, &queueFamilyCount, queueFamilies.data());
    
    int i = 0;
    for (const auto& queueFamily : queueFamilies) {
        if (queueFamily.queueFlags & VK_QUEUE_GRAPHICS_BIT) {
            indices.graphicsFamily = i;
        }
        
        VkBool32 presentSupport = false;
        vkGetPhysicalDeviceSurfaceSupportKHR(physicalDev, i, surface, &presentSupport);
        
        if (presentSupport) {
            indices.presentFamily = i;
        }
        
        if (indices.isComplete()) {
            break;
        }
        
        i++;
    }
    
    return indices;
}

/**
 * @brief 查找深度格式
 * @return VkFormat 深度格式
 */
VkFormat VulkanRenderer::findDepthFormat() {
    return findSupportedFormat(
        {VK_FORMAT_D32_SFLOAT, VK_FORMAT_D32_SFLOAT_S8_UINT, VK_FORMAT_D24_UNORM_S8_UINT},
        VK_IMAGE_TILING_OPTIMAL,
        VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT
    );
}

/**
 * @brief 查找支持的格式
 * @param candidates 候选格式列表
 * @param tiling 图像平铺方式
 * @param features 所需特性
 * @return VkFormat 支持的格式
 */
VkFormat VulkanRenderer::findSupportedFormat(const std::vector<VkFormat>& candidates, VkImageTiling tiling, VkFormatFeatureFlags features) {
    for (VkFormat format : candidates) {
        VkFormatProperties props;
        vkGetPhysicalDeviceFormatProperties(physicalDevice, format, &props);
        
        if (tiling == VK_IMAGE_TILING_LINEAR && (props.linearTilingFeatures & features) == features) {
            return format;
        } else if (tiling == VK_IMAGE_TILING_OPTIMAL && (props.optimalTilingFeatures & features) == features) {
            return format;
        }
    }
    
    throw std::runtime_error("failed to find supported format!");
}

/**
 * @brief 创建同步对象
 * @return VisualizationError 错误码
 */
VisualizationError VulkanRenderer::createSyncObjects() {
    imageAvailableSemaphores.resize(MAX_FRAMES_IN_FLIGHT);
    renderFinishedSemaphores.resize(MAX_FRAMES_IN_FLIGHT);
    inFlightFences.resize(MAX_FRAMES_IN_FLIGHT);
    
    VkSemaphoreCreateInfo semaphoreInfo{};
    semaphoreInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
    
    VkFenceCreateInfo fenceInfo{};
    fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;
    
    for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
        if (vkCreateSemaphore(device, &semaphoreInfo, nullptr, &imageAvailableSemaphores[i]) != VK_SUCCESS ||
            vkCreateSemaphore(device, &semaphoreInfo, nullptr, &renderFinishedSemaphores[i]) != VK_SUCCESS ||
            vkCreateFence(device, &fenceInfo, nullptr, &inFlightFences[i]) != VK_SUCCESS) {
            return VisualizationError::VULKAN_INIT_FAILED;
        }
    }
    
    return VisualizationError::SUCCESS;
}

/**
 * @brief 初始化 Vulkan


/**
 * @brief 创建渲染通道
 * @return VisualizationError 错误码
 */
VisualizationError VulkanRenderer::createRenderPass() {
    VkAttachmentDescription colorAttachment{};
    colorAttachment.format = swapChainImageFormat;
    colorAttachment.samples = VK_SAMPLE_COUNT_1_BIT;
    colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    colorAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    colorAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    colorAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    colorAttachment.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
    
    VkAttachmentDescription depthAttachment{};
    depthAttachment.format = findDepthFormat();
    depthAttachment.samples = VK_SAMPLE_COUNT_1_BIT;
    depthAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    depthAttachment.storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    depthAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    depthAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    depthAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    depthAttachment.finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
    
    VkAttachmentReference colorAttachmentRef{};
    colorAttachmentRef.attachment = 0;
    colorAttachmentRef.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    
    VkAttachmentReference depthAttachmentRef{};
    depthAttachmentRef.attachment = 1;
    depthAttachmentRef.layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
    
    VkSubpassDescription subpass{};
    subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    subpass.colorAttachmentCount = 1;
    subpass.pColorAttachments = &colorAttachmentRef;
    subpass.pDepthStencilAttachment = &depthAttachmentRef;
    
    VkSubpassDependency dependency{};
    dependency.srcSubpass = VK_SUBPASS_EXTERNAL;
    dependency.dstSubpass = 0;
    dependency.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
    dependency.srcAccessMask = 0;
    dependency.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
    dependency.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
    
    std::array<VkAttachmentDescription, 2> attachments = {colorAttachment, depthAttachment};
    VkRenderPassCreateInfo renderPassInfo{};
    renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
    renderPassInfo.attachmentCount = static_cast<uint32_t>(attachments.size());
    renderPassInfo.pAttachments = attachments.data();
    renderPassInfo.subpassCount = 1;
    renderPassInfo.pSubpasses = &subpass;
    renderPassInfo.dependencyCount = 1;
    renderPassInfo.pDependencies = &dependency;
    
    if (vkCreateRenderPass(device, &renderPassInfo, nullptr, &renderPass) != VK_SUCCESS) {
        return VisualizationError::VULKAN_INIT_FAILED;
    }
    
    return VisualizationError::SUCCESS;
}

/**
 * @brief 创建命令缓冲区
 * @return VisualizationError 错误码
 */
VisualizationError VulkanRenderer::createCommandBuffers() {
    commandBuffers.resize(MAX_FRAMES_IN_FLIGHT);
    
    VkCommandBufferAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocInfo.commandPool = commandPool;
    allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocInfo.commandBufferCount = static_cast<uint32_t>(commandBuffers.size());
    
    if (vkAllocateCommandBuffers(device, &allocInfo, commandBuffers.data()) != VK_SUCCESS) {
        return VisualizationError::COMMAND_BUFFER_FAILED;
    }
    
    return VisualizationError::SUCCESS;
}

/**
 * @brief 调试回调函数
 * @param messageSeverity 消息严重性
 * @param messageType 消息类型
 * @param pCallbackData 回调数据
 * @param pUserData 用户数据
 * @return VkBool32 是否中止调用
 */
VkBool32 VulkanRenderer::debugCallback(VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
                                      VkDebugUtilsMessageTypeFlagsEXT messageType,
                                      const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData,
                                      void* pUserData) {
    std::cerr << "validation layer: " << pCallbackData->pMessage << std::endl;
    return VK_FALSE;
}

/**
 * @brief 创建深度资源
 * @return VisualizationError 创建结果
 */
VisualizationError VulkanRenderer::createDepthResources() {
    VkFormat depthFormat = findDepthFormat();
    
    VisualizationError result = createImage(swapChainExtent.width, swapChainExtent.height, depthFormat,
                                           VK_IMAGE_TILING_OPTIMAL, VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT,
                                           VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, depthImage);
    if (result != VisualizationError::SUCCESS) {
        return result;
    }
    
    depthImage.view = createImageView(depthImage.image, depthFormat, VK_IMAGE_ASPECT_DEPTH_BIT);
    if (depthImage.view == VK_NULL_HANDLE) {
        return VisualizationError::VULKAN_INIT_FAILED;
    }
    
    return VisualizationError::SUCCESS;
}

/**
 * @brief 创建帧缓冲区
 * @return VisualizationError 创建结果
 */
VisualizationError VulkanRenderer::createFramebuffers() {
    swapChainFramebuffers.resize(swapChainImageViews.size());
    
    for (size_t i = 0; i < swapChainImageViews.size(); i++) {
        std::array<VkImageView, 2> attachments = {
            swapChainImageViews[i],
            depthImage.view
        };
        
        VkFramebufferCreateInfo framebufferInfo{};
        framebufferInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
        framebufferInfo.renderPass = renderPass;
        framebufferInfo.attachmentCount = static_cast<uint32_t>(attachments.size());
        framebufferInfo.pAttachments = attachments.data();
        framebufferInfo.width = swapChainExtent.width;
        framebufferInfo.height = swapChainExtent.height;
        framebufferInfo.layers = 1;
        
        if (vkCreateFramebuffer(device, &framebufferInfo, nullptr, &swapChainFramebuffers[i]) != VK_SUCCESS) {
            return VisualizationError::VULKAN_INIT_FAILED;
        }
    }
    
    return VisualizationError::SUCCESS;
}

/**
 * @brief 创建命令池
 * @return VisualizationError 创建结果
 */
VisualizationError VulkanRenderer::createCommandPool() {
    QueueFamilyIndices queueFamilyIndices = findQueueFamilies(physicalDevice);
    
    VkCommandPoolCreateInfo poolInfo{};
    poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    poolInfo.queueFamilyIndex = queueFamilyIndices.graphicsFamily.value();
    
    if (vkCreateCommandPool(device, &poolInfo, nullptr, &commandPool) != VK_SUCCESS) {
        return VisualizationError::VulkanError;
    }
    
    return VisualizationError::Success;
}

/**
 * @brief 清理交换链相关资源
 * @note 在重新创建交换链或程序退出时调用
 */
void VulkanRenderer::cleanupSwapChain() {
    // 清理帧缓冲区
    for (auto framebuffer : swapChainFramebuffers) {
        vkDestroyFramebuffer(device, framebuffer, nullptr);
    }
    swapChainFramebuffers.clear();
    
    // 清理图像视图
    for (auto imageView : swapChainImageViews) {
        vkDestroyImageView(device, imageView, nullptr);
    }
    swapChainImageViews.clear();
    
    // 清理交换链
    if (swapChain != VK_NULL_HANDLE) {
        vkDestroySwapchainKHR(device, swapChain, nullptr);
        swapChain = VK_NULL_HANDLE;
    }
}

} // namespace j2_orbit_visualization