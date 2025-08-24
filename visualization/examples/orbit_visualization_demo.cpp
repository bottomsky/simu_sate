/**
 * @file orbit_visualization_demo.cpp
 * @brief J2 轨道可视化演示程序
 * 
 * 这个程序展示了如何使用 Vulkan 可视化系统来渲染地球和卫星轨道，
 * 并集成 J2 轨道外推算法进行实时轨道计算和显示。
 * 
 * @author J2 Orbit Visualization Team
 * @date 2024
 */

#include <iostream>
#include <vector>
#include <memory>
#include <chrono>
#include <thread>

// Vulkan 和窗口系统
#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

// 可视化系统头文件
#include "../include/vulkan_renderer.h"
#include "../include/earth_renderer.h"
#include "../include/orbit_renderer.h"
#include "../include/j2_orbit_integration.h"
#include "../include/visualization_types.h"

using namespace j2_orbit_visualization;

// 全局变量
static GLFWwindow* window = nullptr;
static std::unique_ptr<VulkanRenderer> vulkanRenderer;
static std::unique_ptr<EarthRenderer> earthRenderer;
static std::unique_ptr<OrbitRenderer> orbitRenderer;
static std::unique_ptr<OrbitVisualizationManager> visualizationManager;

// 窗口参数
static const int WINDOW_WIDTH = 1280;
static const int WINDOW_HEIGHT = 720;
static const char* WINDOW_TITLE = "J2 Orbit Visualization Demo";

// 相机参数
static CameraParams camera = {
    glm::vec3(0.0f, 0.0f, 4000000.0f),  // 位置：距离地心 4,000 km
    glm::vec3(0.0f, 0.0f, 0.0f),        // 目标：地心
    glm::vec3(0.0f, 1.0f, 0.0f),        // 上方向
    45.0f,                               // 视野角度
    1000.0f,                             // 近裁剪面：调整为1000米以保持深度精度
    120000000.0f                         // 远裁剪面：调整为120,000公里
};

// 轨道相机控制参数
static bool firstMouse = true;
static float lastX = WINDOW_WIDTH / 2.0f;
static float lastY = WINDOW_HEIGHT / 2.0f;
static float mouseSensitivity = 0.5f;  // 鼠标灵敏度
static float keyboardSpeed = 50.0f;   // 键盘旋转速度 (度/秒)
static bool leftMousePressed = false;  // 鼠标左键按下状态

// 轨道相机球坐标参数
static float orbitDistance = 4000000.0f;  // 距离地心的距离 (m)
static float orbitAzimuth = 0.0f;         // 方位角 (度)
static float orbitElevation = 0.0f;       // 仰角 (度)
static const float MIN_DISTANCE = 1000000.0f;   // 最小距离 1000km
static const float MAX_DISTANCE = 100000000.0f; // 最大距离 100,000km
static const float MIN_ELEVATION = -89.0f;      // 最小仰角
static const float MAX_ELEVATION = 89.0f;       // 最大仰角

/**
 * @brief 更新轨道相机位置
 * 根据球坐标参数计算相机位置，确保相机始终看向地球中心
 */
void updateOrbitCamera() {
    // 将角度转换为弧度
    float azimuthRad = glm::radians(orbitAzimuth);
    float elevationRad = glm::radians(orbitElevation);
    
    // 计算相机在球坐标系中的位置
    camera.position.x = orbitDistance * cos(elevationRad) * cos(azimuthRad);
    camera.position.y = orbitDistance * sin(elevationRad);
    camera.position.z = orbitDistance * cos(elevationRad) * sin(azimuthRad);
    
    // 相机始终看向地球中心
    camera.target = glm::vec3(0.0f, 0.0f, 0.0f);
    
    // 设置上方向向量
    camera.up = glm::vec3(0.0f, 1.0f, 0.0f);
}

/**
 * @brief GLFW 错误回调函数
 * @param error 错误代码
 * @param description 错误描述
 */
void glfwErrorCallback(int error, const char* description) {
    std::cerr << "GLFW Error " << error << ": " << description << std::endl;
}

/**
 * @brief 窗口大小改变回调函数
 * @param window GLFW 窗口指针
 * @param width 新宽度
 * @param height 新高度
 */
void framebufferResizeCallback(GLFWwindow* window, int width, int height) {
    if (vulkanRenderer) {
        // 注意：帧缓冲区大小改变将由 VulkanRenderer 内部处理
    }
    
    // 更新相机宽高比
    // aspectRatio 将在渲染循环中计算
}

/**
 * @brief 鼠标按键回调函数
 * @param window GLFW 窗口指针
 * @param button 鼠标按键
 * @param action 按键动作（按下/释放）
 * @param mods 修饰键
 */
void mouseButtonCallback(GLFWwindow* window, int button, int action, int mods) {
    if (button == GLFW_MOUSE_BUTTON_LEFT) {
        if (action == GLFW_PRESS) {
            leftMousePressed = true;
        } else if (action == GLFW_RELEASE) {
            leftMousePressed = false;
        }
    }
}

/**
 * @brief 鼠标移动回调函数
 * @param window GLFW 窗口指针
 * @param xpos 鼠标 X 坐标
 * @param ypos 鼠标 Y 坐标
 */
void mouseCallback(GLFWwindow* window, double xpos, double ypos) {
    if (firstMouse) {
        lastX = static_cast<float>(xpos);
        lastY = static_cast<float>(ypos);
        firstMouse = false;
        return;
    }
    
    // 只有在鼠标左键按下时才更新相机角度
    if (!leftMousePressed) {
        lastX = static_cast<float>(xpos);
        lastY = static_cast<float>(ypos);
        return;
    }
    
    float xoffset = static_cast<float>(xpos) - lastX;
    float yoffset = lastY - static_cast<float>(ypos); // Y 坐标反转
    lastX = static_cast<float>(xpos);
    lastY = static_cast<float>(ypos);
    
    // 应用鼠标灵敏度
    xoffset *= mouseSensitivity;
    yoffset *= mouseSensitivity;
    
    // 更新轨道相机的方位角和仰角
    orbitAzimuth += xoffset;
    orbitElevation += yoffset;
    
    // 限制仰角范围
    orbitElevation = glm::clamp(orbitElevation, MIN_ELEVATION, MAX_ELEVATION);
    
    // 方位角可以无限旋转，但保持在0-360度范围内便于调试
    if (orbitAzimuth > 360.0f) orbitAzimuth -= 360.0f;
    if (orbitAzimuth < 0.0f) orbitAzimuth += 360.0f;
    
    // 更新相机位置
    updateOrbitCamera();
}

/**
 * @brief 鼠标滚轮回调函数
 * @param window GLFW 窗口指针
 * @param xoffset X 方向滚动偏移
 * @param yoffset Y 方向滚动偏移
 */
void scrollCallback(GLFWwindow* window, double xoffset, double yoffset) {
    // 调整轨道相机距离
    float zoomFactor = 0.1f;
    orbitDistance -= static_cast<float>(yoffset) * orbitDistance * zoomFactor;
    
    // 限制距离范围
    orbitDistance = glm::clamp(orbitDistance, MIN_DISTANCE, MAX_DISTANCE);
    
    // 更新相机位置
    updateOrbitCamera();
}

/**
 * @brief 处理键盘输入
 * @param window GLFW 窗口指针
 * @param deltaTime 帧时间间隔
 */
void processInput(GLFWwindow* window, float deltaTime) {
    if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS) {
        glfwSetWindowShouldClose(window, true);
    }
    
    // 轨道相机控制
    float rotationSpeed = keyboardSpeed * deltaTime; // 度/帧
    bool cameraUpdated = false;
    
    // WASD 控制轨道旋转
    if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS) {
        orbitElevation += rotationSpeed;
        cameraUpdated = true;
    }
    if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS) {
        orbitElevation -= rotationSpeed;
        cameraUpdated = true;
    }
    if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS) {
        orbitAzimuth -= rotationSpeed;
        cameraUpdated = true;
    }
    if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS) {
        orbitAzimuth += rotationSpeed;
        cameraUpdated = true;
    }
    
    // QE 控制距离（缩放）
    if (glfwGetKey(window, GLFW_KEY_Q) == GLFW_PRESS) {
        orbitDistance *= (1.0f - 2.0f * deltaTime); // 拉近
        cameraUpdated = true;
    }
    if (glfwGetKey(window, GLFW_KEY_E) == GLFW_PRESS) {
        orbitDistance *= (1.0f + 2.0f * deltaTime); // 拉远
        cameraUpdated = true;
    }
    
    // 应用限制
    if (cameraUpdated) {
        orbitElevation = glm::clamp(orbitElevation, MIN_ELEVATION, MAX_ELEVATION);
        orbitDistance = glm::clamp(orbitDistance, MIN_DISTANCE, MAX_DISTANCE);
        
        // 方位角归一化
        if (orbitAzimuth > 360.0f) orbitAzimuth -= 360.0f;
        if (orbitAzimuth < 0.0f) orbitAzimuth += 360.0f;
        
        // 更新相机位置
        updateOrbitCamera();
    }
}

/**
 * @brief 初始化 GLFW 和窗口
 * @return bool 是否成功
 */
bool initializeWindow() {
    // 设置错误回调
    glfwSetErrorCallback(glfwErrorCallback);
    
    // 初始化 GLFW
    if (!glfwInit()) {
        std::cerr << "Failed to initialize GLFW" << std::endl;
        return false;
    }
    
    // 配置 GLFW
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API); // 不使用 OpenGL
    glfwWindowHint(GLFW_RESIZABLE, GLFW_TRUE);
    
    // 创建窗口
    window = glfwCreateWindow(WINDOW_WIDTH, WINDOW_HEIGHT, WINDOW_TITLE, nullptr, nullptr);
    if (!window) {
        std::cerr << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        return false;
    }
    
    // 设置回调函数
    glfwSetFramebufferSizeCallback(window, framebufferResizeCallback);
    glfwSetCursorPosCallback(window, mouseCallback);
    glfwSetMouseButtonCallback(window, mouseButtonCallback);
    glfwSetScrollCallback(window, scrollCallback);
    
    // 显示鼠标光标
    glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
    
    return true;
}

/**
 * @brief 初始化 Vulkan 渲染系统
 * @return bool 是否成功
 */
bool initializeVulkan() {
    try {
        // 创建 Vulkan 渲染器（使用已存在的窗口）
        vulkanRenderer = std::make_unique<VulkanRenderer>(window, "J2 Orbit Visualization");
        if (vulkanRenderer->initialize() != VisualizationError::SUCCESS) {
            std::cerr << "Failed to initialize Vulkan renderer" << std::endl;
            return false;
        }
        
        // 创建地球渲染器
        earthRenderer = std::make_unique<EarthRenderer>(*vulkanRenderer);
        if (earthRenderer->initialize("assets/8k_earth_daymap.jpg", "assets/8k_earth_clouds.jpg") != VisualizationError::SUCCESS) {
            std::cerr << "Failed to initialize Earth renderer" << std::endl;
            return false;
        }
        
        // 创建轨道渲染器
    orbitRenderer = std::make_unique<OrbitRenderer>(*vulkanRenderer);
        if (orbitRenderer->initialize() != VisualizationError::SUCCESS) {
            std::cerr << "Failed to initialize Orbit renderer" << std::endl;
            return false;
        }
        
        // 创建可视化管理器
        visualizationManager = std::make_unique<OrbitVisualizationManager>(*earthRenderer, *orbitRenderer);
        
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "Exception during Vulkan initialization: " << e.what() << std::endl;
        return false;
    }
}

/**
 * @brief 创建示例轨道
 */
void createExampleOrbits() {
    // 示例轨道 1：低地球轨道 (LEO)
    OrbitalElements leo;
    leo.a = 6778137.0;                  // 半长轴 a (m) - 400 km 高度
    leo.e = 0.001;                      // 偏心率 e [-]
    leo.i = static_cast<double>(glm::radians(51.6f)); // 倾角 i (rad)
    leo.w = 0.0;                        // 近地点幅角 ω (rad)
    leo.O = 0.0;                        // 升交点赤经 Ω (rad)
    leo.M = 0.0;                        // 平近点角 M (rad)
    
    PropagationParams leoParams;
    leoParams.startTime = 0.0;
    leoParams.endTime = 5400.0;         // 1.5 小时
    leoParams.timeStep = 60.0;          // 1 分钟步长
    
    uint32_t leoTaskId = visualizationManager->addOrbitTask("LEO", leo, glm::vec3(0.0f, 1.0f, 0.0f));
    visualizationManager->executeOrbitPropagation(leoTaskId);
    
    // 示例轨道 2：地球同步轨道 (GEO)
    OrbitalElements geo;
    geo.a = 42164169.0;                 // 半长轴 a (m) - GEO 平均半径
    geo.e = 0.0;
    geo.i = 0.0;                        // 赤道轨道
    geo.w = 0.0;
    geo.O = 0.0;
    geo.M = 0.0;
    
    PropagationParams geoParams;
    geoParams.startTime = 0.0;
    geoParams.endTime = 86400.0;        // 24 小时
    geoParams.timeStep = 300.0;         // 5 分钟步长
    
    uint32_t geoTaskId = visualizationManager->addOrbitTask("GEO", geo, glm::vec3(1.0f, 0.0f, 0.0f));
    visualizationManager->executeOrbitPropagation(geoTaskId);
    
    // 示例轨道 3：极地轨道
    OrbitalElements polar;
    polar.a = 7178137.0;                // 半长轴 a (m) - 800 km 高度
    polar.e = 0.01;
    polar.i = static_cast<double>(glm::radians(90.0f)); // 极地轨道
    polar.w = 0.0;
    polar.O = 0.0;
    polar.M = 0.0;
    
    PropagationParams polarParams;
    polarParams.startTime = 0.0;
    polarParams.endTime = 6000.0;       // 约 1.67 小时
    polarParams.timeStep = 60.0;        // 1 分钟步长
    
    uint32_t polarTaskId = visualizationManager->addOrbitTask("Polar", polar, glm::vec3(0.0f, 0.0f, 1.0f));
    visualizationManager->executeOrbitPropagation(polarTaskId);
    
    std::cout << "Created 3 example orbits:" << std::endl;
    std::cout << "  - LEO (Green): 400 km altitude, 51.6° inclination" << std::endl;
    std::cout << "  - GEO (Red): 35,786 km altitude, 0° inclination" << std::endl;
    std::cout << "  - Polar (Blue): 800 km altitude, 90° inclination" << std::endl;
}

/**
 * @brief 主渲染循环
 */
void renderLoop() {
    auto lastTime = std::chrono::high_resolution_clock::now();
    
    while (!glfwWindowShouldClose(window)) {
        // 计算帧时间
        auto currentTime = std::chrono::high_resolution_clock::now();
        float deltaTime = std::chrono::duration<float>(currentTime - lastTime).count();
        lastTime = currentTime;
        
        // 处理事件
        glfwPollEvents();
        
        // 处理输入
        processInput(window, deltaTime);
        
        // 开始渲染帧
        if (vulkanRenderer->beginFrame()) {
            // 更新相机矩阵
        glm::mat4 view = glm::lookAt(camera.position, camera.target, camera.up);
        
        // 计算宽高比
        int width, height;
        glfwGetFramebufferSize(window, &width, &height);
        float aspectRatio = static_cast<float>(width) / static_cast<float>(height);
        
        glm::mat4 proj = glm::perspective(glm::radians(camera.fov), aspectRatio,
                                        camera.nearPlane, camera.farPlane);
        proj[1][1] *= -1; // Vulkan Y 坐标翻转
            
            // 渲染地球
            EarthRenderParams earthParams;
            earthParams.radius = 6371000.0f; // 地球半径：6,371 km = 6,371,000 m
            earthParams.tessellationLevel = 64;
            earthParams.enableAtmosphere = true;
            earthParams.enableClouds = false;
            earthParams.rotationSpeed = 0.1f;
            earthParams.textureFile = "assets/8k_earth_daymap.jpg";
            earthParams.cloudTextureFile = "assets/8k_earth_clouds.jpg";
            
            CameraParams cameraParams;
            cameraParams.position = camera.position;
            cameraParams.target = camera.target;
            cameraParams.up = camera.up;
            cameraParams.fov = camera.fov;
            cameraParams.nearPlane = camera.nearPlane;
            cameraParams.farPlane = camera.farPlane;
            
            // 调试输出（每60帧输出一次，避免过多输出）
            static int frameCount = 0;
            if (frameCount % 60 == 0) {
                std::cout << "[DEBUG] Frame " << frameCount << ":" << std::endl;
                std::cout << "  Camera position: (" << camera.position.x << ", " << camera.position.y << ", " << camera.position.z << ")" << std::endl;
                std::cout << "  Camera distance from origin: " << glm::length(camera.position) << " meters" << std::endl;
                std::cout << "  Earth radius: " << earthParams.radius << " meters" << std::endl;
                std::cout << "  Near/Far planes: " << camera.nearPlane << "/" << camera.farPlane << std::endl;
                std::cout << "  Calling earthRenderer->render()..." << std::endl;
            }
            frameCount++;
            
            earthRenderer->render(vulkanRenderer->getCurrentCommandBuffer(), earthParams, cameraParams);
            
            if (frameCount % 60 == 1) {
                std::cout << "  earthRenderer->render() completed." << std::endl;
            }
            
            // 渲染轨道
            orbitRenderer->render(vulkanRenderer->getCurrentCommandBuffer(), cameraParams);
            
            // 结束渲染帧
            if (vulkanRenderer->endFrame() != VisualizationError::SUCCESS) {
                std::cerr << "Failed to end frame" << std::endl;
            }
        }
        
        // 限制帧率到 60 FPS
        std::this_thread::sleep_for(std::chrono::milliseconds(16));
    }
    
    // 等待设备空闲
    vulkanRenderer->waitIdle();
}

/**
 * @brief 清理资源
 */
void cleanup() {
    visualizationManager.reset();
    orbitRenderer.reset();
    earthRenderer.reset();
    vulkanRenderer.reset();
    
    // 清理 GLFW（因为 VulkanRenderer 不再拥有窗口所有权）
    if (window) {
        glfwDestroyWindow(window);
        window = nullptr;
    }
    glfwTerminate();
}

/**
 * @brief 主函数
 * @param argc 命令行参数个数
 * @param argv 命令行参数数组
 * @return int 程序退出代码
 */
int main(int argc, char* argv[]) {
    std::cout << "J2 Orbit Visualization Demo" << std::endl;
    std::cout << "===========================" << std::endl;
    std::cout << "Controls:" << std::endl;
    std::cout << "  WASD - Move camera" << std::endl;
    std::cout << "  QE - Move up/down" << std::endl;
    std::cout << "  Mouse - Look around" << std::endl;
    std::cout << "  Scroll - Zoom in/out" << std::endl;
    std::cout << "  ESC - Exit" << std::endl;
    std::cout << std::endl;
    
    try {
        // 初始化窗口
        if (!initializeWindow()) {
            std::cerr << "Failed to initialize window" << std::endl;
            return -1;
        }
        
        // 初始化 Vulkan
        if (!initializeVulkan()) {
            std::cerr << "Failed to initialize Vulkan" << std::endl;
            cleanup();
            return -1;
        }
        
        // 创建示例轨道
        createExampleOrbits();
        
        // 初始化轨道相机位置
        updateOrbitCamera();
        
        std::cout << "Initialization complete. Starting render loop..." << std::endl;
        
        // 主渲染循环
        renderLoop();
        
        std::cout << "Render loop ended. Cleaning up..." << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Exception in main: " << e.what() << std::endl;
        cleanup();
        return -1;
    }
    
    // 清理资源
    cleanup();
    
    std::cout << "Program exited successfully." << std::endl;
    return 0;
}