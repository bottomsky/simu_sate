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

#define _USE_MATH_DEFINES
#include <cmath>
#include <iostream>
#include <vector>
#include <memory>
#include <chrono>
#include <thread>
#include <map>

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

// 轨道根数存储（用于调试输出）
struct OrbitDebugInfo {
    std::string name;
    OrbitalElements elements;
    glm::vec3 color;
    uint32_t taskId;
};
static std::vector<OrbitDebugInfo> orbitDebugInfos;

// 窗口参数
static const int WINDOW_WIDTH = 1280;
static const int WINDOW_HEIGHT = 720;
static const char* WINDOW_TITLE = "J2 Orbit Visualization Demo";

// 相机参数
static CameraParams camera = {
    glm::vec3(0.0f, 0.0f, 15000.0f),    // 位置：距离地心 15,000 km (千米单位)
    glm::vec3(0.0f, 0.0f, 0.0f),        // 目标：地心
    glm::vec3(0.0f, 1.0f, 0.0f),        // 上方向
    45.0f,                               // 视野角度
    1.0f,                                // 近裁剪面：调整为1千米以保持深度精度
    120000.0f                            // 远裁剪面：调整为120,000千米
};

// 轨道相机控制参数
static bool firstMouse = true;
static float lastX = WINDOW_WIDTH / 2.0f;
static float lastY = WINDOW_HEIGHT / 2.0f;
static float mouseSensitivity = 0.5f;  // 鼠标灵敏度
static float keyboardSpeed = 50.0f;   // 键盘旋转速度 (度/秒)
static bool leftMousePressed = false;  // 鼠标左键按下状态

// 轨道相机球坐标参数
static float orbitDistance = 15000.0f;    // 距离地心的距离 (km)
static float orbitAzimuth = 0.0f;         // 方位角 (度)
static float orbitElevation = 0.0f;       // 仰角 (度)
static const float MIN_DISTANCE = 7000.0f;      // 最小距离 7000km (略大于地球半径)
static const float MAX_DISTANCE = 100000.0f;    // 最大距离 100,000km
static const float MIN_ELEVATION = -89.0f;      // 最小仰角
static const float MAX_ELEVATION = 89.0f;       // 最大仰角

// 聚焦功能相关变量
static bool fKeyPressed = false;         // F键按下状态跟踪

// 时间管理相关变量
static double currentSimulationTime = 0.0;  // 当前仿真时间（秒）
static double timeStep = 1.0;               // 时间步长（秒）
// timeScale现在通过命令行参数传递，不再使用全局变量

// 测试卫星相关变量
static SatelliteState testSatelliteState;   // 测试卫星状态
static uint32_t testSatelliteId = 0;        // 测试卫星ID

// 卫星拖尾功能相关变量
struct SatelliteTrail {
    std::vector<glm::vec3> positions;    // 历史位置点
    std::vector<float> timestamps;       // 对应的时间戳
    glm::vec3 color;                     // 拖尾颜色
    float maxTrailLength = 100.0f;       // 最大拖尾长度（秒）
    size_t maxTrailPoints = 200;         // 最大拖尾点数
};
static std::map<uint32_t, SatelliteTrail> satelliteTrails; // 卫星ID到拖尾数据的映射

/**
 * @brief 更新卫星拖尾数据
 * @param satelliteId 卫星ID
 * @param position 当前位置
 * @param timestamp 当前时间戳
 * @param color 拖尾颜色
 */
void updateSatelliteTrail(uint32_t satelliteId, const glm::vec3& position, float timestamp, const glm::vec3& color) {
    auto& trail = satelliteTrails[satelliteId];
    
    // 设置拖尾颜色（如果是第一次添加）
    if (trail.positions.empty()) {
        trail.color = color;
    }
    
    // 添加新位置点
    trail.positions.push_back(position);
    trail.timestamps.push_back(timestamp);
    
    // 移除过旧的点（基于时间）
    while (!trail.timestamps.empty() && 
           (timestamp - trail.timestamps.front()) > trail.maxTrailLength) {
        trail.positions.erase(trail.positions.begin());
        trail.timestamps.erase(trail.timestamps.begin());
    }
    
    // 限制最大点数
    while (trail.positions.size() > trail.maxTrailPoints) {
        trail.positions.erase(trail.positions.begin());
        trail.timestamps.erase(trail.timestamps.begin());
    }
}

/**
 * @brief 渲染卫星拖尾
 * @param commandBuffer Vulkan命令缓冲区
 * @param cameraParams 相机参数
 */
void renderSatelliteTrails(VkCommandBuffer commandBuffer, const CameraParams& cameraParams) {
    // 这里需要实现拖尾线条的渲染
    // 由于当前的轨道渲染器主要处理轨道和卫星球体，
    // 我们可以通过OrbitRenderer的线条渲染功能来实现拖尾
    
    for (const auto& [satelliteId, trail] : satelliteTrails) {
        if (trail.positions.size() < 2) continue; // 至少需要2个点才能画线
        
        // 将拖尾位置转换为OrbitPoint格式
        std::vector<OrbitPoint> trailPoints;
        for (size_t i = 0; i < trail.positions.size(); ++i) {
            OrbitPoint point;
            point.position = trail.positions[i];
            point.color = trail.color;
            point.timestamp = trail.timestamps[i];
            trailPoints.push_back(point);
        }
        
        // 注意：这里需要OrbitRenderer支持动态线条渲染
        // 当前实现中，我们先通过调试输出来验证拖尾数据是否正确
        static int trailDebugCount = 0;
        if (trailDebugCount % 120 == 0) { // 每120帧输出一次拖尾调试信息
            std::cout << "\n[TRAIL DEBUG] Satellite " << satelliteId << " trail:" << std::endl;
            std::cout << "  Trail points: " << trail.positions.size() << std::endl;
            if (!trail.positions.empty()) {
                std::cout << "  Latest position: (" << trail.positions.back().x 
                         << ", " << trail.positions.back().y 
                         << ", " << trail.positions.back().z << ") km" << std::endl;
                std::cout << "  Trail color: RGB(" << trail.color.r 
                         << ", " << trail.color.g 
                         << ", " << trail.color.b << ")" << std::endl;
            }
        }
        trailDebugCount++;
    }
}

/**
 * @brief 打印卫星状态信息
 */
void printSatelliteStatesInfo() {
    std::cout << "\nSatellite States:" << std::endl;
    std::cout << "-----------------" << std::endl;
    
    if (orbitDebugInfos.empty()) {
        std::cout << "  No satellites to display." << std::endl;
        return;
    }
    
    // 遍历所有轨道任务，获取卫星状态
    for (const auto& info : orbitDebugInfos) {
        SatelliteRenderData satelliteData;
        // 使用当前仿真时间而不是初始时间
        VisualizationError result = visualizationManager->getSatelliteDataForRendering(
            info.taskId, currentSimulationTime, satelliteData);
        
        if (result == VisualizationError::SUCCESS) {
            std::cout << "Satellite for orbit: " << info.name << " (Task ID: " << info.taskId << ")" << std::endl;
            std::cout << "  Current time: " << currentSimulationTime << " seconds" << std::endl;
            // 注意：卫星位置数据已经从j2_orbit_integration.cpp的fromCState函数转换为公里单位
            std::cout << "  Position: (" << satelliteData.state.position.x << ", " << satelliteData.state.position.y << ", " << satelliteData.state.position.z << ") km" << std::endl;
            std::cout << "  Velocity: (" << satelliteData.state.velocity.x << ", " << satelliteData.state.velocity.y << ", " << satelliteData.state.velocity.z << ") km/s" << std::endl;
            std::cout << "  Color: RGB(" << satelliteData.color.r << ", " << satelliteData.color.g << ", " << satelliteData.color.b << ")" << std::endl;
            
            // 计算速度大小（单位：km/s）
            double speed = sqrt(satelliteData.state.velocity.x * satelliteData.state.velocity.x + 
                               satelliteData.state.velocity.y * satelliteData.state.velocity.y + 
                               satelliteData.state.velocity.z * satelliteData.state.velocity.z);
            std::cout << "  Speed: " << speed << " km/s (" << speed * 1000.0 << " m/s)" << std::endl;
            
            // 计算距离地心的距离（单位：km）
            double distance = sqrt(satelliteData.state.position.x * satelliteData.state.position.x + 
                                   satelliteData.state.position.y * satelliteData.state.position.y + 
                                   satelliteData.state.position.z * satelliteData.state.position.z);
            double earthRadius = 6371.0; // 地球半径 (km) - 修正单位匹配
            double altitude = distance - earthRadius;
            std::cout << "  Distance from Earth center: " << distance << " km" << std::endl;
            std::cout << "  Altitude: " << altitude << " km" << std::endl;
            
            // 获取实时计算的轨道元素信息
            OrbitalElements currentElements;
            auto propagator = visualizationManager->getPropagator();
            VisualizationError elementsResult = VisualizationError::UNKNOWN_ERROR;
            if (propagator) {
                elementsResult = propagator->stateToElements(satelliteData.state, currentElements);
            }
            
            std::cout << "  \n  Current Orbital Elements:" << std::endl;
            if (elementsResult == VisualizationError::SUCCESS) {
                std::cout << "\033[32m    Semi-major axis (a): " << currentElements.a / 1000.0 << " km\033[0m" << std::endl;
                std::cout << "\033[32m    Eccentricity (e): " << currentElements.e << "\033[0m" << std::endl;
                std::cout << "\033[32m    Inclination (i): " << glm::degrees(currentElements.i) << " degrees\033[0m" << std::endl;
                std::cout << "\033[32m    Argument of periapsis (ω): " << glm::degrees(currentElements.w) << " degrees\033[0m" << std::endl;
                std::cout << "\033[32m    Right ascension of ascending node (Ω): " << glm::degrees(currentElements.O) << " degrees\033[0m" << std::endl;
                std::cout << "\033[32m    Mean anomaly (M): " << glm::degrees(currentElements.M) << " degrees\033[0m" << std::endl;
            } else {
                // 如果实时计算失败，使用初始轨道元素
                std::cout << "\033[32m    Semi-major axis (a): " << info.elements.a / 1000.0 << " km\033[0m" << std::endl;
                std::cout << "\033[32m    Eccentricity (e): " << info.elements.e << "\033[0m" << std::endl;
                std::cout << "\033[32m    Inclination (i): " << glm::degrees(info.elements.i) << " degrees\033[0m" << std::endl;
                std::cout << "\033[32m    Argument of periapsis (ω): " << glm::degrees(info.elements.w) << " degrees\033[0m" << std::endl;
                std::cout << "\033[32m    Right ascension of ascending node (Ω): " << glm::degrees(info.elements.O) << " degrees\033[0m" << std::endl;
                std::cout << "\033[32m    Mean anomaly (M): " << glm::degrees(info.elements.M) << " degrees\033[0m" << std::endl;
            }
            
            // 计算轨道周期
            double mu = 398600441800000.0; // 地球引力参数 (m³/s²)
            double period = 2.0 * M_PI * sqrt(pow(info.elements.a, 3) / mu);
            std::cout << "\033[32m    Orbital period: " << period / 3600.0 << " hours\033[0m" << std::endl;
            
            // 计算当前时间对应的理论平近点角（假设匀速运动）
            double meanMotion = sqrt(mu / pow(info.elements.a, 3)); // 平均角速度 (rad/s)
            double theoreticalMeanAnomaly = info.elements.M + meanMotion * currentSimulationTime;
            // 将角度规范化到0-2π范围
            while (theoreticalMeanAnomaly > 2.0 * M_PI) theoreticalMeanAnomaly -= 2.0 * M_PI;
            while (theoreticalMeanAnomaly < 0.0) theoreticalMeanAnomaly += 2.0 * M_PI;
            std::cout << "\033[32m    Theoretical Mean Anomaly at current time: " << glm::degrees(theoreticalMeanAnomaly) << " degrees\033[0m" << std::endl;
            
        } else {
            std::cout << "Failed to get satellite data for orbit: " << info.name << " (Task ID: " << info.taskId << ")" << std::endl;
        }
        std::cout << std::endl;
    }
}

/**
 * @brief 打印轨道根数信息
 */
void printOrbitElementsInfo() {
    if (orbitDebugInfos.empty()) {
        std::cout << "  No orbit elements to display." << std::endl;
        return;
    }
    
    std::cout << "\nOrbit Elements Details:" << std::endl;
    std::cout << "----------------------" << std::endl;
    
    for (const auto& info : orbitDebugInfos) {
        std::cout << "Orbit: " << info.name << " (Task ID: " << info.taskId << ")" << std::endl;
        std::cout << "  Color: RGB(" << info.color.r << ", " << info.color.g << ", " << info.color.b << ")" << std::endl;
        std::cout << "\033[32m  Semi-major axis (a): " << info.elements.a << " meters\033[0m" << std::endl;
        std::cout << "\033[32m  Eccentricity (e): " << info.elements.e << "\033[0m" << std::endl;
        std::cout << "\033[32m  Inclination (i): " << glm::degrees(info.elements.i) << " degrees\033[0m" << std::endl;
        std::cout << "\033[32m  Argument of periapsis (ω): " << glm::degrees(info.elements.w) << " degrees\033[0m" << std::endl;
        std::cout << "\033[32m  Right ascension of ascending node (Ω): " << glm::degrees(info.elements.O) << " degrees\033[0m" << std::endl;
        std::cout << "\033[32m  Mean anomaly (M): " << glm::degrees(info.elements.M) << " degrees\033[0m" << std::endl;
        
        // 计算轨道高度（轨道根数中的半长轴单位为米）
        double earthRadius = 6371000.0; // 地球半径 (m)
        double altitude = info.elements.a - earthRadius;
        std::cout << "\033[32m  Altitude: " << altitude / 1000.0 << " km\033[0m" << std::endl;
        
        // 计算轨道周期
        double mu = 398600441800000.0; // 地球引力参数 (m³/s²)
        double period = 2.0 * M_PI * sqrt(pow(info.elements.a, 3) / mu);
        std::cout << "\033[32m  Orbital period: " << period / 3600.0 << " hours\033[0m" << std::endl;
        std::cout << std::endl;
    }
}

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
 * @brief 聚焦到卫星位置
 * 找到第一个可见的卫星并将相机聚焦到其位置
 */
void focusOnSatellite() {
    if (orbitDebugInfos.empty() || !visualizationManager) {
        std::cout << "No satellites available to focus on." << std::endl;
        return;
    }
    
    // 遍历所有轨道任务，找到第一个可见的卫星
    for (const auto& info : orbitDebugInfos) {
        SatelliteRenderData satelliteData;
        double currentTime = 0.0; // 使用初始时间
        
        VisualizationError result = visualizationManager->getSatelliteDataForRendering(
            info.taskId, currentTime, satelliteData);
        
        if (result == VisualizationError::SUCCESS) {
            // 获取卫星位置（单位：km）
            glm::vec3 satellitePos = satelliteData.state.position;
            
            // 计算合适的相机距离（使用千米单位）
            float satelliteDistance = glm::length(satellitePos);
            float focusDistance = 2.0f; // 固定距离2km，确保能清楚看到卫星
            focusDistance = glm::clamp(focusDistance, 1.0f, 5.0f); // 限制在1-5km之间
            
            // 计算从卫星位置到地心的方向
            glm::vec3 toEarth = glm::normalize(-satellitePos);
            
            // 将相机放置在卫星和地球之间，稍微偏移以获得更好的视角
            glm::vec3 offset = glm::vec3(0.1f, 0.1f, 0.0f) * focusDistance;
            camera.position = satellitePos + toEarth * focusDistance + offset;
            
            // 相机看向卫星位置
            camera.target = satellitePos;
            
            // 设置上方向向量
            camera.up = glm::vec3(0.0f, 1.0f, 0.0f);
            
            // 更新轨道相机参数以保持一致性
            orbitDistance = glm::length(camera.position);
            
            // 计算方位角和仰角
            float x = camera.position.x;
            float y = camera.position.y;
            float z = camera.position.z;
            
            orbitAzimuth = glm::degrees(atan2(z, x));
            if (orbitAzimuth < 0.0f) orbitAzimuth += 360.0f;
            
            float horizontalDistance = sqrt(x * x + z * z);
            orbitElevation = glm::degrees(atan2(y, horizontalDistance));
            
            std::cout << "\n[FOCUS DEBUG] Camera focused on satellite: " << info.name << std::endl;
            std::cout << "  Satellite position: (" << satellitePos.x << ", " << satellitePos.y << ", " << satellitePos.z << ") km" << std::endl;
            std::cout << "  Camera position: (" << camera.position.x << ", " << camera.position.y << ", " << camera.position.z << ") km" << std::endl;
            std::cout << "  Camera target: (" << camera.target.x << ", " << camera.target.y << ", " << camera.target.z << ") km" << std::endl;
            std::cout << "  Focus distance: " << focusDistance << " km" << std::endl;
            std::cout << "  Satellite distance from Earth: " << satelliteDistance << " km" << std::endl;
            std::cout << "  Camera far plane: " << camera.farPlane << " km" << std::endl;
            std::cout << "  Satellite scale factor: 500000000.0f (500M)" << std::endl;
            return;
        }
    }
    
    std::cout << "No visible satellites found to focus on." << std::endl;
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
    
    // F键聚焦卫星功能（防止重复触发）
    if (glfwGetKey(window, GLFW_KEY_F) == GLFW_PRESS) {
        if (!fKeyPressed) {
            fKeyPressed = true;
            focusOnSatellite();
        }
    } else {
        fKeyPressed = false;
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
        
        // 创建J2轨道传播器
        auto j2Propagator = std::make_shared<J2OrbitPropagator>();
        
        // 设置传播参数
        PropagationParams propagationParams;
        propagationParams.startTime = 0.0;
        propagationParams.endTime = 86400.0;    // 24小时
        propagationParams.timeStep = 60.0;      // 1分钟步长
        propagationParams.includeJ2 = true;     // 启用J2摄动
        
        // 初始化传播器
        if (j2Propagator->initialize(propagationParams) != VisualizationError::SUCCESS) {
            std::cerr << "Failed to initialize J2 orbit propagator" << std::endl;
            return false;
        }
        
        std::cout << "J2 orbit propagator initialized successfully" << std::endl;
        
        // 创建可视化管理器
        visualizationManager = std::make_unique<OrbitVisualizationManager>(*earthRenderer, *orbitRenderer);
        
        // 初始化可视化管理器并设置传播器
        if (visualizationManager->initialize(j2Propagator) != VisualizationError::SUCCESS) {
            std::cerr << "Failed to initialize orbit visualization manager" << std::endl;
            return false;
        }
        
        std::cout << "Orbit visualization manager initialized successfully" << std::endl;
        
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
    
    // 保存轨道信息用于调试
    orbitDebugInfos.push_back({"LEO", leo, glm::vec3(0.0f, 1.0f, 0.0f), leoTaskId});
    
    // 示例轨道 2：地球同步轨道 (GEO)
    OrbitalElements geo;
    geo.a = 42164169.0;                 // 半长轴 a (m) - 35,786 km 高度
    geo.e = 0.0;
    geo.i = 0.0;
    geo.w = 0.0;
    geo.O = 0.0;
    geo.M = 0.0;
    
    PropagationParams geoParams;
    geoParams.startTime = 0.0;
    geoParams.endTime = 86400.0;        // 24 小时
    geoParams.timeStep = 300.0;         // 5 分钟步长
    
    uint32_t geoTaskId = visualizationManager->addOrbitTask("GEO", geo, glm::vec3(1.0f, 0.0f, 0.0f));
    visualizationManager->executeOrbitPropagation(geoTaskId);
    
    // 保存轨道信息用于调试
    orbitDebugInfos.push_back({"GEO", geo, glm::vec3(1.0f, 0.0f, 0.0f), geoTaskId});
    
    // 示例轨道 3：极地轨道
    OrbitalElements polar;
    polar.a = 7178137.0;                // 半长轴 a (m) - 800 km 高度
    polar.e = 0.001;
    polar.i = static_cast<double>(glm::radians(90.0f)); // 极地轨道
    polar.w = 0.0;
    polar.O = 0.0;
    polar.M = 0.0;
    
    PropagationParams polarParams;
    polarParams.startTime = 0.0;
    polarParams.endTime = 7200.0;       // 2 小时
    polarParams.timeStep = 60.0;        // 1 分钟步长
    
    uint32_t polarTaskId = visualizationManager->addOrbitTask("Polar", polar, glm::vec3(0.0f, 0.0f, 1.0f));
    visualizationManager->executeOrbitPropagation(polarTaskId);
    
    // 保存轨道信息用于调试
    orbitDebugInfos.push_back({"Polar", polar, glm::vec3(0.0f, 0.0f, 1.0f), polarTaskId});
    
    // 添加测试卫星：在LEO轨道上的球形卫星
    testSatelliteState.position = glm::vec3(6778.137f, 0.0f, 0.0f); // LEO轨道半径位置 (km)
    testSatelliteState.velocity = glm::vec3(0.0f, 0.0f, 7.668f);    // 近似LEO轨道速度 (km/s)
    testSatelliteState.timestamp = 0.0;
    testSatelliteState.scale = 20000000.0f;  // 调整为更合理的缩放因子，避免渲染问题
    testSatelliteState.color = glm::vec3(1.0f, 1.0f, 0.0f);  // 保持亮黄色，更醒目
    
    // 添加黄色球形卫星用于测试轨道预报算法
    testSatelliteId = orbitRenderer->addSatellite(
        testSatelliteState,
        testSatelliteState.color,     // 使用状态中的黄色
        testSatelliteState.scale,     // 使用状态中的缩放因子
        true                          // 可见
    );
    
    // 输出测试卫星创建信息
    std::cout << "\n[SATELLITE DEBUG] Test satellite created:" << std::endl;
    std::cout << "  Satellite ID: " << testSatelliteId << std::endl;
    std::cout << "  Initial position: (" << testSatelliteState.position.x << ", " 
              << testSatelliteState.position.y << ", " << testSatelliteState.position.z << ") km" << std::endl;
    std::cout << "  Initial velocity: (" << testSatelliteState.velocity.x << ", " 
              << testSatelliteState.velocity.y << ", " << testSatelliteState.velocity.z << ") km/s" << std::endl;
    std::cout << "  Scale factor: " << testSatelliteState.scale << std::endl;
    std::cout << "  Color: RGB(" << testSatelliteState.color.r << ", " 
              << testSatelliteState.color.g << ", " << testSatelliteState.color.b << ")" << std::endl;
    std::cout << "  Distance from Earth center: " << glm::length(testSatelliteState.position) << " km" << std::endl;
    std::cout << "  Altitude: " << (glm::length(testSatelliteState.position) - 6371.0f) << " km" << std::endl;
    
    std::cout << "Created 3 example orbits and 1 test satellite:" << std::endl;
    std::cout << "  - LEO (Green): 400 km altitude, 51.6° inclination" << std::endl;
    std::cout << "  - GEO (Red): 35,786 km altitude, 0° inclination" << std::endl;
    std::cout << "  - Polar (Blue): 800 km altitude, 90° inclination" << std::endl;
    std::cout << "  - Test Satellite (Yellow): Spherical satellite for orbit propagation testing" << std::endl;
}

/**
 * @brief 主渲染循环
 * @param timeScale 时间缩放因子
 */
void renderLoop(double timeScale) {
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
        
        // 更新仿真时间
        currentSimulationTime += deltaTime * timeScale;
        
        // 每帧打印时间信息
        static int timeFrameCount = 0;
        timeFrameCount++;
        if (timeFrameCount % 60 == 0) { // 每60帧打印一次时间信息
            std::cout << "\n[TIME DEBUG] Frame " << timeFrameCount << ":" << std::endl;
            std::cout << "\033[33m  Current simulation time: " << currentSimulationTime << " seconds\033[0m" << std::endl;
            std::cout << "\033[33m  Time scale: " << timeScale << "x\033[0m" << std::endl;
            std::cout << "\033[33m  Delta time: " << deltaTime << " seconds\033[0m" << std::endl;
            std::cout << "\033[33m  Time increment per frame: " << (deltaTime * timeScale) << " seconds\033[0m" << std::endl;
        }
        
        // 帧计数器（用于控制调试信息输出频率）
        static int frameCount = 0;
        frameCount++;
        
        // 更新所有卫星位置和拖尾
        for (const auto& orbitInfo : orbitDebugInfos) {
            SatelliteRenderData satelliteData;
            VisualizationError result = visualizationManager->getSatelliteDataForRendering(
                orbitInfo.taskId, currentSimulationTime, satelliteData);
            
            // 调试：检查getSatelliteDataForRendering的返回状态
            if (frameCount % 10 == 0) {
                std::cout << "[DEBUG] getSatelliteDataForRendering for " << orbitInfo.name 
                          << " (taskId: " << orbitInfo.taskId << ") returned: " 
                          << (result == VisualizationError::SUCCESS ? "SUCCESS" : "ERROR") << std::endl;
            }
            
            if (result == VisualizationError::SUCCESS) {
                // 获取对应的卫星ID
                uint32_t satelliteId = visualizationManager->getSatelliteId(orbitInfo.taskId);
                if (satelliteId != 0) {
                    // 更新卫星位置
                    orbitRenderer->updateSatellite(satelliteId, satelliteData.state);
                    
                    // 更新卫星拖尾数据
                    updateSatelliteTrail(satelliteId, satelliteData.state.position, 
                                        static_cast<float>(currentSimulationTime), 
                                        satelliteData.state.color);
                    
                    // 详细的卫星渲染调试信息（每20帧输出一次）
            if (frameCount % 20 == 0) {
                        std::cout << "\n[SATELLITE RENDER DEBUG] " << orbitInfo.name << " (ID: " << satelliteId << "):" << std::endl;
                        std::cout << "  Current position: (" << satelliteData.state.position.x << ", " 
                                  << satelliteData.state.position.y << ", " << satelliteData.state.position.z << ") km" << std::endl;
                        std::cout << "  Current velocity: (" << satelliteData.state.velocity.x << ", " 
                                  << satelliteData.state.velocity.y << ", " << satelliteData.state.velocity.z << ") km/s" << std::endl;
                        std::cout << "  Scale factor: " << satelliteData.state.scale << std::endl;
                        std::cout << "  Color: RGB(" << satelliteData.state.color.r << ", " 
                                  << satelliteData.state.color.g << ", " << satelliteData.state.color.b << ")" << std::endl;
                        std::cout << "  Distance from Earth: " << glm::length(satelliteData.state.position) << " km" << std::endl;
                        std::cout << "  Distance from camera: " << glm::length(satelliteData.state.position - camera.position) << " km" << std::endl;
                        
                        // 可见性检查
                        glm::vec3 toSatellite = satelliteData.state.position - camera.position;
                        glm::vec3 cameraForward = glm::normalize(camera.target - camera.position);
                        float dotProduct = glm::dot(glm::normalize(toSatellite), cameraForward);
                        float angle = glm::degrees(acos(glm::clamp(dotProduct, -1.0f, 1.0f)));
                        bool inFOV = angle < (camera.fov / 2.0f);
                        
                        std::cout << "  Angle from camera center: " << angle << " degrees (FOV: " << camera.fov << " degrees)" << std::endl;
                        std::cout << "  In camera FOV: " << (inFOV ? "YES" : "NO") << std::endl;
                        
                        // 检查是否被地球遮挡
                        float earthRadius = 6371.0f; // km
                        glm::vec3 earthCenter = glm::vec3(0.0f, 0.0f, 0.0f);
                        glm::vec3 cameraToSatellite = satelliteData.state.position - camera.position;
                        glm::vec3 cameraToEarth = earthCenter - camera.position;
                        
                        // 简单的遮挡检查：如果卫星在地球后面
                        float cameraEarthDist = glm::length(cameraToEarth);
                        float cameraSatDist = glm::length(cameraToSatellite);
                        bool behindEarth = (cameraEarthDist < cameraSatDist) && (glm::length(satelliteData.state.position) < cameraEarthDist);
                        
                        std::cout << "  Behind Earth: " << (behindEarth ? "YES" : "NO") << std::endl;
                        std::cout << "  Should be visible: " << (inFOV && !behindEarth ? "YES" : "NO") << std::endl;
                    }
                }
            }
        }
        
        // 为testSatellite添加专门的调试信息（每20帧输出一次）
        if (frameCount % 20 == 0) {
            std::cout << "\n[TEST SATELLITE DEBUG] Test Satellite (ID: " << testSatelliteId << "):" << std::endl;
            std::cout << "  Initial position: (" << testSatelliteState.position.x << ", " 
                      << testSatelliteState.position.y << ", " << testSatelliteState.position.z << ") km" << std::endl;
            std::cout << "  Scale factor: " << testSatelliteState.scale << std::endl;
            std::cout << "  Color: RGB(" << testSatelliteState.color.r << ", " 
                      << testSatelliteState.color.g << ", " << testSatelliteState.color.b << ")" << std::endl;
            std::cout << "  Distance from Earth: " << glm::length(testSatelliteState.position) << " km" << std::endl;
            std::cout << "  Distance from camera: " << glm::length(testSatelliteState.position - camera.position) << " km" << std::endl;
            
            // 可见性检查
            glm::vec3 toTestSatellite = testSatelliteState.position - camera.position;
            glm::vec3 cameraForward = glm::normalize(camera.target - camera.position);
            float dotProduct = glm::dot(glm::normalize(toTestSatellite), cameraForward);
            float angle = glm::degrees(acos(glm::clamp(dotProduct, -1.0f, 1.0f)));
            bool inFOV = angle < (camera.fov / 2.0f);
            
            std::cout << "  Angle from camera center: " << angle << " degrees (FOV: " << camera.fov << " degrees)" << std::endl;
            std::cout << "  In camera FOV: " << (inFOV ? "YES" : "NO") << std::endl;
            
            // 检查是否被地球遮挡
            float earthRadius = 6371.0f; // km
            glm::vec3 earthCenter = glm::vec3(0.0f, 0.0f, 0.0f);
            glm::vec3 cameraToTestSatellite = testSatelliteState.position - camera.position;
            glm::vec3 cameraToEarth = earthCenter - camera.position;
            
            // 简单的遮挡检查：如果卫星在地球后面
            float cameraEarthDist = glm::length(cameraToEarth);
            float cameraTestSatDist = glm::length(cameraToTestSatellite);
            bool behindEarth = (cameraEarthDist < cameraTestSatDist) && (glm::length(testSatelliteState.position) < cameraEarthDist);
            
            std::cout << "  Behind Earth: " << (behindEarth ? "YES" : "NO") << std::endl;
            std::cout << "  Should be visible: " << (inFOV && !behindEarth ? "YES" : "NO") << std::endl;
        }
        
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
            earthParams.radius = 6371.0f; // 地球半径：6,371 km（与卫星位置单位保持一致）
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
            
            // 调试输出（每30帧输出一次，避免过多输出）
            if (frameCount % 30 == 0) {
                std::cout << "\n[DEBUG] Frame " << frameCount << " - Orbit Visualization Status:" << std::endl;
                std::cout << "========================================" << std::endl;
                
                // 打印相机信息
                std::cout << "Camera Info:" << std::endl;
                std::cout << "  Position: (" << camera.position.x << ", " << camera.position.y << ", " << camera.position.z << ")" << std::endl;
                std::cout << "  Distance from origin: " << glm::length(camera.position) << " meters" << std::endl;
                std::cout << "  Target: (" << camera.target.x << ", " << camera.target.y << ", " << camera.target.z << ")" << std::endl;
                
                // 打印轨道渲染器状态
                std::cout << "\nOrbit Renderer Status:" << std::endl;
                std::cout << "  Total orbits: " << orbitRenderer->getOrbitCount() << std::endl;
                std::cout << "  Total satellites: " << orbitRenderer->getSatelliteCount() << std::endl;
                
                // 打印可视化管理器状态
                if (visualizationManager) {
                    std::cout << "\nVisualization Manager Status:" << std::endl;
                    std::cout << "  Total orbit tasks: " << visualizationManager->getTaskCount() << std::endl;
                }
                
                std::cout << "\nEarth Renderer Status:" << std::endl;
                std::cout << "  Earth radius: " << earthParams.radius << " meters" << std::endl;
                std::cout << "  Tessellation level: " << earthParams.tessellationLevel << std::endl;
                std::cout << "  Atmosphere enabled: " << (earthParams.enableAtmosphere ? "Yes" : "No") << std::endl;
                std::cout << "  Clouds enabled: " << (earthParams.enableClouds ? "Yes" : "No") << std::endl;
                
                // 打印轨道根数信息
                printOrbitElementsInfo();
                
                // 打印卫星状态信息
                printSatelliteStatesInfo();
                
                // 检查轨道渲染状态
                std::cout << "\nOrbit Rendering Status Check:" << std::endl;
                std::cout << "-----------------------------" << std::endl;
                
                // 检查轨道任务状态
                if (visualizationManager) {
                    std::cout << "  Visualization manager has " << visualizationManager->getTaskCount() << " orbit tasks" << std::endl;
                    // 注意：getOrbitDataForRendering需要taskId参数，这里暂时跳过详细轨道点信息
                    std::cout << "  (Detailed orbit point data requires specific task ID)" << std::endl;
                }
                
                // 检查渲染器状态
                std::cout << "\nRenderer Status:" << std::endl;
                std::cout << "  Vulkan renderer initialized: " << (vulkanRenderer ? "Yes" : "No") << std::endl;
                std::cout << "  Earth renderer initialized: " << (earthRenderer ? "Yes" : "No") << std::endl;
                std::cout << "  Orbit renderer initialized: " << (orbitRenderer ? "Yes" : "No") << std::endl;
                std::cout << "  Visualization manager initialized: " << (visualizationManager ? "Yes" : "No") << std::endl;
                
                std::cout << "\nRendering Earth..." << std::endl;
            }
            
            earthRenderer->render(vulkanRenderer->getCurrentCommandBuffer(), earthParams, cameraParams);
            
            if (frameCount % 30 == 1) {
                std::cout << "Earth rendering completed." << std::endl;
                std::cout << "Rendering orbits and satellites..." << std::endl;
            }
            
            // 渲染轨道
            orbitRenderer->render(vulkanRenderer->getCurrentCommandBuffer(), cameraParams);
            
            // 渲染卫星拖尾
            renderSatelliteTrails(vulkanRenderer->getCurrentCommandBuffer(), cameraParams);
            
            // 渲染管线状态检查（每60帧输出一次）
            if (frameCount % 60 == 0) {
                std::cout << "\n[RENDER PIPELINE DEBUG]:" << std::endl;
                std::cout << "  Orbit renderer satellite count: " << orbitRenderer->getSatelliteCount() << std::endl;
                std::cout << "  Command buffer valid: " << (vulkanRenderer->getCurrentCommandBuffer() != VK_NULL_HANDLE ? "YES" : "NO") << std::endl;
                std::cout << "  Camera position: (" << cameraParams.position.x << ", " 
                          << cameraParams.position.y << ", " << cameraParams.position.z << ") km" << std::endl;
                std::cout << "  Camera target: (" << cameraParams.target.x << ", " 
                          << cameraParams.target.y << ", " << cameraParams.target.z << ") km" << std::endl;
                std::cout << "  Camera FOV: " << cameraParams.fov << " degrees" << std::endl;
                std::cout << "  Near plane: " << cameraParams.nearPlane << " km" << std::endl;
                std::cout << "  Far plane: " << cameraParams.farPlane << " km" << std::endl;
            }
            
            // 轨道渲染完成后的调试输出
            if (frameCount % 30 == 2) {
                std::cout << "Orbit and satellite rendering completed." << std::endl;
                std::cout << "========================================\n" << std::endl;
            }
            
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
 * 主函数
 * @param argc 命令行参数个数
 * @param argv 命令行参数数组
 * @return 程序退出码
 */
int main(int argc, char* argv[]) {
    std::cout << "J2 Orbit Visualization Demo" << std::endl;
    std::cout << "===========================" << std::endl;
    std::cout << "Controls:" << std::endl;
    std::cout << "  WASD - Move camera" << std::endl;
    std::cout << "  QE - Move up/down" << std::endl;
    std::cout << "  Mouse - Look around" << std::endl;
    std::cout << "  Scroll - Zoom in/out" << std::endl;
    std::cout << "  F - Focus on satellite" << std::endl;
    std::cout << "  ESC - Exit" << std::endl;
    std::cout << std::endl;
    std::cout << "Tip: Press F to automatically focus the camera on a satellite!" << std::endl;
    std::cout << std::endl;
    
    try {
        // 解析命令行参数
        double timeScale = 1.0; // 默认时间缩放因子
        if (argc > 1) {
            try {
                timeScale = std::stod(argv[1]);
                if (timeScale <= 0) {
                    std::cerr << "错误：时间缩放因子必须大于0，当前值: " << timeScale << std::endl;
                    return 1;
                }
                std::cout << "使用时间缩放因子: " << timeScale << std::endl;
            } catch (const std::exception& e) {
                std::cerr << "错误：无法解析时间缩放因子参数 '" << argv[1] << "': " << e.what() << std::endl;
                std::cerr << "使用默认值: 1.0" << std::endl;
                timeScale = 1.0;
            }
        }
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
        renderLoop(timeScale);
        
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