#pragma once

#include "visualization_types.h"
#include "orbit_renderer.h"
#include "earth_renderer.h"
#include <vector>
#include <string>
#include <functional>
#include <memory>
#include <unordered_map>
#include <j2_orbit_propagator_c.h>

namespace j2_orbit_visualization {

/**
 * @brief 轨道元素结构
 * 统一使用核心库的 COrbitalElements
 */
using OrbitalElements = COrbitalElements;

/**
 * @brief 轨道传播参数
 * 控制轨道传播的参数
 */
struct PropagationParams {
    double startTime;         ///< 开始时间 (秒)
    double endTime;           ///< 结束时间 (秒)
    double timeStep;          ///< 时间步长 (秒)
    bool includeJ2;           ///< 是否包含 J2 摄动
    bool includeAtmosphericDrag; ///< 是否包含大气阻力
    bool includeSolarRadiation;  ///< 是否包含太阳辐射压
    double j2Coefficient;     ///< J2 系数
    double earthRadius;       ///< 地球半径 (km)
    double earthMu;           ///< 地球引力参数 (km³/s²)
};

/**
 * @brief 轨道传播结果
 * 存储传播计算的结果
 */
struct PropagationResult {
    std::vector<OrbitPoint> orbitPoints; ///< 轨道点集合
    std::vector<SatelliteState> states;  ///< 卫星状态集合
    double totalTime;         ///< 总传播时间
    size_t stepCount;         ///< 计算步数
    bool success;             ///< 是否成功
    std::string errorMessage; ///< 错误信息
};

/**
 * @brief 进度回调函数类型
 * 用于报告轨道传播进度
 * @param progress 进度百分比 (0.0 - 1.0)
 * @param currentTime 当前时间
 * @param message 状态消息
 */
using ProgressCallback = std::function<void(double progress, double currentTime, const std::string& message)>;

/**
 * @brief J2 轨道传播器类
 * 负责与现有 J2 轨道外推算法的集成
 */
class J2OrbitPropagator {
public:
    /**
     * @brief 构造函数
     */
    J2OrbitPropagator();
    
    /**
     * @brief 析构函数
     */
    ~J2OrbitPropagator();
    
    /**
     * @brief 初始化传播器
     * @param params 传播参数
     * @return VisualizationError 初始化结果
     */
    VisualizationError initialize(const PropagationParams& params);
    
    /**
     * @brief 从轨道元素传播轨道
     * @param elements 初始轨道元素
     * @param result 输出传播结果
     * @param progressCallback 进度回调函数（可选）
     * @return VisualizationError 传播结果
     */
    VisualizationError propagateFromElements(const OrbitalElements& elements,
                                           PropagationResult& result,
                                           ProgressCallback progressCallback = nullptr);
    
    /**
     * @brief 从状态向量传播轨道
     * @param initialState 初始状态向量
     * @param result 输出传播结果
     * @param progressCallback 进度回调函数（可选）
     * @return VisualizationError 传播结果
     */
    VisualizationError propagateFromState(const SatelliteState& initialState,
                                        PropagationResult& result,
                                        ProgressCallback progressCallback = nullptr);
    
    /**
     * @brief 从 TLE 数据传播轨道
     * @param tleLine1 TLE 第一行
     * @param tleLine2 TLE 第二行
     * @param result 输出传播结果
     * @param progressCallback 进度回调函数（可选）
     * @return VisualizationError 传播结果
     */
    VisualizationError propagateFromTLE(const std::string& tleLine1,
                                      const std::string& tleLine2,
                                      PropagationResult& result,
                                      ProgressCallback progressCallback = nullptr);
    
    /**
     * @brief 计算单个时刻的卫星状态
     * @param elements 轨道元素
     * @param time 目标时间（相对于历元的秒数）
     * @param state 输出卫星状态
     * @return VisualizationError 计算结果
     */
    VisualizationError computeStateAtTime(const OrbitalElements& elements,
                                        double time,
                                        SatelliteState& state);
    
    /**
     * @brief 设置传播参数
     * @param params 新的传播参数
     */
    void setPropagationParams(const PropagationParams& params) { propagationParams = params; }
    
    /**
     * @brief 获取传播参数
     * @return const PropagationParams& 当前传播参数
     */
    const PropagationParams& getPropagationParams() const { return propagationParams; }
    
    /**
     * @brief 设置 J2 系数
     * @param j2 J2 系数值
     */
    void setJ2Coefficient(double j2) { propagationParams.j2Coefficient = j2; }
    
    /**
     * @brief 获取 J2 系数
     * @return double J2 系数值
     */
    double getJ2Coefficient() const { return propagationParams.j2Coefficient; }
    
    /**
     * @brief 启用/禁用 J2 摄动
     * @param enable 是否启用
     */
    void setJ2Enabled(bool enable) { propagationParams.includeJ2 = enable; }
    
    /**
     * @brief 检查 J2 摄动是否启用
     * @return bool 是否启用
     */
    bool isJ2Enabled() const { return propagationParams.includeJ2; }
    
    /**
     * @brief 将轨道元素转换为状态向量
     * @param elements 轨道元素
     * @param state 输出状态向量
     * @return VisualizationError 转换结果
     */
    static VisualizationError elementsToState(const OrbitalElements& elements, SatelliteState& state);
    
    /**
     * @brief 将状态向量转换为轨道元素
     * @param state 状态向量
     * @param elements 输出轨道元素
     * @return VisualizationError 转换结果
     */
    static VisualizationError stateToElements(const SatelliteState& state, OrbitalElements& elements);
    
    /**
     * @brief 解析 TLE 数据
     * @param tleLine1 TLE 第一行
     * @param tleLine2 TLE 第二行
     * @param elements 输出轨道元素
     * @return VisualizationError 解析结果
     */
    static VisualizationError parseTLE(const std::string& tleLine1,
                                     const std::string& tleLine2,
                                     OrbitalElements& elements);
    
    /**
     * @brief 计算轨道周期
     * @param semiMajorAxis 半长轴 (km)
     * @param mu 引力参数 (km³/s²)
     * @return double 轨道周期 (秒)
     */
    static double calculateOrbitalPeriod(double semiMajorAxis, double mu = 398600.4418);
    
    /**
     * @brief 计算平均运动
     * @param semiMajorAxis 半长轴 (km)
     * @param mu 引力参数 (km³/s²)
     * @return double 平均运动 (rad/s)
     */
    static double calculateMeanMotion(double semiMajorAxis, double mu = 398600.4418);

private:
    PropagationParams propagationParams; ///< 传播参数
    bool initialized = false;            ///< 是否已初始化
    
    /**
     * @brief 执行数值积分
     * @param initialState 初始状态
     * @param timeStep 时间步长
     * @param duration 积分持续时间
     * @param result 输出结果
     * @param progressCallback 进度回调
     * @return VisualizationError 积分结果
     */
    VisualizationError performNumericalIntegration(const SatelliteState& initialState,
                                                  double timeStep,
                                                  double duration,
                                                  PropagationResult& result,
                                                  ProgressCallback progressCallback);
    
    /**
     * @brief 计算加速度（包含 J2 摄动）
     * @param position 位置向量 (km)
     * @param velocity 速度向量 (km/s)
     * @param acceleration 输出加速度向量 (km/s²)
     */
    void computeAcceleration(const glm::dvec3& position,
                           const glm::dvec3& velocity,
                           glm::dvec3& acceleration);
    
    /**
     * @brief 计算 J2 摄动加速度
     * @param position 位置向量 (km)
     * @param j2Acceleration 输出 J2 加速度向量 (km/s²)
     */
    void computeJ2Acceleration(const glm::dvec3& position, glm::dvec3& j2Acceleration);
    
    /**
     * @brief 四阶龙格-库塔积分步骤
     * @param state 当前状态
     * @param timeStep 时间步长
     * @param newState 输出新状态
     */
    void rungeKutta4Step(const SatelliteState& state, double timeStep, SatelliteState& newState);
    
    /**
     * @brief 状态导数函数
     * @param state 当前状态
     * @param derivative 输出状态导数
     */
    void stateDerivative(const SatelliteState& state, SatelliteState& derivative);
    
    /**
     * @brief 验证轨道元素的有效性
     * @param elements 轨道元素
     * @return bool 是否有效
     */
    bool validateOrbitalElements(const OrbitalElements& elements);
    
    /**
     * @brief 验证状态向量的有效性
     * @param state 状态向量
     * @return bool 是否有效
     */
    bool validateSatelliteState(const SatelliteState& state);

    bool validatePropagationParams(const PropagationParams& params);
    bool computeStateVector(const OrbitalElements& elements, double time, glm::vec3& position, glm::vec3& velocity);
    bool propagateStep(OrbitalElements& elements, double timeStep);
    bool stateToElements(const glm::vec3& position, const glm::vec3& velocity, OrbitalElements& elements);
};

/**
 * @brief 轨道可视化管理器类
 * 整合轨道传播和可视化功能
 */
class OrbitVisualizationManager {
public:
    /**
     * @brief 构造函数
     * @param earthRenderer 地球渲染器引用
     * @param orbitRenderer 轨道渲染器引用
     */
    OrbitVisualizationManager(EarthRenderer& earthRenderer, OrbitRenderer& orbitRenderer);
    
    /**
     * @brief 析构函数
     */
    ~OrbitVisualizationManager();
    
    /**
     * @brief 初始化管理器
     * @param propagator 轨道传播器
     * @return VisualizationError 初始化结果
     */
    VisualizationError initialize(std::shared_ptr<J2OrbitPropagator> propagator);
    
    /**
     * @brief 添加轨道任务
     * @param name 轨道名称
     * @param elements 轨道元素
     * @param color 轨道颜色
     * @return uint32_t 轨道任务ID
     */
    uint32_t addOrbitTask(const std::string& name,
                         const OrbitalElements& elements,
                         const glm::vec3& color = glm::vec3(1.0f, 1.0f, 1.0f));
    
    /**
     * @brief 执行轨道传播
     * @param taskId 轨道任务ID
     * @param progressCallback 进度回调
     * @return VisualizationError 执行结果
     */
    VisualizationError executeOrbitPropagation(uint32_t taskId,
                                              ProgressCallback progressCallback = nullptr);
    
    /**
     * @brief 获取轨道数据用于渲染
     * @param taskId 轨道任务ID
     * @param orbitData 输出轨道数据
     * @return VisualizationError 获取结果
     */
    VisualizationError getOrbitDataForRendering(uint32_t taskId, OrbitData& orbitData);
    
    /**
     * @brief 获取卫星状态用于渲染
     * @param taskId 轨道任务ID
     * @param time 目标时间
     * @param satelliteData 输出卫星数据
     * @return VisualizationError 获取结果
     */
    VisualizationError getSatelliteDataForRendering(uint32_t taskId,
                                                   double time,
                                                   SatelliteRenderData& satelliteData);
    
    /**
     * @brief 移除轨道任务
     * @param taskId 轨道任务ID
     * @return VisualizationError 移除结果
     */
    VisualizationError removeOrbitTask(uint32_t taskId);
    
    /**
     * @brief 清除所有轨道任务
     */
    void clearAllTasks();
    
    /**
     * @brief 清除所有轨道
     */
    void clearAllOrbits();
    
    /**
     * @brief 设置轨道可见性
     * @param taskId 任务ID
     * @param visible 是否可见
     */
    void setOrbitVisible(uint32_t taskId, bool visible);
    
    /**
     * @brief 获取任务数量
     * @return size_t 任务数量
     */
    size_t getTaskCount() const;
    
    /**
     * @brief 获取任务对应的卫星ID
     * @param taskId 轨道任务ID
     * @return uint32_t 卫星ID，如果任务不存在或未计算则返回0
     */
    uint32_t getSatelliteId(uint32_t taskId) const;
    
    /**
     * @brief 获取轨道传播器实例
     * @return std::shared_ptr<J2OrbitPropagator> 轨道传播器实例
     */
    std::shared_ptr<J2OrbitPropagator> getPropagator() const;

private:
    struct OrbitTask {
        uint32_t id;
        std::string name;
        OrbitalElements elements;
        glm::vec3 color;
        PropagationResult result;
        bool computed = false;
        uint32_t orbitId = 0;      ///< 轨道渲染ID
        uint32_t satelliteId = 0;  ///< 卫星渲染ID
    };
    
    std::shared_ptr<J2OrbitPropagator> propagator; ///< 轨道传播器
    std::vector<std::unique_ptr<OrbitTask>> tasks; ///< 轨道任务列表
    std::unordered_map<uint32_t, OrbitTask> orbitTasks; ///< 轨道任务映射
    EarthRenderer& earthRenderer; ///< 地球渲染器引用
    OrbitRenderer& orbitRenderer; ///< 轨道渲染器引用
    uint32_t nextTaskId = 1;                      ///< 下一个任务ID
    bool initialized = false;                     ///< 是否已初始化
};

} // namespace j2_orbit_visualization