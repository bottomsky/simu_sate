#include "j2_orbit_integration.h"
#include <iostream>
#include <cmath>
#include <algorithm>
#include <thread>
#include <chrono>
#include <cstring>
#include <limits>

// 引入核心库的 C 接口
#include <j2_orbit_propagator_c.h>
// 引入数学常数定义
#include <math_defs.h>

namespace j2_orbit_visualization {

// 单位换算工具
static inline double km_to_m(double km) { return km * 1000.0; }
static inline double m_to_km(double m) { return m / 1000.0; }

// 将 C 状态向量转换为可视化卫星状态（单位转为 km / km/s）
static SatelliteState fromCState(const CStateVector& cs, double timestamp_sec) {
    SatelliteState s{};
    s.position = glm::vec3(
        static_cast<float>(m_to_km(cs.r[0])),
        static_cast<float>(m_to_km(cs.r[1])),
        static_cast<float>(m_to_km(cs.r[2]))
    );
    s.velocity = glm::vec3(
        static_cast<float>(cs.v[0] / 1000.0),
        static_cast<float>(cs.v[1] / 1000.0),
        static_cast<float>(cs.v[2] / 1000.0)
    );
    s.timestamp = timestamp_sec;
    s.scale = 1.0f;
    s.color = glm::vec3(1.0f, 1.0f, 1.0f);
    return s;
}

/**
 * @brief 构造函数
 */
J2OrbitPropagator::J2OrbitPropagator() {
    // 无状态构造
}

/**
 * @brief 析构函数
 */
J2OrbitPropagator::~J2OrbitPropagator() {
    // 无需显式资源清理（按调用点创建/销毁 C 句柄）
}

VisualizationError J2OrbitPropagator::initialize(const PropagationParams& params) {
    propagationParams = params;
    initialized = true;
    return VisualizationError::SUCCESS;
}

VisualizationError J2OrbitPropagator::propagateFromElements(const OrbitalElements& elements,
                                                            PropagationResult& outResult,
                                                            ProgressCallback progressCallback) {
    if (!initialized) {
        return VisualizationError::UNKNOWN_ERROR;
    }
    if (!validateOrbitalElements(elements) || !validatePropagationParams(propagationParams)) {
        outResult.success = false;
        outResult.errorMessage = "Invalid elements or propagation parameters";
        return VisualizationError::UNKNOWN_ERROR;
    }

    const double start = propagationParams.startTime;
    const double end = propagationParams.endTime;
    const double dt = propagationParams.timeStep;
    if (end <= start || dt <= 0) {
        outResult.success = false;
        outResult.errorMessage = "Invalid time range or step";
        return VisualizationError::UNKNOWN_ERROR;
    }

    // 创建传播器句柄（以 start 作为历元秒）
    COrbitalElements c_init = elements; c_init.t = start;
    J2PropagatorHandle handle = j2_propagator_create(&c_init);
    if (!handle) {
        outResult.success = false;
        outResult.errorMessage = "Failed to create propagator handle";
        return VisualizationError::UNKNOWN_ERROR;
    }

    // 设置步长（非必须）
    j2_propagator_set_step_size(handle, dt);

    outResult.orbitPoints.clear();
    outResult.states.clear();

    const int steps = static_cast<int>(std::floor((end - start) / dt));
    const int totalSteps = std::max(0, steps);

    for (int k = 0; k <= totalSteps; ++k) {
        double t = start + k * dt;

        // 将内部状态传播到 t
        COrbitalElements c_elem_out{};
        if (j2_propagator_propagate(handle, t, &c_elem_out) != 0) {
            j2_propagator_destroy(handle);
            outResult.success = false;
            outResult.errorMessage = "Propagation failed at step " + std::to_string(k);
            return VisualizationError::UNKNOWN_ERROR;
        }

        // 计算状态向量
        CStateVector c_state{};
        if (j2_propagator_elements_to_state(handle, &c_elem_out, &c_state) != 0) {
            j2_propagator_destroy(handle);
            outResult.success = false;
            outResult.errorMessage = "elements_to_state failed at step " + std::to_string(k);
            return VisualizationError::UNKNOWN_ERROR;
        }

        SatelliteState sat = fromCState(c_state, t);
        outResult.states.push_back(sat);

        OrbitPoint pt{};
        pt.position = sat.position;
        pt.color = glm::vec3(1.0f, 1.0f, 1.0f);
        pt.timestamp = static_cast<float>(t);
        outResult.orbitPoints.push_back(pt);

        if (progressCallback) {
            double progress = (totalSteps == 0) ? 1.0 : static_cast<double>(k) / static_cast<double>(totalSteps);
            progressCallback(progress, t, "propagating");
        }
    }

    // 最终要素（从最后一步取回）
    if (!outResult.states.empty()) {
        CStateVector c_last{};
        c_last.r[0] = outResult.states.back().position.x * 1000.0;
        c_last.r[1] = outResult.states.back().position.y * 1000.0;
        c_last.r[2] = outResult.states.back().position.z * 1000.0;
        c_last.v[0] = outResult.states.back().velocity.x * 1000.0;
        c_last.v[1] = outResult.states.back().velocity.y * 1000.0;
        c_last.v[2] = outResult.states.back().velocity.z * 1000.0;
        COrbitalElements c_last_el{};
        // 使用句柄将状态反解为要素（时间取 end）
        if (j2_propagator_state_to_elements(handle, &c_last, end, &c_last_el) == 0) {
            // 可选：如需要，可将 outResult 内部扩展存储最终要素
        }
    }

    j2_propagator_destroy(handle);

    outResult.stepCount = outResult.orbitPoints.size();
    outResult.totalTime = end - start;
    outResult.success = true;
    outResult.errorMessage.clear();
    return VisualizationError::SUCCESS;
}

VisualizationError J2OrbitPropagator::propagateFromState(const SatelliteState& initialState,
                                                        PropagationResult& result,
                                                        ProgressCallback progressCallback) {
    // 先将状态转换为要素，再复用 elements 流程
    OrbitalElements elems{};
    auto err = J2OrbitPropagator::stateToElements(initialState, elems);
    if (err != VisualizationError::SUCCESS) {
        result.success = false;
        result.errorMessage = "stateToElements failed";
        return err;
    }
    return propagateFromElements(elems, result, progressCallback);
}

VisualizationError J2OrbitPropagator::computeStateAtTime(const OrbitalElements& elements,
                                                         double time,
                                                         SatelliteState& state) {
    // 构造临时句柄并传播到指定时刻，然后计算状态
    COrbitalElements c_init = elements; // 保持原始时间戳，不修改c_init.t
    J2PropagatorHandle handle = j2_propagator_create(&c_init);
    if (!handle) return VisualizationError::UNKNOWN_ERROR;

    COrbitalElements c_out{};
    if (j2_propagator_propagate(handle, time, &c_out) != 0) {
        j2_propagator_destroy(handle);
        return VisualizationError::UNKNOWN_ERROR;
    }
    CStateVector c_state{};
    if (j2_propagator_elements_to_state(handle, &c_out, &c_state) != 0) {
        j2_propagator_destroy(handle);
        return VisualizationError::UNKNOWN_ERROR;
    }
    state = fromCState(c_state, time);
    j2_propagator_destroy(handle);
    return VisualizationError::SUCCESS;
}

VisualizationError J2OrbitPropagator::elementsToState(const OrbitalElements& elements,
                                                      SatelliteState& state) {
    COrbitalElements c_init = elements;
    J2PropagatorHandle handle = j2_propagator_create(&c_init);
    if (!handle) return VisualizationError::UNKNOWN_ERROR;
    CStateVector c_state{};
    if (j2_propagator_elements_to_state(handle, &c_init, &c_state) != 0) {
        j2_propagator_destroy(handle);
        return VisualizationError::UNKNOWN_ERROR;
    }
    state = fromCState(c_state, 0.0);
    j2_propagator_destroy(handle);
    return VisualizationError::SUCCESS;
}

VisualizationError J2OrbitPropagator::stateToElements(const SatelliteState& state,
                                                      OrbitalElements& elements) {
    // 使用状态的时间戳进行换算
    CStateVector cs{};
    cs.r[0] = state.position.x * 1000.0;
    cs.r[1] = state.position.y * 1000.0;
    cs.r[2] = state.position.z * 1000.0;
    cs.v[0] = state.velocity.x * 1000.0;
    cs.v[1] = state.velocity.y * 1000.0;
    cs.v[2] = state.velocity.z * 1000.0;

    // 创建一个占位初始要素（半长轴等值可以从位置速度反推，句柄仅用于调用接口）
    COrbitalElements c_dummy{};
    c_dummy.a = std::max(1.0, std::sqrt(cs.r[0]*cs.r[0] + cs.r[1]*cs.r[1] + cs.r[2]*cs.r[2]));
    c_dummy.e = 0.0; c_dummy.i = 0.0; c_dummy.O = 0.0; c_dummy.w = 0.0; c_dummy.M = 0.0; c_dummy.t = 0.0;

    J2PropagatorHandle handle = j2_propagator_create(&c_dummy);
    if (!handle) return VisualizationError::UNKNOWN_ERROR;

    COrbitalElements c_out{};
    // 使用状态的时间戳而不是0.0，这样可以正确计算当前时间的轨道元素
    if (j2_propagator_state_to_elements(handle, &cs, state.timestamp, &c_out) != 0) {
        j2_propagator_destroy(handle);
        return VisualizationError::UNKNOWN_ERROR;
    }
    elements = c_out;
    j2_propagator_destroy(handle);
    return VisualizationError::SUCCESS;
}

VisualizationError J2OrbitPropagator::parseTLE(const std::string& /*tleLine1*/,
                                               const std::string& /*tleLine2*/,
                                               OrbitalElements& /*elements*/) {
    // 可按需实现；当前返回未实现
    return VisualizationError::UNKNOWN_ERROR;
}

double J2OrbitPropagator::calculateOrbitalPeriod(double semiMajorAxis, double mu) {
    // 输入单位：a (m), mu (m^3/s^2)
    if (semiMajorAxis <= 0 || mu <= 0) return std::numeric_limits<double>::quiet_NaN();
    return 2.0 * M_PI * std::sqrt((semiMajorAxis*semiMajorAxis*semiMajorAxis) / mu);
}

double J2OrbitPropagator::calculateMeanMotion(double semiMajorAxis, double mu) {
    // 输入单位：a (m), mu (m^3/s^2)
    if (semiMajorAxis <= 0 || mu <= 0) return std::numeric_limits<double>::quiet_NaN();
    return std::sqrt(mu / (semiMajorAxis*semiMajorAxis*semiMajorAxis));
}

bool J2OrbitPropagator::validateOrbitalElements(const OrbitalElements& elements) {
    // COrbitalElements: a(m), e, i(rad)
    if (!(elements.a > RE_EARTH)) return false;
    if (elements.e < 0.0 || elements.e >= 1.0) return false;
    if (elements.i < 0.0 || elements.i > M_PI) return false;
    return true;
}

bool J2OrbitPropagator::validateSatelliteState(const SatelliteState& state) {
    (void)state;
    return true;
}

bool J2OrbitPropagator::validatePropagationParams(const PropagationParams& params) {
    if (params.startTime >= params.endTime) return false;
    if (params.timeStep <= 0.0) return false;
    return true;
}

// 以下三个函数为旧的算法占位实现（当前未在动态库模式下使用），保留以兼容私有声明
bool J2OrbitPropagator::computeStateVector(const OrbitalElements& elements, double time,
                                          glm::vec3& position, glm::vec3& velocity) {
    // 改为使用核心库进行计算
    SatelliteState st{};
    if (computeStateAtTime(elements, time, st) != VisualizationError::SUCCESS) {
        return false;
    }
    position = st.position;
    velocity = st.velocity;
    return true;
}

bool J2OrbitPropagator::propagateStep(OrbitalElements& elements, double timeStep) {
    // 使用核心库推进一步：从 elements.t 推进到 elements.t + timeStep
    COrbitalElements c_init = elements;
    J2PropagatorHandle handle = j2_propagator_create(&c_init);
    if (!handle) return false;
    COrbitalElements c_out{};
    if (j2_propagator_propagate(handle, c_init.t + timeStep, &c_out) != 0) {
        j2_propagator_destroy(handle);
        return false;
    }
    elements = c_out;
    j2_propagator_destroy(handle);
    return true;
}

bool J2OrbitPropagator::stateToElements(const glm::vec3& position, const glm::vec3& velocity,
                                        OrbitalElements& elements) {
    // 通过上层接口调用核心库进行反解
    SatelliteState st{};
    st.position = position;     // km
    st.velocity = velocity;     // km/s
    st.timestamp = 0.0;
    return J2OrbitPropagator::stateToElements(st, elements) == VisualizationError::SUCCESS;
}




// OrbitVisualizationManager 实现

/**
 * @brief 构造函数
 * @param earthRenderer 地球渲染器引用
 * @param orbitRenderer 轨道渲染器引用
 */
OrbitVisualizationManager::OrbitVisualizationManager(EarthRenderer& earthRenderer, OrbitRenderer& orbitRenderer)
    : earthRenderer(earthRenderer), orbitRenderer(orbitRenderer) {
}

/**
 * @brief 析构函数
 */
OrbitVisualizationManager::~OrbitVisualizationManager() {
    clearAllTasks();
}

/**
 * @brief 初始化管理器
 * @param propagator 轨道传播器
 * @return VisualizationError 初始化结果
 */
VisualizationError OrbitVisualizationManager::initialize(std::shared_ptr<J2OrbitPropagator> propagator) {
    if (!propagator) {
        return VisualizationError::UNKNOWN_ERROR;
    }
    this->propagator = propagator;
    initialized = true;
    return VisualizationError::SUCCESS;
}

/**
 * @brief 添加轨道任务
 * @param name 轨道名称
 * @param elements 轨道元素
 * @param color 轨道颜色
 * @return uint32_t 轨道任务ID
 */
uint32_t OrbitVisualizationManager::addOrbitTask(const std::string& name,
                                                 const OrbitalElements& elements,
                                                 const glm::vec3& color) {
    if (!initialized || !propagator) {
        return 0;
    }
    
    OrbitTask task;
    task.id = nextTaskId++;
    task.name = name;
    task.elements = elements;
    task.color = color;
    task.computed = false;
    task.orbitId = 0;
    task.satelliteId = 0;
    
    orbitTasks[task.id] = task;
    return task.id;
}

/**
 * @brief 执行轨道传播
 * @param taskId 轨道任务ID
 * @param progressCallback 进度回调
 * @return VisualizationError 执行结果
 */
VisualizationError OrbitVisualizationManager::executeOrbitPropagation(uint32_t taskId,
                                                                      ProgressCallback progressCallback) {
    if (!initialized || !propagator) {
        return VisualizationError::UNKNOWN_ERROR;
    }
    
    auto it = orbitTasks.find(taskId);
    if (it == orbitTasks.end()) {
        return VisualizationError::UNKNOWN_ERROR;
    }
    
    OrbitTask& task = it->second;
    
    // 执行轨道传播
    VisualizationError result = propagator->propagateFromElements(task.elements, task.result, progressCallback);
    if (result != VisualizationError::SUCCESS) {
        return result;
    }
    
    task.computed = true;
    
    // 将轨道数据添加到渲染器
    if (!task.result.orbitPoints.empty()) {
        task.orbitId = orbitRenderer.addOrbit(task.result.orbitPoints, task.color, true);
    }
    
    // 添加卫星到渲染器
    if (!task.result.states.empty()) {
        const auto& firstState = task.result.states[0];
        task.satelliteId = orbitRenderer.addSatellite(firstState, task.color, 1.0f, true);
    }
    
    return VisualizationError::SUCCESS;
}

/**
 * @brief 获取轨道数据用于渲染
 * @param taskId 轨道任务ID
 * @param orbitData 输出轨道数据
 * @return VisualizationError 获取结果
 */
VisualizationError OrbitVisualizationManager::getOrbitDataForRendering(uint32_t taskId, OrbitData& orbitData) {
    auto it = orbitTasks.find(taskId);
    if (it == orbitTasks.end() || !it->second.computed) {
        return VisualizationError::UNKNOWN_ERROR;
    }
    
    const OrbitTask& task = it->second;
    orbitData.points = task.result.orbitPoints;
    orbitData.color = task.color;
    orbitData.visible = true;
    
    return VisualizationError::SUCCESS;
}

/**
 * @brief 获取卫星状态用于渲染
 * @param taskId 轨道任务ID
 * @param time 目标时间
 * @param satelliteData 输出卫星数据
 * @return VisualizationError 获取结果
 */
VisualizationError OrbitVisualizationManager::getSatelliteDataForRendering(uint32_t taskId,
                                                                           double time,
                                                                           SatelliteRenderData& satelliteData) {
    auto it = orbitTasks.find(taskId);
    if (it == orbitTasks.end()) {
        return VisualizationError::UNKNOWN_ERROR;
    }
    
    const OrbitTask& task = it->second;
    
    // 检查轨道传播器是否可用
    if (!propagator) {
        return VisualizationError::UNKNOWN_ERROR;
    }
    
    // 实时计算指定时间的卫星状态
    SatelliteState currentState;
    VisualizationError result = propagator->computeStateAtTime(task.elements, time, currentState);
    if (result != VisualizationError::SUCCESS) {
        return result;
    }
    
    // 计算当前时间的轨道元素
    OrbitalElements currentElements;
    result = propagator->stateToElements(currentState, currentElements);
    if (result != VisualizationError::SUCCESS) {
        // 如果轨道元素计算失败，仍然使用状态数据，但设置默认轨道元素
        currentElements = task.elements;
        currentElements.t = time;
    }
    
    // 设置卫星渲染数据
    satelliteData.state = currentState;
    satelliteData.state.timestamp = time;
    satelliteData.color = task.color;
    satelliteData.scale = 1.0f; // 使用默认缩放
    satelliteData.visible = true;
    
    // 注意：轨道元素信息已通过computeStateAtTime和stateToElements计算，
    // 但SatelliteState结构体中没有elements成员，轨道元素信息需要通过其他方式传递
    
    return VisualizationError::SUCCESS;
}

/**
 * @brief 移除轨道任务
 * @param taskId 轨道任务ID
 * @return VisualizationError 移除结果
 */
VisualizationError OrbitVisualizationManager::removeOrbitTask(uint32_t taskId) {
    auto it = orbitTasks.find(taskId);
    if (it == orbitTasks.end()) {
        return VisualizationError::UNKNOWN_ERROR;
    }
    
    const OrbitTask& task = it->second;
    
    // 从渲染器中移除
    if (task.orbitId != 0) {
        orbitRenderer.removeOrbit(task.orbitId);
    }
    if (task.satelliteId != 0) {
        orbitRenderer.removeSatellite(task.satelliteId);
    }
    
    orbitTasks.erase(it);
    return VisualizationError::SUCCESS;
}

/**
 * @brief 清除所有轨道任务
 */
void OrbitVisualizationManager::clearAllTasks() {
    for (const auto& pair : orbitTasks) {
        const OrbitTask& task = pair.second;
        if (task.orbitId != 0) {
            orbitRenderer.removeOrbit(task.orbitId);
        }
        if (task.satelliteId != 0) {
            orbitRenderer.removeSatellite(task.satelliteId);
        }
    }
    orbitTasks.clear();
}

/**
 * @brief 清除所有轨道
 */
void OrbitVisualizationManager::clearAllOrbits() {
    clearAllTasks();
}

/**
 * @brief 设置轨道可见性
 * @param taskId 任务ID
 * @param visible 是否可见
 */
void OrbitVisualizationManager::setOrbitVisible(uint32_t taskId, bool visible) {
    auto it = orbitTasks.find(taskId);
    if (it != orbitTasks.end()) {
        const OrbitTask& task = it->second;
        if (task.orbitId != 0) {
            orbitRenderer.setOrbitVisible(task.orbitId, visible);
        }
        if (task.satelliteId != 0) {
            orbitRenderer.setSatelliteVisible(task.satelliteId, visible);
        }
    }
}

/**
 * @brief 获取任务数量
 * @return size_t 任务数量
 */
size_t OrbitVisualizationManager::getTaskCount() const {
    return orbitTasks.size();
}

/**
 * @brief 获取任务对应的卫星ID
 * @param taskId 轨道任务ID
 * @return uint32_t 卫星ID，如果任务不存在或未计算则返回0
 */
uint32_t OrbitVisualizationManager::getSatelliteId(uint32_t taskId) const {
    auto it = orbitTasks.find(taskId);
    if (it != orbitTasks.end()) {
        return it->second.satelliteId;
    }
    return 0;
}

/**
 * @brief 获取轨道传播器实例
 * @return std::shared_ptr<J2OrbitPropagator> 轨道传播器实例
 */
std::shared_ptr<J2OrbitPropagator> OrbitVisualizationManager::getPropagator() const {
    return propagator;
}

} // namespace j2_orbit_visualization