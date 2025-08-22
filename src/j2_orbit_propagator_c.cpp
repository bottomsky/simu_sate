/**
 * @file j2_orbit_propagator_c.cpp
 * @brief J2轨道传播器的C语言封装实现
 *
 * 该文件提供了J2轨道传播器的C语言接口，使得外部语言（如Python、C#等）
 * 可以通过动态库调用J2轨道传播功能。
 */

#include "j2_orbit_propagator_c.h"
#include "j2_orbit_propagator.h"
#include "math_defs.h"
#include <cstring>
#include <stdexcept>

// === 内部辅助函数 ===

/**
 * @brief 将C格式的轨道要素转换为C++格式
 */
static OrbitalElements c_to_cpp_elements(const COrbitalElements* c_elements) {
    OrbitalElements cpp_elements;
    cpp_elements.a = c_elements->a;
    cpp_elements.e = c_elements->e;
    cpp_elements.i = c_elements->i;
    cpp_elements.O = c_elements->O;
    cpp_elements.w = c_elements->w;
    cpp_elements.M = c_elements->M;
    cpp_elements.t = c_elements->t;
    return cpp_elements;
}

/**
 * @brief 将C++格式的轨道要素转换为C格式
 */
static void cpp_to_c_elements(const OrbitalElements& cpp_elements, COrbitalElements* c_elements) {
    c_elements->a = cpp_elements.a;
    c_elements->e = cpp_elements.e;
    c_elements->i = cpp_elements.i;
    c_elements->O = cpp_elements.O;
    c_elements->w = cpp_elements.w;
    c_elements->M = cpp_elements.M;
    c_elements->t = cpp_elements.t;
}

/**
 * @brief 将C格式的状态向量转换为C++格式
 */
static StateVector c_to_cpp_state(const CStateVector* c_state) {
    StateVector cpp_state;
    cpp_state.r = Eigen::Vector3d(c_state->r[0], c_state->r[1], c_state->r[2]);
    cpp_state.v = Eigen::Vector3d(c_state->v[0], c_state->v[1], c_state->v[2]);
    return cpp_state;
}

/**
 * @brief 将C++格式的状态向量转换为C格式
 */
static void cpp_to_c_state(const StateVector& cpp_state, CStateVector* c_state) {
    c_state->r[0] = cpp_state.r.x();
    c_state->r[1] = cpp_state.r.y();
    c_state->r[2] = cpp_state.r.z();
    c_state->v[0] = cpp_state.v.x();
    c_state->v[1] = cpp_state.v.y();
    c_state->v[2] = cpp_state.v.z();
}

/**
 * @brief 验证句柄的有效性
 */
static J2OrbitPropagator* validate_handle(J2PropagatorHandle handle) {
    if (handle == nullptr) {
        return nullptr;
    }
    return static_cast<J2OrbitPropagator*>(handle);
}

// === 创建和销毁函数 ===

extern "C" {

J2PropagatorHandle j2_propagator_create(const COrbitalElements* initial_elements) {
    if (initial_elements == nullptr) {
        return nullptr;
    }
    
    try {
        OrbitalElements cpp_elements = c_to_cpp_elements(initial_elements);
        J2OrbitPropagator* propagator = new J2OrbitPropagator(cpp_elements);
        return static_cast<J2PropagatorHandle>(propagator);
    } catch (const std::exception&) {
        return nullptr;
    }
}

void j2_propagator_destroy(J2PropagatorHandle handle) {
    J2OrbitPropagator* propagator = validate_handle(handle);
    if (propagator != nullptr) {
        delete propagator;
    }
}

// === 轨道传播函数 ===

int j2_propagator_propagate(J2PropagatorHandle handle, double target_time, COrbitalElements* result) {
    J2OrbitPropagator* propagator = validate_handle(handle);
    if (propagator == nullptr || result == nullptr) {
        return -1;
    }
    
    try {
        OrbitalElements cpp_result = propagator->propagate(target_time);
        cpp_to_c_elements(cpp_result, result);
        return 0;
    } catch (const std::exception&) {
        return -1;
    }
}

int j2_propagator_elements_to_state(J2PropagatorHandle handle, const COrbitalElements* elements, CStateVector* state) {
    J2OrbitPropagator* propagator = validate_handle(handle);
    if (propagator == nullptr || elements == nullptr || state == nullptr) {
        return -1;
    }
    
    try {
        OrbitalElements cpp_elements = c_to_cpp_elements(elements);
        StateVector cpp_state = propagator->elementsToState(cpp_elements);
        cpp_to_c_state(cpp_state, state);
        return 0;
    } catch (const std::exception&) {
        return -1;
    }
}

int j2_propagator_state_to_elements(J2PropagatorHandle handle, const CStateVector* state, double time, COrbitalElements* elements) {
    J2OrbitPropagator* propagator = validate_handle(handle);
    if (propagator == nullptr || state == nullptr || elements == nullptr) {
        return -1;
    }
    
    try {
        StateVector cpp_state = c_to_cpp_state(state);
        OrbitalElements cpp_elements = propagator->stateToElements(cpp_state, time);
        cpp_to_c_elements(cpp_elements, elements);
        return 0;
    } catch (const std::exception&) {
        return -1;
    }
}

int j2_propagator_apply_impulse(J2PropagatorHandle handle, const COrbitalElements* elements, const double delta_v[3], double t, COrbitalElements* result) {
    J2OrbitPropagator* propagator = validate_handle(handle);
    if (propagator == nullptr || elements == nullptr || delta_v == nullptr || result == nullptr) {
        return -1;
    }

    try {
        OrbitalElements cpp_elements = c_to_cpp_elements(elements);
        Eigen::Vector3d dv(delta_v[0], delta_v[1], delta_v[2]);
        OrbitalElements updated = propagator->applyImpulse(cpp_elements, dv, t);
        cpp_to_c_elements(updated, result);
        return 0;
    } catch (const std::exception&) {
        return -1;
    }
}

// === 参数设置函数 ===

int j2_propagator_set_step_size(J2PropagatorHandle handle, double step_size) {
    J2OrbitPropagator* propagator = validate_handle(handle);
    if (propagator == nullptr || step_size <= 0) {
        return -1;
    }
    
    try {
        propagator->setStepSize(step_size);
        return 0;
    } catch (const std::exception&) {
        return -1;
    }
}

int j2_propagator_get_step_size(J2PropagatorHandle handle, double* step_size) {
    J2OrbitPropagator* propagator = validate_handle(handle);
    if (propagator == nullptr || step_size == nullptr) {
        return -1;
    }
    
    try {
        *step_size = propagator->getStepSize();
        return 0;
    } catch (const std::exception&) {
        return -1;
    }
}

int j2_propagator_set_adaptive_step_size(J2PropagatorHandle handle, int enable) {
    J2OrbitPropagator* propagator = validate_handle(handle);
    if (propagator == nullptr) {
        return -1;
    }
    
    try {
        propagator->setAdaptiveStepSize(enable != 0);
        return 0;
    } catch (const std::exception&) {
        return -1;
    }
}

int j2_propagator_set_adaptive_parameters(J2PropagatorHandle handle, double tolerance, double min_step, double max_step) {
    J2OrbitPropagator* propagator = validate_handle(handle);
    if (propagator == nullptr || tolerance <= 0 || min_step <= 0 || max_step <= min_step) {
        return -1;
    }
    
    try {
        propagator->setAdaptiveParameters(tolerance, min_step, max_step);
        return 0;
    } catch (const std::exception&) {
        return -1;
    }
}

// === 坐标转换函数 ===

int j2_eci_to_ecef_position(const double eci_position[3], double utc_seconds, double ecef_position[3]) {
    if (eci_position == nullptr || ecef_position == nullptr) {
        return -1;
    }
    
    try {
        Eigen::Vector3d eci_pos(eci_position[0], eci_position[1], eci_position[2]);
        Eigen::Vector3d ecef_pos = eciToEcefPosition(eci_pos, utc_seconds);
        
        ecef_position[0] = ecef_pos.x();
        ecef_position[1] = ecef_pos.y();
        ecef_position[2] = ecef_pos.z();
        return 0;
    } catch (const std::exception&) {
        return -1;
    }
}

int j2_ecef_to_eci_position(const double ecef_position[3], double utc_seconds, double eci_position[3]) {
    if (ecef_position == nullptr || eci_position == nullptr) {
        return -1;
    }
    
    try {
        Eigen::Vector3d ecef_pos(ecef_position[0], ecef_position[1], ecef_position[2]);
        Eigen::Vector3d eci_pos = ecefToEciPosition(ecef_pos, utc_seconds);
        
        eci_position[0] = eci_pos.x();
        eci_position[1] = eci_pos.y();
        eci_position[2] = eci_pos.z();
        return 0;
    } catch (const std::exception&) {
        return -1;
    }
}

int j2_eci_to_ecef_velocity(const double eci_position[3], const double eci_velocity[3], double utc_seconds, double ecef_velocity[3]) {
    if (eci_position == nullptr || eci_velocity == nullptr || ecef_velocity == nullptr) {
        return -1;
    }
    
    try {
        Eigen::Vector3d eci_pos(eci_position[0], eci_position[1], eci_position[2]);
        Eigen::Vector3d eci_vel(eci_velocity[0], eci_velocity[1], eci_velocity[2]);
        Eigen::Vector3d ecef_vel = eciToEcefVelocity(eci_pos, eci_vel, utc_seconds);
        
        ecef_velocity[0] = ecef_vel.x();
        ecef_velocity[1] = ecef_vel.y();
        ecef_velocity[2] = ecef_vel.z();
        return 0;
    } catch (const std::exception&) {
        return -1;
    }
}

int j2_ecef_to_eci_velocity(const double ecef_position[3], const double ecef_velocity[3], double utc_seconds, double eci_velocity[3]) {
    if (ecef_position == nullptr || ecef_velocity == nullptr || eci_velocity == nullptr) {
        return -1;
    }
    try {
        Eigen::Vector3d ecef_pos(ecef_position[0], ecef_position[1], ecef_position[2]);
        Eigen::Vector3d ecef_vel(ecef_velocity[0], ecef_velocity[1], ecef_velocity[2]);
        Eigen::Vector3d eci_vel = ecefToEciVelocity(ecef_pos, ecef_vel, utc_seconds);
        eci_velocity[0] = eci_vel.x();
        eci_velocity[1] = eci_vel.y();
        eci_velocity[2] = eci_vel.z();
        return 0;
    } catch (const std::exception&) {
        return -1;
    }
}

int j2_ecef_to_geodetic(const double ecef_position[3], double geodetic_llh[3]) {
    if (ecef_position == nullptr || geodetic_llh == nullptr) {
        return -1;
    }
    try {
        Eigen::Vector3d ecef_pos(ecef_position[0], ecef_position[1], ecef_position[2]);
        Eigen::Vector3d geo = ecefToGeodeticVec(ecef_pos);
        geodetic_llh[0] = geo.x(); // lat (rad)
        geodetic_llh[1] = geo.y(); // lon (rad)
        geodetic_llh[2] = geo.z(); // h (m)
        return 0;
    } catch (const std::exception&) {
        return -1;
    }
}

int j2_geodetic_to_ecef(const double geodetic_llh[3], double ecef_position[3]) {
    if (geodetic_llh == nullptr || ecef_position == nullptr) {
        return -1;
    }
    try {
        Eigen::Vector3d geo(geodetic_llh[0], geodetic_llh[1], geodetic_llh[2]);
        Eigen::Vector3d ecef = geodeticToEcefVec(geo);
        ecef_position[0] = ecef.x();
        ecef_position[1] = ecef.y();
        ecef_position[2] = ecef.z();
        return 0;
    } catch (const std::exception&) {
        return -1;
    }
}

int j2_eci_to_geodetic(const double eci_position[3], double utc_seconds, double geodetic_llh[3]) {
    if (eci_position == nullptr || geodetic_llh == nullptr) {
        return -1;
    }
    try {
        Eigen::Vector3d eci_pos(eci_position[0], eci_position[1], eci_position[2]);
        Eigen::Vector3d geo = eciToGeodeticVec(eci_pos, utc_seconds);
        geodetic_llh[0] = geo.x();
        geodetic_llh[1] = geo.y();
        geodetic_llh[2] = geo.z();
        return 0;
    } catch (const std::exception&) {
        return -1;
    }
}

int j2_geodetic_to_eci(const double geodetic_llh[3], double utc_seconds, double eci_position[3]) {
    if (geodetic_llh == nullptr || eci_position == nullptr) {
        return -1;
    }
    try {
        Eigen::Vector3d geo(geodetic_llh[0], geodetic_llh[1], geodetic_llh[2]);
        Eigen::Vector3d eci = geodeticToEciVec(geo, utc_seconds);
        eci_position[0] = eci.x();
        eci_position[1] = eci.y();
        eci_position[2] = eci.z();
        return 0;
    } catch (const std::exception&) {
        return -1;
    }
}

// === RTN/ECI 坐标转换函数实现 ===
int j2_rtn_to_eci_rotation(const double r_eci[3], const double v_eci[3], double R_out[9]) {
    if (r_eci == nullptr || v_eci == nullptr || R_out == nullptr) {
        return -1;
    }
    try {
        Eigen::Vector3d r(r_eci[0], r_eci[1], r_eci[2]);
        Eigen::Vector3d v(v_eci[0], v_eci[1], v_eci[2]);
        Eigen::Matrix3d R = rtnToEciRotationMatrix(r, v);
        // 行优先输出
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                R_out[i * 3 + j] = R(i, j);
            }
        }
        return 0;
    } catch (const std::exception&) {
        return -1;
    }
}

int j2_eci_to_rtn_rotation(const double r_eci[3], const double v_eci[3], double R_out[9]) {
    if (r_eci == nullptr || v_eci == nullptr || R_out == nullptr) {
        return -1;
    }
    try {
        Eigen::Vector3d r(r_eci[0], r_eci[1], r_eci[2]);
        Eigen::Vector3d v(v_eci[0], v_eci[1], v_eci[2]);
        Eigen::Matrix3d R = eciToRtnRotationMatrix(r, v);
        // 行优先输出
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                R_out[i * 3 + j] = R(i, j);
            }
        }
        return 0;
    } catch (const std::exception&) {
        return -1;
    }
}

int j2_eci_to_rtn_vector(const double r_eci[3], const double v_eci[3], const double vec_eci[3], double vec_rtn[3]) {
    if (r_eci == nullptr || v_eci == nullptr || vec_eci == nullptr || vec_rtn == nullptr) {
        return -1;
    }
    try {
        Eigen::Vector3d r(r_eci[0], r_eci[1], r_eci[2]);
        Eigen::Vector3d v(v_eci[0], v_eci[1], v_eci[2]);
        Eigen::Vector3d vec(vec_eci[0], vec_eci[1], vec_eci[2]);
        Eigen::Vector3d out = eciToRtnVector(r, v, vec);
        vec_rtn[0] = out.x();
        vec_rtn[1] = out.y();
        vec_rtn[2] = out.z();
        return 0;
    } catch (const std::exception&) {
        return -1;
    }
}

int j2_rtn_to_eci_vector(const double r_eci[3], const double v_eci[3], const double vec_rtn[3], double vec_eci[3]) {
    if (r_eci == nullptr || v_eci == nullptr || vec_rtn == nullptr || vec_eci == nullptr) {
        return -1;
    }
    try {
        Eigen::Vector3d r(r_eci[0], r_eci[1], r_eci[2]);
        Eigen::Vector3d v(v_eci[0], v_eci[1], v_eci[2]);
        Eigen::Vector3d vec(vec_rtn[0], vec_rtn[1], vec_rtn[2]);
        Eigen::Vector3d out = rtnToEciVector(r, v, vec);
        vec_eci[0] = out.x();
        vec_eci[1] = out.y();
        vec_eci[2] = out.z();
        return 0;
    } catch (const std::exception&) {
        return -1;
    }
}

int j2_compute_gmst(double utc_seconds, double* gmst) {
    if (gmst == nullptr) {
        return -1;
    }
    try {
        *gmst = computeGMST(utc_seconds);
        return 0;
    } catch (const std::exception&) {
        return -1;
    }
}

double j2_normalize_angle(double angle) {
    return normalizeAngle(angle);
}

} // extern "C"