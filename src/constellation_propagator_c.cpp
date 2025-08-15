/**
 * @file constellation_propagator_c.cpp
 * @brief 星座传播器的C语言封装实现
 */

#include "constellation_propagator_c.h"
#include "constellation_propagator.h"
#include <vector>
#include <cstring>
#include <stdexcept>

// 内部辅助：验证句柄
static ConstellationPropagator* validate_handle(ConstellationPropagatorHandle handle) {
    if (handle == nullptr) return nullptr;
    return static_cast<ConstellationPropagator*>(handle);
}

extern "C" {

J2_API ConstellationPropagatorHandle constellation_propagator_create(double epoch_time) {
    try {
        ConstellationPropagator* ptr = new ConstellationPropagator(epoch_time);
        return static_cast<ConstellationPropagatorHandle>(ptr);
    } catch (const std::exception&) {
        return nullptr;
    }
}

J2_API void constellation_propagator_destroy(ConstellationPropagatorHandle handle) {
    ConstellationPropagator* p = validate_handle(handle);
    if (p) delete p;
}

J2_API int constellation_propagator_add_satellites(ConstellationPropagatorHandle handle,
                                                  const CCompactOrbitalElements* satellites,
                                                  size_t count) {
    ConstellationPropagator* p = validate_handle(handle);
    if (!p || (!satellites && count>0)) return -1;
    try {
        std::vector<CompactOrbitalElements> vec;
        vec.reserve(count);
        for (size_t i=0;i<count;++i) {
            CompactOrbitalElements e{satellites[i].a, satellites[i].e, satellites[i].i, satellites[i].O, satellites[i].w, satellites[i].M};
            vec.push_back(e);
        }
        p->addSatellites(vec);
        return 0;
    } catch (const std::exception&) {
        return -1;
    }
}

J2_API int constellation_propagator_add_satellite(ConstellationPropagatorHandle handle,
                                                 const CCompactOrbitalElements* satellite) {
    ConstellationPropagator* p = validate_handle(handle);
    if (!p || !satellite) return -1;
    try {
        CompactOrbitalElements e{satellite->a, satellite->e, satellite->i, satellite->O, satellite->w, satellite->M};
        p->addSatellite(e);
        return 0;
    } catch (const std::exception&) {
        return -1;
    }
}

J2_API int constellation_propagator_get_satellite_count(ConstellationPropagatorHandle handle,
                                                       size_t* count) {
    ConstellationPropagator* p = validate_handle(handle);
    if (!p || !count) return -1;
    try {
        *count = p->getSatelliteCount();
        return 0;
    } catch (const std::exception&) { return -1; }
}

J2_API int constellation_propagator_propagate(ConstellationPropagatorHandle handle, double target_time) {
    ConstellationPropagator* p = validate_handle(handle);
    if (!p) return -1;
    try { p->propagateConstellation(target_time); return 0; }
    catch (const std::exception&) { return -1; }
}

J2_API int constellation_propagator_get_satellite_elements(ConstellationPropagatorHandle handle,
                                                          size_t satellite_id,
                                                          CCompactOrbitalElements* elements) {
    ConstellationPropagator* p = validate_handle(handle);
    if (!p || !elements) return -1;
    try {
        auto e = p->getSatelliteElements(satellite_id);
        elements->a = e.a; elements->e = e.e; elements->i = e.i; elements->O = e.O; elements->w = e.w; elements->M = e.M;
        return 0;
    } catch (const std::exception&) { return -1; }
}

J2_API int constellation_propagator_get_satellite_state(ConstellationPropagatorHandle handle,
                                                       size_t satellite_id,
                                                       CStateVector* state) {
    ConstellationPropagator* p = validate_handle(handle);
    if (!p || !state) return -1;
    try {
        auto s = p->getSatelliteState(satellite_id);
        state->r[0] = s.r.x(); state->r[1] = s.r.y(); state->r[2] = s.r.z();
        state->v[0] = s.v.x(); state->v[1] = s.v.y(); state->v[2] = s.v.z();
        return 0;
    } catch (const std::exception&) { return -1; }
}

J2_API int constellation_propagator_get_all_positions(ConstellationPropagatorHandle handle,
                                                     double* positions,
                                                     size_t* count) {
    ConstellationPropagator* p = validate_handle(handle);
    if (!p || !positions || !count) return -1;
    try {
        size_t n = p->getSatelliteCount();
        if (*count < n) return -2; // 缓冲区不足
        auto mat = p->getAllPositions();
        // mat (3 x N)
        for (size_t j=0;j<n;++j) {
            positions[3*j+0] = mat(0,j);
            positions[3*j+1] = mat(1,j);
            positions[3*j+2] = mat(2,j);
        }
        *count = n;
        return 0;
    } catch (const std::exception&) { return -1; }
}

J2_API int constellation_propagator_apply_impulse_to_constellation(ConstellationPropagatorHandle handle,
                                                                  const double* delta_vs,
                                                                  size_t count,
                                                                  double impulse_time) {
    ConstellationPropagator* p = validate_handle(handle);
    if (!p || (!delta_vs && count>0)) return -1;
    try {
        std::vector<Eigen::Vector3d> dvs;
        dvs.reserve(count);
        for (size_t j=0;j<count;++j) {
            dvs.emplace_back(delta_vs[3*j+0], delta_vs[3*j+1], delta_vs[3*j+2]);
        }
        p->applyImpulseToConstellation(dvs, impulse_time);
        return 0;
    } catch (const std::exception&) { return -1; }
}

J2_API int constellation_propagator_apply_impulse_to_satellites(ConstellationPropagatorHandle handle,
                                                               const size_t* satellite_ids,
                                                               const double* delta_vs,
                                                               size_t count,
                                                               double impulse_time) {
    ConstellationPropagator* p = validate_handle(handle);
    if (!p || (!satellite_ids && count>0) || (!delta_vs && count>0)) return -1;
    try {
        std::vector<size_t> ids(satellite_ids, satellite_ids+count);
        std::vector<Eigen::Vector3d> dvs; dvs.reserve(count);
        for (size_t j=0;j<count;++j) {
            dvs.emplace_back(delta_vs[3*j+0], delta_vs[3*j+1], delta_vs[3*j+2]);
        }
        p->applyImpulseToSatellites(ids, dvs, impulse_time);
        return 0;
    } catch (const std::exception&) { return -1; }
}

J2_API int constellation_propagator_set_step_size(ConstellationPropagatorHandle handle, double step_size) {
    ConstellationPropagator* p = validate_handle(handle);
    if (!p || step_size<=0) return -1;
    try { p->setStepSize(step_size); return 0; } catch (const std::exception&) { return -1; }
}

J2_API int constellation_propagator_set_compute_mode(ConstellationPropagatorHandle handle, ComputeMode mode) {
    ConstellationPropagator* p = validate_handle(handle);
    if (!p) return -1;
    try {
        switch (mode) {
            case COMPUTE_MODE_CPU_SCALAR: p->setComputeMode(ConstellationPropagator::CPU_SCALAR); break;
            case COMPUTE_MODE_CPU_SIMD:   p->setComputeMode(ConstellationPropagator::CPU_SIMD);   break;
            case COMPUTE_MODE_GPU_CUDA:   p->setComputeMode(ConstellationPropagator::GPU_CUDA);   break;
            default: return -1;
        }
        return 0;
    } catch (const std::exception&) { return -1; }
}

J2_API int constellation_propagator_set_adaptive_step_size(ConstellationPropagatorHandle handle, int enable) {
    ConstellationPropagator* p = validate_handle(handle);
    if (!p) return -1;
    try { p->setAdaptiveStepSize(enable!=0); return 0; } catch (const std::exception&) { return -1; }
}

J2_API int constellation_propagator_set_adaptive_parameters(ConstellationPropagatorHandle handle,
                                                           double tolerance,
                                                           double min_step,
                                                           double max_step) {
    ConstellationPropagator* p = validate_handle(handle);
    if (!p || tolerance<=0 || min_step<=0 || max_step<=min_step) return -1;
    try { p->setAdaptiveParameters(tolerance, min_step, max_step); return 0; } catch (const std::exception&) { return -1; }
}

J2_API int constellation_propagator_is_cuda_available(void) {
    try { return ConstellationPropagator::isCudaAvailable() ? 1 : 0; }
    catch (...) { return 0; }
}

} // extern "C"