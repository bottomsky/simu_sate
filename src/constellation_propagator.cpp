#include "constellation_propagator.h"
#include "j2_orbit_propagator.h"  // 用于CUDA路径中的状态->要素转换
#include <algorithm>
#include <array>
#include <cstring>
#include <limits>
#if defined(__AVX2__) || defined(__AVX__)
#include <immintrin.h>
#endif
#if defined(HAVE_CUDA_TOOLKIT) && HAVE_CUDA_TOOLKIT
#include <cuda_runtime_api.h>
#endif

namespace {

std::array<double, 3> computeJ2SecularRates(double semi_major_axis,
                                            double eccentricity,
                                            double inclination) {
    std::array<double, 3> rates{0.0, 0.0, 0.0};

    if (semi_major_axis <= 0.0) {
        return rates;
    }

    double one_minus_e2 = 1.0 - eccentricity * eccentricity;
    if (std::abs(one_minus_e2) < EPSILON) {
        return rates;
    }

    double mean_motion = std::sqrt(MU / (semi_major_axis * semi_major_axis * semi_major_axis));
    double p = semi_major_axis * one_minus_e2;
    double re_over_p = RE / p;
    double re_over_p_sq = re_over_p * re_over_p;
    double factor = 1.5 * J2 * mean_motion * re_over_p_sq;
    double cos_i = std::cos(inclination);
    double cos_i_sq = cos_i * cos_i;

    rates[0] = -factor * cos_i;  // d(O)/dt
    rates[1] = 0.5 * factor * (5.0 * cos_i_sq - 1.0);  // d(w)/dt
    rates[2] = mean_motion +
               0.5 * factor * std::sqrt(one_minus_e2) * (3.0 * cos_i_sq - 1.0);  // d(M)/dt

    return rates;
}

}  // namespace

ConstellationPropagator::ConstellationPropagator(double epoch_time)
    : epoch_time_(epoch_time), current_time_(epoch_time), step_size_(60.0), 
      compute_mode_(CPU_SIMD) {
#if defined(HAVE_CUDA_TOOLKIT) && HAVE_CUDA_TOOLKIT
    d_a_ = d_e_ = d_i_ = d_O_ = d_w_ = d_M_ = nullptr;
    d_x_ = d_y_ = d_z_ = nullptr;
    gpu_buffer_size_ = 0;
    cuda_stream_ = 0;
#endif
    sample_interval_ = step_size_;
    steps_per_sample_ = 1;
    // 自动检测 CUDA 并优先启用 GPU 模式（当运行环境支持且构建启用 CUDA 时）
    if (isCudaAvailable()) {
        compute_mode_ = GPU_CUDA;
    }
}

ConstellationPropagator::~ConstellationPropagator() {
#if defined(HAVE_CUDA_TOOLKIT) && HAVE_CUDA_TOOLKIT
    cleanupCUDA();
#endif
}

void ConstellationPropagator::addSatellites(const std::vector<CompactOrbitalElements>& satellites) {
    ensureHostElementsUpToDate();

    size_t old_size = elements_.size();
    size_t new_size = old_size + satellites.size();
    
    elements_.resize(new_size);
    
    // 批量复制数据 (SoA格式优化缓存访问)
    for (size_t i = 0; i < satellites.size(); ++i) {
        size_t idx = old_size + i;
        elements_.a[idx] = satellites[i].a;
        elements_.e[idx] = satellites[i].e;
        elements_.i[idx] = satellites[i].i;
        elements_.O[idx] = satellites[i].O;
        elements_.w[idx] = satellites[i].w;
        elements_.M[idx] = satellites[i].M;
    }

    markDeviceElementsDirty();
}

void ConstellationPropagator::addSatellite(const CompactOrbitalElements& satellite) {
    ensureHostElementsUpToDate();

    size_t idx = elements_.size();
    elements_.resize(idx + 1);
    
    elements_.a[idx] = satellite.a;
    elements_.e[idx] = satellite.e;
    elements_.i[idx] = satellite.i;
    elements_.O[idx] = satellite.O;
    elements_.w[idx] = satellite.w;
    elements_.M[idx] = satellite.M;

    markDeviceElementsDirty();
}

void ConstellationPropagator::setStepSize(double step) {
    if (step <= 0.0) {
        throw std::invalid_argument("Step size must be positive");
    }
    step_size_ = step;
    recalcSampleStride();
}

void ConstellationPropagator::setSampleInterval(double interval) {
    if (interval <= 0.0) {
        throw std::invalid_argument("Sample interval must be positive");
    }
    sample_interval_ = interval;
    recalcSampleStride();
}

void ConstellationPropagator::recalcSampleStride() {
    const double ratio = sample_interval_ / step_size_;
    if (ratio < 1.0 - 1e-9) {
        throw std::invalid_argument("Sample interval must not be smaller than step size");
    }
    const double rounded = std::round(ratio);
    if (std::abs(rounded - ratio) > 1e-8) {
        throw std::invalid_argument("Sample interval must be an integer multiple of step size");
    }
    steps_per_sample_ = std::max<size_t>(1, static_cast<size_t>(rounded));
    sample_interval_ = steps_per_sample_ * step_size_;
}

void ConstellationPropagator::propagateSamples(size_t sample_count) {
    if (sample_count == 0) {
        return;
    }

    for (size_t s = 0; s < sample_count; ++s) {
        integrateSteps(steps_per_sample_);
        current_time_ += step_size_ * static_cast<double>(steps_per_sample_);
    }
}

void ConstellationPropagator::integrateSteps(size_t steps) {
    if (steps == 0) {
        return;
    }

    switch (compute_mode_) {
        case CPU_SCALAR: {
            for (size_t iter = 0; iter < steps; ++iter) {
                propagateScalar(step_size_);
            }
            break;
        }
        case CPU_SIMD: {
            for (size_t iter = 0; iter < steps; ++iter) {
                propagateSIMD(step_size_);
            }
            break;
        }
        case GPU_CUDA: {
            if (isCudaAvailable()) {
                propagateCUDA(step_size_, steps);
            } else {
                std::cerr << "CUDA not available, falling back to SIMD" << std::endl;
                for (size_t iter = 0; iter < steps; ++iter) {
                    propagateSIMD(step_size_);
                }
            }
            break;
        }
    }
}

void ConstellationPropagator::integrateRemainder(double dt) {
    if (dt <= EPSILON) {
        return;
    }

    switch (compute_mode_) {
        case CPU_SCALAR:
            propagateScalar(dt);
            break;
        case CPU_SIMD:
            propagateSIMD(dt);
            break;
        case GPU_CUDA:
            if (isCudaAvailable()) {
                propagateCUDA(dt, 1);
            } else {
                std::cerr << "CUDA not available, falling back to SIMD" << std::endl;
                propagateSIMD(dt);
            }
            break;
    }
}

void ConstellationPropagator::propagateConstellation(double target_time) {
    double dt_total = target_time - current_time_;
    
    if (dt_total < EPSILON) {
        return;
    }
    
    // 分步积分（固定步长）
    size_t steps = 0;
    double remainder = dt_total;
    if (step_size_ > EPSILON) {
        double raw_steps = std::floor(dt_total / step_size_);
        if (raw_steps > 0.0) {
            steps = static_cast<size_t>(raw_steps);
            remainder = dt_total - step_size_ * raw_steps;
        }
        if (remainder < EPSILON) {
            remainder = 0.0;
        } else if (remainder > step_size_ - EPSILON) {
            ++steps;
            remainder = 0.0;
        }
    }

    integrateSteps(steps);
    integrateRemainder(remainder);

    current_time_ = target_time;
}

void ConstellationPropagator::propagateScalarRange(size_t begin, size_t end, double dt) {
    auto computeDerivatives = [&](const CompactOrbitalElements& e) -> std::array<double, 3> {
        return computeJ2SecularRates(e.a, e.e, e.i);
    };

    for (size_t i = begin; i < end; ++i) {
        CompactOrbitalElements elem{elements_.a[i], elements_.e[i], elements_.i[i],
                                   elements_.O[i], elements_.w[i], elements_.M[i]};

        auto k1 = computeDerivatives(elem);

        CompactOrbitalElements temp = elem;
        temp.O = normalizeAngle(temp.O + k1[0] * dt / 2.0);
        temp.w = normalizeAngle(temp.w + k1[1] * dt / 2.0);
        temp.M = normalizeAngle(temp.M + k1[2] * dt / 2.0);
        auto k2 = computeDerivatives(temp);

        temp = elem;
        temp.O = normalizeAngle(temp.O + k2[0] * dt / 2.0);
        temp.w = normalizeAngle(temp.w + k2[1] * dt / 2.0);
        temp.M = normalizeAngle(temp.M + k2[2] * dt / 2.0);
        auto k3 = computeDerivatives(temp);

        temp = elem;
        temp.O = normalizeAngle(temp.O + k3[0] * dt);
        temp.w = normalizeAngle(temp.w + k3[1] * dt);
        temp.M = normalizeAngle(temp.M + k3[2] * dt);
        auto k4 = computeDerivatives(temp);

        elements_.O[i] = normalizeAngle(elem.O + (k1[0] + 2.0 * k2[0] + 2.0 * k3[0] + k4[0]) * dt / 6.0);
        elements_.w[i] = normalizeAngle(elem.w + (k1[1] + 2.0 * k2[1] + 2.0 * k3[1] + k4[1]) * dt / 6.0);
        elements_.M[i] = normalizeAngle(elem.M + (k1[2] + 2.0 * k2[2] + 2.0 * k3[2] + k4[2]) * dt / 6.0);
    }
}

void ConstellationPropagator::propagateScalar(double dt) {
    ensureHostElementsUpToDate();

    propagateScalarRange(0, elements_.size(), dt);

    markDeviceElementsDirty();
}

void ConstellationPropagator::propagateSIMD(double dt) {
    ensureHostElementsUpToDate();

    size_t n = elements_.size();
    size_t simd_count = (n / 4) * 4;  // AVX2处理4个double
    
    // SIMD常数
    const __m256d mu_vec = _mm256_set1_pd(MU);
    const __m256d re_vec = _mm256_set1_pd(RE);
    const __m256d j2_vec = _mm256_set1_pd(J2);
    const __m256d dt_vec = _mm256_set1_pd(dt);
    const __m256d two = _mm256_set1_pd(2.0);
    const __m256d half = _mm256_set1_pd(0.5);
    const __m256d six = _mm256_set1_pd(6.0);
    const __m256d one = _mm256_set1_pd(1.0);
    
    // SIMD RK4积分器
    auto computeDerivativesSIMD = [&](__m256d a_vec, __m256d e_vec, __m256d i_vec) -> std::array<__m256d, 3> {
        // 计算平均角速度 n = sqrt(MU / a^3)
        __m256d a3 = _mm256_mul_pd(_mm256_mul_pd(a_vec, a_vec), a_vec);
        __m256d mean_motion_vec = _mm256_sqrt_pd(_mm256_div_pd(mu_vec, a3));
        
        // 避免奇异性检查（简化）
        __m256d e2 = _mm256_mul_pd(e_vec, e_vec);
        __m256d one_minus_e2 = _mm256_sub_pd(one, e2);
        
        // 计算J2摄动参数，与标量实现一致：factor = (3/2) * J2 * n * (RE/p)^2
        __m256d p = _mm256_mul_pd(a_vec, one_minus_e2);
        __m256d re_over_p = _mm256_div_pd(re_vec, p);
        __m256d re_over_p_sq = _mm256_mul_pd(re_over_p, re_over_p);
        __m256d factor_norm = _mm256_mul_pd(
            _mm256_mul_pd(_mm256_mul_pd(_mm256_set1_pd(1.5), j2_vec), mean_motion_vec), re_over_p_sq);
        
        // 计算三角函数
        __m256d cos_i;
        alignas(32) double i_vals[4];
        _mm256_store_pd(i_vals, i_vec);
        cos_i = _mm256_set_pd(std::cos(i_vals[3]), std::cos(i_vals[2]), std::cos(i_vals[1]), std::cos(i_vals[0]));
        __m256d cos2_i = _mm256_mul_pd(cos_i, cos_i);

        // 计算导数
        __m256d dO_dt = _mm256_mul_pd(_mm256_sub_pd(_mm256_setzero_pd(), factor_norm), cos_i);
        __m256d dw_dt = _mm256_mul_pd(_mm256_mul_pd(half, factor_norm),
                                      _mm256_sub_pd(_mm256_mul_pd(_mm256_set1_pd(5.0), cos2_i), one));
        __m256d sqrt_one_minus_e2 = _mm256_sqrt_pd(one_minus_e2);
        __m256d dM_term = _mm256_mul_pd(_mm256_mul_pd(half, factor_norm),
                                        _mm256_mul_pd(sqrt_one_minus_e2,
                                                      _mm256_sub_pd(_mm256_mul_pd(_mm256_set1_pd(3.0), cos2_i), one)));
        __m256d dM_dt = _mm256_add_pd(mean_motion_vec, dM_term);

        return {dO_dt, dw_dt, dM_dt};
    };
    
    // 批量RK4处理 (每次4个卫星)
    for (size_t i = 0; i < simd_count; i += 4) {
        // 加载轨道要素
        __m256d a_vec = _mm256_load_pd(&elements_.a[i]);
        __m256d e_vec = _mm256_load_pd(&elements_.e[i]);
        __m256d i_vec = _mm256_load_pd(&elements_.i[i]);
        __m256d O_vec = _mm256_load_pd(&elements_.O[i]);
        __m256d w_vec = _mm256_load_pd(&elements_.w[i]);
        __m256d M_vec = _mm256_load_pd(&elements_.M[i]);
        
        // k1 = f(t, y)
        auto k1 = computeDerivativesSIMD(a_vec, e_vec, i_vec);
        
        // k2 = f(t + dt/2, y + k1*dt/2)
        // 对于J2摄动，a、e、i保持不变，只需要更新O、w、M进行导数计算
        // 由于computeDerivativesSIMD只使用a、e、i，所以这里直接使用原始值即可
        auto k2 = computeDerivativesSIMD(a_vec, e_vec, i_vec);
        
        // k3 = f(t + dt/2, y + k2*dt/2)
        auto k3 = computeDerivativesSIMD(a_vec, e_vec, i_vec);
        
        // k4 = f(t + dt, y + k3*dt)
        auto k4 = computeDerivativesSIMD(a_vec, e_vec, i_vec);
        
        // RK4最终更新：y = y + (k1 + 2*k2 + 2*k3 + k4) * dt / 6
        __m256d dO_final = _mm256_mul_pd(
            _mm256_add_pd(_mm256_add_pd(k1[0], _mm256_mul_pd(two, k2[0])),
                         _mm256_add_pd(_mm256_mul_pd(two, k3[0]), k4[0])),
            _mm256_div_pd(dt_vec, six));
        
        __m256d dw_final = _mm256_mul_pd(
            _mm256_add_pd(_mm256_add_pd(k1[1], _mm256_mul_pd(two, k2[1])),
                         _mm256_add_pd(_mm256_mul_pd(two, k3[1]), k4[1])),
            _mm256_div_pd(dt_vec, six));
        
        __m256d dM_final = _mm256_mul_pd(
            _mm256_add_pd(_mm256_add_pd(k1[2], _mm256_mul_pd(two, k2[2])),
                         _mm256_add_pd(_mm256_mul_pd(two, k3[2]), k4[2])),
            _mm256_div_pd(dt_vec, six));
        
        // 更新轨道要素
        O_vec = _mm256_add_pd(O_vec, dO_final);
        w_vec = _mm256_add_pd(w_vec, dw_final);
        M_vec = _mm256_add_pd(M_vec, dM_final);
        
        // 存储结果
        _mm256_store_pd(&elements_.O[i], O_vec);
        _mm256_store_pd(&elements_.w[i], w_vec);
        _mm256_store_pd(&elements_.M[i], M_vec);
    }
    
    if (simd_count < n) {
        propagateScalarRange(simd_count, n, dt);
    }

    // 批量角度归一化
    normalizeAnglesSIMD(elements_.O);
    normalizeAnglesSIMD(elements_.w);
    normalizeAnglesSIMD(elements_.M);

    markDeviceElementsDirty();
}

void ConstellationPropagator::normalizeAnglesSIMD(std::vector<double, Eigen::aligned_allocator<double>>& angles) {
    size_t n = angles.size();
    size_t simd_count = (n / 4) * 4;
    
    const __m256d two_pi = _mm256_set1_pd(2.0 * M_PI);
    const __m256d zero = _mm256_setzero_pd();
    
    for (size_t i = 0; i < simd_count; i += 4) {
        __m256d angle_vec = _mm256_load_pd(&angles[i]);
        
        // angle = fmod(angle, 2*pi)
        angle_vec = _mm256_sub_pd(angle_vec, _mm256_mul_pd(two_pi, 
            _mm256_floor_pd(_mm256_div_pd(angle_vec, two_pi))));
        
        // if (angle < 0) angle += 2*pi
        __m256d mask = _mm256_cmp_pd(angle_vec, zero, _CMP_LT_OQ);
        angle_vec = _mm256_add_pd(angle_vec, _mm256_and_pd(mask, two_pi));
        
        _mm256_store_pd(&angles[i], angle_vec);
    }
    
    // 处理剩余角度
    for (size_t i = simd_count; i < n; ++i) {
        angles[i] = normalizeAngle(angles[i]);
    }
}

CompactOrbitalElements ConstellationPropagator::getSatelliteElements(size_t satellite_id) const {
    ensureHostElementsUpToDate();

    if (satellite_id >= elements_.size()) {
        throw std::out_of_range("Satellite ID out of range");
    }

    CompactOrbitalElements elem;
    elem.a = elements_.a[satellite_id];
    elem.e = elements_.e[satellite_id];
    elem.i = elements_.i[satellite_id];
    elem.O = elements_.O[satellite_id];
    elem.w = elements_.w[satellite_id];
    elem.M = elements_.M[satellite_id];
    
    return elem;
}

StateVector ConstellationPropagator::getSatelliteState(size_t satellite_id) const {
    CompactOrbitalElements elem = getSatelliteElements(satellite_id);
    return elementsToState(elem);
}

StateVector ConstellationPropagator::elementsToState(const CompactOrbitalElements& elements) const {
    StateVector state;
    
    double a = elements.a, e = elements.e, i = elements.i;
    double O = elements.O, w = elements.w, M = elements.M;
    
    // 计算偏近点角和真近点角
    double E = computeEccentricAnomaly(M, e);
    double nu = computeTrueAnomaly(E, e);
    
    // 计算地心距
    double r = a * (1.0 - e * std::cos(E));
    
    // 轨道平面内的位置矢量
    double x_perifocal = r * std::cos(nu);
    double y_perifocal = r * std::sin(nu);
    
    // 计算转换矩阵
    double cosO = std::cos(O), sinO = std::sin(O);
    double cosi = std::cos(i), sini = std::sin(i);
    double cosw = std::cos(w), sinw = std::sin(w);
    
    Eigen::Matrix3d R;
    R << cosO*cosw - sinO*sinw*cosi, -cosO*sinw - sinO*cosw*cosi, sinO*sini,
         sinO*cosw + cosO*sinw*cosi, -sinO*sinw + cosO*cosw*cosi, -cosO*sini,
         sinw*sini, cosw*sini, cosi;
    
    // 转换到惯性系
    Eigen::Vector3d r_perifocal(x_perifocal, y_perifocal, 0.0);
    state.r = R * r_perifocal;
    
    // 计算速度矢量 (与 J2OrbitPropagator 保持一致的方法)
    double p = a * (1.0 - e * e); // 半通径
    double v_mag_factor = std::sqrt(MU / p);
    Eigen::Vector3d v_perifocal(-v_mag_factor * std::sin(nu), v_mag_factor * (e + std::cos(nu)), 0.0);
    
    state.v = R * v_perifocal;
    
    return state;
}

CompactOrbitalElements ConstellationPropagator::applyImpulseScalar(const CompactOrbitalElements& elements,
                                                                  const Eigen::Vector3d& delta_v, double t) const {
    // 将要素转为状态
    StateVector s = elementsToState(elements);
    // 施加ΔV
    StateVector s_new;
    s_new.r = s.r;
    s_new.v = s.v + delta_v;
    
    // 将状态转回要素（复用J2OrbitPropagator实现更稳妥，但此处按星座类已有流程实现）
    // 这里我们临时构建一个J2OrbitPropagator来复用其stateToElements逻辑
    OrbitalElements oe_full; oe_full.a = elements.a; oe_full.e = elements.e; oe_full.i = elements.i;
    oe_full.O = elements.O; oe_full.w = elements.w; oe_full.M = elements.M; oe_full.t = t;
    J2OrbitPropagator propagator(oe_full);
    OrbitalElements new_full = propagator.stateToElements(s_new, t);
    
    CompactOrbitalElements out;
    out.a = new_full.a; out.e = new_full.e; out.i = new_full.i;
    out.O = new_full.O; out.w = new_full.w; out.M = new_full.M;
    return out;
}

void ConstellationPropagator::applyImpulseToConstellation(const std::vector<Eigen::Vector3d>& delta_vs, double t) {
    ensureHostElementsUpToDate();

    size_t n = elements_.size();
    if (delta_vs.size() != n) {
        throw std::invalid_argument("delta_vs size must match satellite count");
    }

    switch (compute_mode_) {
        case CPU_SCALAR: {
            for (size_t idx = 0; idx < n; ++idx) {
                CompactOrbitalElements elem{elements_.a[idx], elements_.e[idx], elements_.i[idx],
                                            elements_.O[idx], elements_.w[idx], elements_.M[idx]};
                CompactOrbitalElements updated = applyImpulseScalar(elem, delta_vs[idx], t);
                elements_.a[idx] = updated.a;
                elements_.e[idx] = updated.e;
                elements_.i[idx] = updated.i;
                elements_.O[idx] = updated.O;
                elements_.w[idx] = updated.w;
                elements_.M[idx] = updated.M;
            }
            break;
        }
        case CPU_SIMD: {
            applyImpulseSIMD(delta_vs, t);
            break;
        }
        case GPU_CUDA: {
            if (!isCudaAvailable()) {
                std::cerr << "CUDA not available, falling back to SIMD" << std::endl;
                applyImpulseSIMD(delta_vs, t);
                break;
            }
#if defined(HAVE_CUDA_TOOLKIT) && HAVE_CUDA_TOOLKIT
            // 构建SoA的r、v、dv数组
            std::vector<double> rxyz(3 * n), vxyz(3 * n), dvxyz(3 * n);
            for (size_t idx = 0; idx < n; ++idx) {
                CompactOrbitalElements elem{elements_.a[idx], elements_.e[idx], elements_.i[idx],
                                            elements_.O[idx], elements_.w[idx], elements_.M[idx]};
                StateVector state = elementsToState(elem);
                rxyz[idx] = state.r.x();
                rxyz[idx + n] = state.r.y();
                rxyz[idx + 2 * n] = state.r.z();
                vxyz[idx] = state.v.x();
                vxyz[idx + n] = state.v.y();
                vxyz[idx + 2 * n] = state.v.z();
                dvxyz[idx] = delta_vs[idx].x();
                dvxyz[idx + n] = delta_vs[idx].y();
                dvxyz[idx + 2 * n] = delta_vs[idx].z();
            }
            
            // 调用CUDA接口施加脉冲（更新vxyz）
            cuda_apply_impulse(rxyz.data(), vxyz.data(), dvxyz.data(), n);
            
            // 将新的状态向量（r不变，v已更新）转换回轨道要素并写回
            for (size_t idx = 0; idx < n; ++idx) {
                StateVector new_state;
                new_state.r = Eigen::Vector3d(rxyz[idx], rxyz[idx + n], rxyz[idx + 2 * n]);
                new_state.v = Eigen::Vector3d(vxyz[idx], vxyz[idx + n], vxyz[idx + 2 * n]);
                // 复用单星J2转换逻辑
                J2OrbitPropagator temp(OrbitalElements{elements_.a[idx], elements_.e[idx], elements_.i[idx],
                                                       elements_.O[idx], elements_.w[idx], elements_.M[idx], t});
                OrbitalElements updated_full = temp.stateToElements(new_state, t);
                elements_.a[idx] = updated_full.a;
                elements_.e[idx] = updated_full.e;
                elements_.i[idx] = updated_full.i;
                elements_.O[idx] = updated_full.O;
                elements_.w[idx] = updated_full.w;
                elements_.M[idx] = updated_full.M;
            }
#else
            // 理论上不会到这里，但为了安全，回退
            applyImpulseSIMD(delta_vs, t);
#endif
            break;
        }
    }
}

void ConstellationPropagator::applyImpulseToSatellites(const std::vector<size_t>& satellite_ids,
                                                       const std::vector<Eigen::Vector3d>& delta_vs,
                                                       double t) {
    ensureHostElementsUpToDate();

    if (satellite_ids.size() != delta_vs.size()) {
        throw std::invalid_argument("satellite_ids and delta_vs must have same length");
    }
    const size_t n_total = elements_.size();
    const size_t m = satellite_ids.size();
    for (size_t k = 0; k < m; ++k) {
        if (satellite_ids[k] >= n_total) {
            throw std::out_of_range("satellite id out of range");
        }
    }

    switch (compute_mode_) {
        case CPU_SCALAR: {
            for (size_t k = 0; k < m; ++k) {
                size_t idx = satellite_ids[k];
                CompactOrbitalElements elem{elements_.a[idx], elements_.e[idx], elements_.i[idx],
                                            elements_.O[idx], elements_.w[idx], elements_.M[idx]};
                CompactOrbitalElements updated = applyImpulseScalar(elem, delta_vs[k], t);
                elements_.a[idx] = updated.a;
                elements_.e[idx] = updated.e;
                elements_.i[idx] = updated.i;
                elements_.O[idx] = updated.O;
                elements_.w[idx] = updated.w;
                elements_.M[idx] = updated.M;
            }
            break;
        }
        case CPU_SIMD: {
            applyImpulseSubsetSIMD(satellite_ids, delta_vs, t);
            break;
        }
        case GPU_CUDA: {
            if (!isCudaAvailable()) {
                std::cerr << "CUDA not available, falling back to SIMD" << std::endl;
                applyImpulseSubsetSIMD(satellite_ids, delta_vs, t);
                break;
            }
#if defined(HAVE_CUDA_TOOLKIT) && HAVE_CUDA_TOOLKIT
            // 仅为子集构建紧凑 SoA r/v/dv
            std::vector<double> rxyz(3 * m), vxyz(3 * m), dvxyz(3 * m);
            for (size_t k = 0; k < m; ++k) {
                size_t idx = satellite_ids[k];
                CompactOrbitalElements elem{elements_.a[idx], elements_.e[idx], elements_.i[idx],
                                            elements_.O[idx], elements_.w[idx], elements_.M[idx]};
                StateVector state = elementsToState(elem);
                rxyz[k] = state.r.x();
                rxyz[k + m] = state.r.y();
                rxyz[k + 2 * m] = state.r.z();
                vxyz[k] = state.v.x();
                vxyz[k + m] = state.v.y();
                vxyz[k + 2 * m] = state.v.z();
                dvxyz[k] = delta_vs[k].x();
                dvxyz[k + m] = delta_vs[k].y();
                dvxyz[k + 2 * m] = delta_vs[k].z();
            }
            // 调用CUDA接口对紧凑数组施加脉冲
            cuda_apply_impulse(rxyz.data(), vxyz.data(), dvxyz.data(), m);
            // 将结果映射回相应卫星并回写要素
            for (size_t k = 0; k < m; ++k) {
                size_t idx = satellite_ids[k];
                StateVector new_state;
                new_state.r = Eigen::Vector3d(rxyz[k], rxyz[k + m], rxyz[k + 2 * m]);
                new_state.v = Eigen::Vector3d(vxyz[k], vxyz[k + m], vxyz[k + 2 * m]);
                J2OrbitPropagator temp(OrbitalElements{elements_.a[idx], elements_.e[idx], elements_.i[idx],
                                                       elements_.O[idx], elements_.w[idx], elements_.M[idx], t});
                OrbitalElements updated_full = temp.stateToElements(new_state, t);
                elements_.a[idx] = updated_full.a;
                elements_.e[idx] = updated_full.e;
                elements_.i[idx] = updated_full.i;
                elements_.O[idx] = updated_full.O;
                elements_.w[idx] = updated_full.w;
                elements_.M[idx] = updated_full.M;
            }
#else
            applyImpulseSubsetSIMD(satellite_ids, delta_vs, t);
#endif
            break;
        }
    }

    markDeviceElementsDirty();
}

void ConstellationPropagator::applyImpulseSubsetSIMD(const std::vector<size_t>& satellite_ids,
                                                     const std::vector<Eigen::Vector3d>& delta_vs,
                                                     double t) {
    ensureHostElementsUpToDate();

    const size_t m = satellite_ids.size();
    
#if defined(__AVX2__) || defined(__AVX__)
    // AVX2优化：收集相邻索引的卫星数据进行批处理
    const size_t simd_width = 4; // AVX2并行处理4个double
    size_t simd_count = m / simd_width;
    
    // 批量处理：每次处理4颗卫星
    for (size_t batch = 0; batch < simd_count; ++batch) {
        size_t start = batch * simd_width;
        
        // 收集4颗卫星的轨道要素到连续缓冲区
        alignas(32) double a_vals[4], e_vals[4], i_vals[4];
        alignas(32) double O_vals[4], w_vals[4], M_vals[4];
        alignas(32) double dvx_vals[4], dvy_vals[4], dvz_vals[4];
        size_t indices[4];
        
        for (size_t k = 0; k < simd_width; ++k) {
            size_t idx = satellite_ids[start + k];
            indices[k] = idx;
            a_vals[k] = elements_.a[idx];
            e_vals[k] = elements_.e[idx];
            i_vals[k] = elements_.i[idx];
            O_vals[k] = elements_.O[idx];
            w_vals[k] = elements_.w[idx];
            M_vals[k] = elements_.M[idx];
            dvx_vals[k] = delta_vs[start + k].x();
            dvy_vals[k] = delta_vs[start + k].y();
            dvz_vals[k] = delta_vs[start + k].z();
        }
        
        // 对收集的数据进行批量处理（目前仍为优化的标量循环）
        // 未来可实现真正的AVX2向量化运算
        for (size_t k = 0; k < simd_width; ++k) {
            CompactOrbitalElements elem{a_vals[k], e_vals[k], i_vals[k], O_vals[k], w_vals[k], M_vals[k]};
            Eigen::Vector3d dv(dvx_vals[k], dvy_vals[k], dvz_vals[k]);
            CompactOrbitalElements updated = applyImpulseScalar(elem, dv, t);
            
            // 写回到原始索引位置
            size_t idx = indices[k];
            elements_.a[idx] = updated.a;
            elements_.e[idx] = updated.e;
            elements_.i[idx] = updated.i;
            elements_.O[idx] = updated.O;
            elements_.w[idx] = updated.w;
            elements_.M[idx] = updated.M;
        }
    }
    
    // 处理剩余的卫星（不足4颗）
    for (size_t k = simd_count * simd_width; k < m; ++k) {
        size_t idx = satellite_ids[k];
        CompactOrbitalElements elem{elements_.a[idx], elements_.e[idx], elements_.i[idx],
                                    elements_.O[idx], elements_.w[idx], elements_.M[idx]};
        CompactOrbitalElements updated = applyImpulseScalar(elem, delta_vs[k], t);
        elements_.a[idx] = updated.a;
        elements_.e[idx] = updated.e;
        elements_.i[idx] = updated.i;
        elements_.O[idx] = updated.O;
        elements_.w[idx] = updated.w;
        elements_.M[idx] = updated.M;
    }
#else
    // 回退到标量循环
    for (size_t k = 0; k < m; ++k) {
        size_t idx = satellite_ids[k];
        CompactOrbitalElements elem{elements_.a[idx], elements_.e[idx], elements_.i[idx],
                                    elements_.O[idx], elements_.w[idx], elements_.M[idx]};
        CompactOrbitalElements updated = applyImpulseScalar(elem, delta_vs[k], t);
        elements_.a[idx] = updated.a;
        elements_.e[idx] = updated.e;
        elements_.i[idx] = updated.i;
        elements_.O[idx] = updated.O;
        elements_.w[idx] = updated.w;
        elements_.M[idx] = updated.M;
    }
#endif

    markDeviceElementsDirty();
}

Eigen::MatrixXd ConstellationPropagator::getAllPositions() const {
    size_t n = elements_.size();
    Eigen::MatrixXd positions(3, n);
    
    for (size_t i = 0; i < n; ++i) {
        StateVector state = getSatelliteState(i);
        positions.col(i) = state.r;
    }
    
    return positions;
}

double ConstellationPropagator::computeEccentricAnomaly(double M, double e) const {
    M = normalizeAngle(M);
    double E = (e < 0.8) ? M : (M > M_PI ? M - e : M + e);
    
    for (int iter = 0; iter < 20; ++iter) {
        double delta = (E - e * std::sin(E) - M) / (1.0 - e * std::cos(E));
        E -= delta;
        if (std::abs(delta) < EPSILON) break;
    }
    
    return E;
}

double ConstellationPropagator::computeTrueAnomaly(double E, double e) const {
    double tan_nu_2 = std::sqrt((1.0 + e) / (1.0 - e)) * std::tan(E / 2.0);
    return normalizeAngle(2.0 * std::atan(tan_nu_2));
}

double ConstellationPropagator::normalizeAngle(double angle) const {
    angle = std::fmod(angle, 2.0 * M_PI);
    if (angle < 0) angle += 2.0 * M_PI;
    return angle;
}

bool ConstellationPropagator::isCudaAvailable() noexcept {
#if defined(HAVE_CUDA_TOOLKIT) && HAVE_CUDA_TOOLKIT
    // 缓存检测结果，避免每帧重复调用带来的开销
    static const bool available = []() noexcept {
        int device_count = 0;
        cudaError_t err = cudaGetDeviceCount(&device_count);
        return (err == cudaSuccess && device_count > 0);
    }();
    return available;
#else
    // 未启用CUDA工具链时，避免链接到cudart，直接返回不可用
    return false;
#endif
}

void ConstellationPropagator::propagateCUDA(double dt, size_t iterations) {
#if defined(HAVE_CUDA_TOOLKIT) && HAVE_CUDA_TOOLKIT
    size_t n = elements_.size();
    if (n == 0 || iterations == 0) return;

    initializeCUDA();

    ensureDeviceElementsUpToDate();

    cudaStream_t stream = cuda_stream_ ? cuda_stream_ : 0;

    const size_t max_iterations = static_cast<size_t>(std::numeric_limits<int>::max());
    size_t remaining = iterations;
    while (remaining > 0) {
        size_t batch = std::min(remaining, max_iterations);
        cuda_propagate_j2_persistent(d_a_, d_e_, d_i_, d_O_, d_w_, d_M_,
                                     n, dt, MU, RE, J2, batch, stream);
        cudaStreamSynchronize(stream);
        remaining -= batch;
        // 更新常量后下一批次仍可复用同一缓冲区
    }

    markHostElementsDirty();
#else
    // 回退到CPU实现
    std::cerr << "CUDA not available, falling back to SIMD" << std::endl;
    for (size_t iter = 0; iter < iterations; ++iter) {
        propagateSIMD(dt);
    }
#endif
}

void ConstellationPropagator::initializeCUDA() {
#if defined(HAVE_CUDA_TOOLKIT) && HAVE_CUDA_TOOLKIT
    size_t n = elements_.size();
    if (n == 0) {
        return;
    }

    if (gpu_buffer_size_ < n) {
        // 清理旧缓冲区
        cleanupCUDA();

        // 分配新的持久化缓冲区
        size_t size = n * sizeof(double);
        cudaMalloc(&d_a_, size);
        cudaMalloc(&d_e_, size);
        cudaMalloc(&d_i_, size);
        cudaMalloc(&d_O_, size);
        cudaMalloc(&d_w_, size);
        cudaMalloc(&d_M_, size);
        cudaMalloc(&d_x_, size);
        cudaMalloc(&d_y_, size);
        cudaMalloc(&d_z_, size);

        gpu_buffer_size_ = n;
        device_elements_dirty_ = true;
    }

    if (!cuda_stream_) {
        cudaStreamCreate(&cuda_stream_);
    }
#endif
}

void ConstellationPropagator::cleanupCUDA() {
#if defined(HAVE_CUDA_TOOLKIT) && HAVE_CUDA_TOOLKIT
    if (gpu_buffer_size_ > 0) {
        cudaFree(d_a_);
        cudaFree(d_e_);
        cudaFree(d_i_);
        cudaFree(d_O_);
        cudaFree(d_w_);
        cudaFree(d_M_);
        cudaFree(d_x_);
        cudaFree(d_y_);
        cudaFree(d_z_);
        
        if (cuda_stream_) {
            cudaStreamDestroy(cuda_stream_);
            cuda_stream_ = 0;
        }
        
        d_a_ = d_e_ = d_i_ = d_O_ = d_w_ = d_M_ = nullptr;
        d_x_ = d_y_ = d_z_ = nullptr;
        gpu_buffer_size_ = 0;
        device_elements_dirty_ = true;
        host_elements_dirty_ = false;
    }
#endif
}

void ConstellationPropagator::ensureDeviceElementsUpToDate() {
#if defined(HAVE_CUDA_TOOLKIT) && HAVE_CUDA_TOOLKIT
    if (!device_elements_dirty_) {
        return;
    }

    size_t n = elements_.size();
    if (n == 0 || gpu_buffer_size_ < n) {
        return;
    }

    size_t bytes = n * sizeof(double);
    cudaStream_t stream = cuda_stream_ ? cuda_stream_ : 0;

    cudaMemcpyAsync(d_a_, elements_.a.data(), bytes, cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_e_, elements_.e.data(), bytes, cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_i_, elements_.i.data(), bytes, cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_O_, elements_.O.data(), bytes, cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_w_, elements_.w.data(), bytes, cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_M_, elements_.M.data(), bytes, cudaMemcpyHostToDevice, stream);

    device_elements_dirty_ = false;
#endif
}

void ConstellationPropagator::ensureHostElementsUpToDate() const {
#if defined(HAVE_CUDA_TOOLKIT) && HAVE_CUDA_TOOLKIT
    if (!host_elements_dirty_) {
        return;
    }

    size_t n = elements_.size();
    if (n == 0) {
        host_elements_dirty_ = false;
        return;
    }

    size_t bytes = n * sizeof(double);
    cudaStream_t stream = cuda_stream_ ? cuda_stream_ : 0;

    cudaMemcpyAsync(elements_.O.data(), d_O_, bytes, cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(elements_.w.data(), d_w_, bytes, cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(elements_.M.data(), d_M_, bytes, cudaMemcpyDeviceToHost, stream);

    cudaStreamSynchronize(stream);

    host_elements_dirty_ = false;
#endif
}

void ConstellationPropagator::markDeviceElementsDirty() {
#if defined(HAVE_CUDA_TOOLKIT) && HAVE_CUDA_TOOLKIT
    device_elements_dirty_ = true;
    host_elements_dirty_ = false;
#endif
}

void ConstellationPropagator::markHostElementsDirty() {
#if defined(HAVE_CUDA_TOOLKIT) && HAVE_CUDA_TOOLKIT
    host_elements_dirty_ = true;
    device_elements_dirty_ = false;
#endif
}

void ConstellationPropagator::applyImpulseSIMD(const std::vector<Eigen::Vector3d>& delta_vs, double t) {
    ensureHostElementsUpToDate();

    size_t n = elements_.size();
    if (delta_vs.size() != n) {
        throw std::invalid_argument("delta_vs size must match satellite count");
    }

#if defined(__AVX2__) || defined(__AVX__)
    // 启用AVX2批处理模式
    const size_t simd_width = 4; // AVX2可并行处理4个double
    size_t simd_count = n / simd_width;
    size_t remaining = n % simd_width;
    
    // 批量处理：每次处理4颗卫星
    for (size_t batch = 0; batch < simd_count; ++batch) {
        size_t start_idx = batch * simd_width;
        
        // 为每个轨道要素分量准备AVX2打包数据
        alignas(32) double a_vals[4], e_vals[4], i_vals[4];
        alignas(32) double O_vals[4], w_vals[4], M_vals[4];
        alignas(32) double dvx_vals[4], dvy_vals[4], dvz_vals[4];
        
        // 加载4颗卫星的轨道要素
        for (size_t k = 0; k < simd_width; ++k) {
            size_t idx = start_idx + k;
            a_vals[k] = elements_.a[idx];
            e_vals[k] = elements_.e[idx];
            i_vals[k] = elements_.i[idx];
            O_vals[k] = elements_.O[idx];
            w_vals[k] = elements_.w[idx];
            M_vals[k] = elements_.M[idx];
            dvx_vals[k] = delta_vs[idx].x();
            dvy_vals[k] = delta_vs[idx].y();
            dvz_vals[k] = delta_vs[idx].z();
        }
        
        // 对于AVX2实现，我们将要素→状态→脉冲→状态→要素的流程进行向量化
        // 当前AVX2实现的复杂度较高，暂时退化为优化的标量循环
        for (size_t k = 0; k < simd_width; ++k) {
            size_t idx = start_idx + k;
            CompactOrbitalElements elem{a_vals[k], e_vals[k], i_vals[k], O_vals[k], w_vals[k], M_vals[k]};
            CompactOrbitalElements updated = applyImpulseScalar(elem, delta_vs[idx], t);
            elements_.a[idx] = updated.a;
            elements_.e[idx] = updated.e;
            elements_.i[idx] = updated.i;
            elements_.O[idx] = updated.O;
            elements_.w[idx] = updated.w;
            elements_.M[idx] = updated.M;
        }
    }
    
    // 处理剩余的卫星（不足4颗）
    for (size_t idx = simd_count * simd_width; idx < n; ++idx) {
        CompactOrbitalElements elem{elements_.a[idx], elements_.e[idx], elements_.i[idx],
                                    elements_.O[idx], elements_.w[idx], elements_.M[idx]};
        CompactOrbitalElements updated = applyImpulseScalar(elem, delta_vs[idx], t);
        elements_.a[idx] = updated.a;
        elements_.e[idx] = updated.e;
        elements_.i[idx] = updated.i;
        elements_.O[idx] = updated.O;
        elements_.w[idx] = updated.w;
        elements_.M[idx] = updated.M;
    }
#else
    // 回退到标量实现
    for (size_t idx = 0; idx < n; ++idx) {
        CompactOrbitalElements elem{elements_.a[idx], elements_.e[idx], elements_.i[idx],
                                    elements_.O[idx], elements_.w[idx], elements_.M[idx]};
        CompactOrbitalElements updated = applyImpulseScalar(elem, delta_vs[idx], t);
        elements_.a[idx] = updated.a;
        elements_.e[idx] = updated.e;
        elements_.i[idx] = updated.i;
        elements_.O[idx] = updated.O;
        elements_.w[idx] = updated.w;
        elements_.M[idx] = updated.M;
    }
#endif

    markDeviceElementsDirty();
}
