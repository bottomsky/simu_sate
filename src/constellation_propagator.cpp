#include "constellation_propagator.h"
#include <algorithm>
#include <cstring>

ConstellationPropagator::ConstellationPropagator(double epoch_time)
    : epoch_time_(epoch_time), current_time_(epoch_time), step_size_(60.0), 
      compute_mode_(CPU_SIMD) {
#ifdef __CUDACC__
    d_elements_data_ = nullptr;
    gpu_buffer_size_ = 0;
#endif
}

ConstellationPropagator::~ConstellationPropagator() {
#ifdef __CUDACC__
    cleanupCUDA();
#endif
}

void ConstellationPropagator::addSatellites(const std::vector<CompactOrbitalElements>& satellites) {
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
}

void ConstellationPropagator::addSatellite(const CompactOrbitalElements& satellite) {
    size_t idx = elements_.size();
    elements_.resize(idx + 1);
    
    elements_.a[idx] = satellite.a;
    elements_.e[idx] = satellite.e;
    elements_.i[idx] = satellite.i;
    elements_.O[idx] = satellite.O;
    elements_.w[idx] = satellite.w;
    elements_.M[idx] = satellite.M;
}

void ConstellationPropagator::propagateConstellation(double target_time) {
    double dt_total = target_time - current_time_;
    
    if (dt_total < EPSILON) {
        return;
    }
    
    // 分步积分
    double remaining_time = dt_total;
    if (!adaptive_step_size_) {
        while (remaining_time > EPSILON) {
            double dt = std::min(remaining_time, step_size_);
            
            switch (compute_mode_) {
                case CPU_SCALAR:
                    propagateScalar(dt);
                    break;
                case CPU_SIMD:
                    propagateSIMD(dt);
                    break;
                case GPU_CUDA:
#ifdef __CUDACC__
                    propagateCUDA(dt);
#else
                    std::cerr << "CUDA not available, falling back to SIMD" << std::endl;
                    propagateSIMD(dt);
#endif
                    break;
            }
            
            remaining_time -= dt;
        }
    } else {
        // 自适应步长：对全体卫星估计局部误差，选取全体可接受的步长
        double dt = std::min(remaining_time, step_size_);
        dt = std::max(min_step_size_, std::min(dt, max_step_size_));
        while (remaining_time > EPSILON) {
            if (dt > remaining_time) dt = remaining_time;
            // 以采样的方式评估误差（若星座很大，抽样评估）
            double max_err = 0.0;
            size_t n = elements_.size();
            size_t sample = std::max<size_t>(1, std::min<size_t>(n, 16));
            size_t stride = std::max<size_t>(1, n / sample);
            for (size_t i = 0; i < n; i += stride) {
                CompactOrbitalElements elem{elements_.a[i], elements_.e[i], elements_.i[i], elements_.O[i], elements_.w[i], elements_.M[i]};
                double err = estimateLocalErrorScalar(elem, dt);
                if (err > max_err) max_err = err;
            }
            if (max_err <= tolerance_ || dt <= min_step_size_ + 1e-12) {
                // 接受步长
                switch (compute_mode_) {
                    case CPU_SCALAR: propagateScalar(dt); break;
                    case CPU_SIMD:   propagateSIMD(dt);   break;
                    case GPU_CUDA:
#ifdef __CUDACC__
                        propagateCUDA(dt);
#else
                        std::cerr << "CUDA not available, falling back to SIMD" << std::endl;
                        propagateSIMD(dt);
#endif
                        break;
                }
                remaining_time -= dt;
                // 放宽步长
                double safety = 0.9, growth = 1.5;
                dt = std::min(max_step_size_, std::min(remaining_time, std::max(min_step_size_, dt * safety * std::pow(std::max(max_err, 1e-16), -0.2))));
                dt = std::min(dt, step_size_ * growth);
            } else {
                // 缩小步长重试
                double safety = 0.9;
                dt = std::max(min_step_size_, dt * safety * std::pow(std::max(max_err, 1e-16), -0.25));
            }
        }
    }
    
    current_time_ = target_time;
}

void ConstellationPropagator::propagateScalar(double dt) {
    size_t n = elements_.size();
    
    for (size_t i = 0; i < n; ++i) {
        // 提取单个卫星的轨道要素
        CompactOrbitalElements elem = getSatelliteElements(i);
        
        // 使用RK4积分器替代简单欧拉积分，提高精度
        auto computeDerivatives = [&](const CompactOrbitalElements& e) -> std::array<double, 3> {
            double a = e.a, ec = e.e, inc = e.i;
            if (std::abs(1.0 - ec * ec) < EPSILON) {
                return {0.0, 0.0, 0.0}; // 避免奇异性
            }
            
            double n = std::sqrt(MU / (a * a * a));
            double p = a * (1.0 - ec * ec);
            double factor = (3.0 / 2.0) * J2 * n * (RE / p) * (RE / p);
            double cos_i = std::cos(inc);
            double sin_i_sq = std::sin(inc) * std::sin(inc);
            
            double dO_dt = -factor * cos_i;
            double dw_dt = factor * (2.0 - 2.5 * sin_i_sq);
            double dM_dt = n - factor * std::sqrt(1.0 - ec * ec) * (1.5 * sin_i_sq - 0.5);
            
            return {dO_dt, dw_dt, dM_dt};
        };
        
        // RK4积分：k1在起点
        auto k1 = computeDerivatives(elem);
        
        // k2在中点，使用k1预测
        CompactOrbitalElements temp = elem;
        temp.O = normalizeAngle(temp.O + k1[0] * dt / 2.0);
        temp.w = normalizeAngle(temp.w + k1[1] * dt / 2.0);
        temp.M = normalizeAngle(temp.M + k1[2] * dt / 2.0);
        auto k2 = computeDerivatives(temp);
        
        // k3在中点，使用k2预测
        temp = elem;
        temp.O = normalizeAngle(temp.O + k2[0] * dt / 2.0);
        temp.w = normalizeAngle(temp.w + k2[1] * dt / 2.0);
        temp.M = normalizeAngle(temp.M + k2[2] * dt / 2.0);
        auto k3 = computeDerivatives(temp);
        
        // k4在终点，使用k3预测
        temp = elem;
        temp.O = normalizeAngle(temp.O + k3[0] * dt);
        temp.w = normalizeAngle(temp.w + k3[1] * dt);
        temp.M = normalizeAngle(temp.M + k3[2] * dt);
        auto k4 = computeDerivatives(temp);
        
        // 使用RK4加权平均更新轨道要素
        elements_.O[i] = normalizeAngle(elem.O + (k1[0] + 2.0*k2[0] + 2.0*k3[0] + k4[0]) * dt / 6.0);
        elements_.w[i] = normalizeAngle(elem.w + (k1[1] + 2.0*k2[1] + 2.0*k3[1] + k4[1]) * dt / 6.0);
        elements_.M[i] = normalizeAngle(elem.M + (k1[2] + 2.0*k2[2] + 2.0*k3[2] + k4[2]) * dt / 6.0);
    }
}

void ConstellationPropagator::propagateSIMD(double dt) {
    size_t n = elements_.size();
    size_t simd_count = (n / 4) * 4;  // AVX2处理4个double
    
    // SIMD常数
    const __m256d mu_vec = _mm256_set1_pd(MU);
    const __m256d re_vec = _mm256_set1_pd(RE);
    const __m256d j2_vec = _mm256_set1_pd(J2);
    const __m256d dt_vec = _mm256_set1_pd(dt);
    const __m256d dt_half_vec = _mm256_set1_pd(dt / 2.0);
    const __m256d three = _mm256_set1_pd(3.0);
    const __m256d two = _mm256_set1_pd(2.0);
    const __m256d one_point_five = _mm256_set1_pd(1.5);
    const __m256d two_point_five = _mm256_set1_pd(2.5);
    const __m256d half = _mm256_set1_pd(0.5);
    const __m256d six = _mm256_set1_pd(6.0);
    const __m256d one = _mm256_set1_pd(1.0);
    const __m256d epsilon = _mm256_set1_pd(EPSILON);
    
    // SIMD RK4积分器
    auto computeDerivativesSIMD = [&](__m256d a_vec, __m256d e_vec, __m256d i_vec) -> std::array<__m256d, 3> {
        // 计算平均角速度 n = sqrt(MU / a^3)
        __m256d a3 = _mm256_mul_pd(_mm256_mul_pd(a_vec, a_vec), a_vec);
        __m256d n_vec = _mm256_sqrt_pd(_mm256_div_pd(mu_vec, a3));
        
        // 避免奇异性检查（简化）
        __m256d e2 = _mm256_mul_pd(e_vec, e_vec);
        __m256d one_minus_e2 = _mm256_sub_pd(one, e2);
        
        // 计算J2摄动参数，与标量实现一致：factor = (3/2) * J2 * n * (RE/p)^2
        __m256d p = _mm256_mul_pd(a_vec, one_minus_e2);
        __m256d re_over_p = _mm256_div_pd(re_vec, p);
        __m256d re_over_p_sq = _mm256_mul_pd(re_over_p, re_over_p);
        __m256d factor_norm = _mm256_mul_pd(_mm256_mul_pd(_mm256_mul_pd(one_point_five, j2_vec), n_vec), re_over_p_sq);
        
        // 计算三角函数
        __m256d cos_i, sin_i;
        // SIMD 三角函数计算（使用标准库）
        alignas(32) double i_vals[4];
        _mm256_store_pd(i_vals, i_vec);
        cos_i = _mm256_set_pd(std::cos(i_vals[3]), std::cos(i_vals[2]), std::cos(i_vals[1]), std::cos(i_vals[0]));
        sin_i = _mm256_set_pd(std::sin(i_vals[3]), std::sin(i_vals[2]), std::sin(i_vals[1]), std::sin(i_vals[0]));
        
        __m256d sin2_i = _mm256_mul_pd(sin_i, sin_i);
        
        // 计算导数
        __m256d dO_dt = _mm256_mul_pd(_mm256_sub_pd(_mm256_setzero_pd(), factor_norm), cos_i);
        __m256d dw_dt = _mm256_mul_pd(factor_norm, _mm256_sub_pd(two, _mm256_mul_pd(two_point_five, sin2_i)));
        __m256d sqrt_one_minus_e2 = _mm256_sqrt_pd(one_minus_e2);
        __m256d dM_term = _mm256_mul_pd(factor_norm, _mm256_mul_pd(sqrt_one_minus_e2, 
                                       _mm256_sub_pd(_mm256_mul_pd(one_point_five, sin2_i), half)));
        __m256d dM_dt = _mm256_sub_pd(n_vec, dM_term);
        
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
        // 注意：对于J2摄动，只有i会影响导数，而i在短时间内变化很小，所以简化处理
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
    
    // 处理剩余的卫星 (标量RK4方式)
    for (size_t i = simd_count; i < n; ++i) {
        CompactOrbitalElements elem = getSatelliteElements(i);
        
        // 与propagateScalar相同的RK4实现
        auto computeDerivatives = [&](const CompactOrbitalElements& e) -> std::array<double, 3> {
            double a = e.a, ec = e.e, inc = e.i;
            if (std::abs(1.0 - ec * ec) < EPSILON) {
                return {0.0, 0.0, 0.0}; // 避免奇异性
            }
            
            double mean_motion = std::sqrt(MU / (a * a * a));
            double p = a * (1.0 - ec * ec);
            double factor = (3.0 / 2.0) * J2 * mean_motion * (RE / p) * (RE / p);
            double cos_i = std::cos(inc);
            double sin_i_sq = std::sin(inc) * std::sin(inc);
            
            double dO_dt = -factor * cos_i;
            double dw_dt = factor * (2.0 - 2.5 * sin_i_sq);
            double dM_dt = mean_motion - factor * std::sqrt(1.0 - ec * ec) * (1.5 * sin_i_sq - 0.5);
            
            return {dO_dt, dw_dt, dM_dt};
        };
        
        // RK4积分：k1在起点
        auto k1 = computeDerivatives(elem);
        
        // k2在中点，使用k1预测
        CompactOrbitalElements temp = elem;
        temp.O = normalizeAngle(temp.O + k1[0] * dt / 2.0);
        temp.w = normalizeAngle(temp.w + k1[1] * dt / 2.0);
        temp.M = normalizeAngle(temp.M + k1[2] * dt / 2.0);
        auto k2 = computeDerivatives(temp);
        
        // k3在中点，使用k2预测
        temp = elem;
        temp.O = normalizeAngle(temp.O + k2[0] * dt / 2.0);
        temp.w = normalizeAngle(temp.w + k2[1] * dt / 2.0);
        temp.M = normalizeAngle(temp.M + k2[2] * dt / 2.0);
        auto k3 = computeDerivatives(temp);
        
        // k4在终点，使用k3预测
        temp = elem;
        temp.O = normalizeAngle(temp.O + k3[0] * dt);
        temp.w = normalizeAngle(temp.w + k3[1] * dt);
        temp.M = normalizeAngle(temp.M + k3[2] * dt);
        auto k4 = computeDerivatives(temp);
        
        // 使用RK4加权平均更新轨道要素
        elements_.O[i] = normalizeAngle(elem.O + (k1[0] + 2.0*k2[0] + 2.0*k3[0] + k4[0]) * dt / 6.0);
        elements_.w[i] = normalizeAngle(elem.w + (k1[1] + 2.0*k2[1] + 2.0*k3[1] + k4[1]) * dt / 6.0);
        elements_.M[i] = normalizeAngle(elem.M + (k1[2] + 2.0*k2[2] + 2.0*k3[2] + k4[2]) * dt / 6.0);
    }
    
    // 批量角度归一化
    normalizeAnglesSIMD(elements_.O);
    normalizeAnglesSIMD(elements_.w);
    normalizeAnglesSIMD(elements_.M);
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

bool ConstellationPropagator::isCudaAvailable() {
#ifdef __CUDACC__
    int device_count;
    return (cudaGetDeviceCount(&device_count) == cudaSuccess && device_count > 0);
#else
    return false;
#endif
}

#ifdef __CUDACC__
void ConstellationPropagator::propagateCUDA(double dt) {
    // CUDA实现留给具体的CUDA文件
    std::cerr << "CUDA implementation not yet complete" << std::endl;
    propagateSIMD(dt);  // 回退到SIMD
}

void ConstellationPropagator::initializeCUDA() {
    // CUDA初始化代码
}

void ConstellationPropagator::cleanupCUDA() {
    if (d_elements_data_) {
        cudaFree(d_elements_data_);
        d_elements_data_ = nullptr;
    }
}
#endif

double ConstellationPropagator::estimateLocalErrorScalar(const CompactOrbitalElements& elem, double dt) {
    // 单步：基于简化模型积分一次
    auto step_once = [&](const CompactOrbitalElements& e, double h) {
        double a = e.a, ec = e.e, inc = e.i;
        double n = std::sqrt(MU / std::pow(a, 3));
        double factor = (3.0 * J2 * MU * RE * RE) / (2.0 * std::pow(a * (1.0 - ec*ec), 2));
        double common = factor / (n * a * a);
        CompactOrbitalElements r = e;
        r.O = normalizeAngle(r.O + (-common * std::cos(inc)) * h);
        r.w = normalizeAngle(r.w + (common * (2.0 - 2.5 * std::pow(std::sin(inc), 2))) * h);
        double sin_i = std::sin(inc);
        double sin_i_sq = sin_i * sin_i;
        double sqrt_one_minus_e2 = std::sqrt(1.0 - ec*ec);
        double j2_term = factor * sqrt_one_minus_e2 * (1.5 * sin_i_sq - 0.5) / (n * a * a);
        r.M = normalizeAngle(r.M + (n - j2_term) * h);
        return r;
    };
    // 单步结果
    CompactOrbitalElements y1 = step_once(elem, dt);
    // 两个半步
    CompactOrbitalElements half = step_once(elem, dt * 0.5);
    CompactOrbitalElements y2 = step_once(half, dt * 0.5);
    auto angle_diff = [&](double a1, double a2){
        double d = std::fmod(std::abs(a1 - a2), 2.0 * M_PI);
        if (d > M_PI) d = 2.0 * M_PI - d;
        return d;
    };
    double eO = angle_diff(y1.O, y2.O);
    double ew = angle_diff(y1.w, y2.w);
    double eM = angle_diff(y1.M, y2.M);
    double ang_norm = (eO + ew + eM) / (3.0 * M_PI);
    double e_scale = 1.0 + 5.0 * elem.e;
    return e_scale * ang_norm;
}

double ConstellationPropagator::estimateLocalErrorSIMD(size_t idx, double dt) {
    CompactOrbitalElements elem{elements_.a[idx], elements_.e[idx], elements_.i[idx], elements_.O[idx], elements_.w[idx], elements_.M[idx]};
    return estimateLocalErrorScalar(elem, dt);
}