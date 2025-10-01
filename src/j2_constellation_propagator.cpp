#include "j2_constellation_propagator.h"
#include "CoordinateConverter.h"
#include <algorithm>
#include <array>
#include <cstring>
#include <limits>
#include <stdexcept>
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

J2ConstellationPropagator::J2ConstellationPropagator(double epoch_time)
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

J2ConstellationPropagator::J2ConstellationPropagator(const J2ConstellationPropagator& other)
    : J2ConstellationPropagator(other.epoch_time_) {
    copyFrom(other);
}

J2ConstellationPropagator& J2ConstellationPropagator::operator=(const J2ConstellationPropagator& other) {
    if (this == &other) {
        return *this;
    }
    J2ConstellationPropagator tmp(other);
    swap(tmp);
    return *this;
}

J2ConstellationPropagator::J2ConstellationPropagator(J2ConstellationPropagator&& other) noexcept
#if defined(HAVE_CUDA_TOOLKIT) && HAVE_CUDA_TOOLKIT
    : d_a_(nullptr), d_e_(nullptr), d_i_(nullptr), d_O_(nullptr), d_w_(nullptr), d_M_(nullptr),
      d_x_(nullptr), d_y_(nullptr), d_z_(nullptr), gpu_buffer_size_(0), cuda_stream_(nullptr),
      cublas_handle_(nullptr)
#endif
{
    moveFrom(std::move(other));
}

J2ConstellationPropagator& J2ConstellationPropagator::operator=(J2ConstellationPropagator&& other) noexcept {
    if (this == &other) {
        return *this;
    }
    swap(other);
    return *this;
}

J2ConstellationPropagator::~J2ConstellationPropagator() {
#if defined(HAVE_CUDA_TOOLKIT) && HAVE_CUDA_TOOLKIT
    cleanupCUDA();
#endif
}

void J2ConstellationPropagator::copyFrom(const J2ConstellationPropagator& other) {
    if (this == &other) {
        return;
    }

    elements_ = other.elements_;
    epoch_time_ = other.epoch_time_;
    current_time_ = other.current_time_;
    step_size_ = other.step_size_;
    compute_mode_ = other.compute_mode_;
    sample_interval_ = other.sample_interval_;
    steps_per_sample_ = other.steps_per_sample_;
    device_elements_dirty_ = other.device_elements_dirty_;
    host_elements_dirty_ = other.host_elements_dirty_;

#if defined(HAVE_CUDA_TOOLKIT) && HAVE_CUDA_TOOLKIT
    d_a_ = d_e_ = d_i_ = d_O_ = d_w_ = d_M_ = nullptr;
    d_x_ = d_y_ = d_z_ = nullptr;
    gpu_buffer_size_ = 0;
    cuda_stream_ = nullptr;
    cublas_handle_ = nullptr;

    if (other.gpu_buffer_size_ > 0 && isCudaAvailable()) {
        const size_t bytes = other.gpu_buffer_size_ * sizeof(double);

        auto alloc_and_copy = [&](double*& dst, const double* src) {
            cudaMalloc(&dst, bytes);
            if (src) {
                cudaMemcpy(dst, src, bytes, cudaMemcpyDeviceToDevice);
            }
        };

        alloc_and_copy(d_a_, other.d_a_);
        alloc_and_copy(d_e_, other.d_e_);
        alloc_and_copy(d_i_, other.d_i_);
        alloc_and_copy(d_O_, other.d_O_);
        alloc_and_copy(d_w_, other.d_w_);
        alloc_and_copy(d_M_, other.d_M_);
        alloc_and_copy(d_x_, other.d_x_);
        alloc_and_copy(d_y_, other.d_y_);
        alloc_and_copy(d_z_, other.d_z_);

        gpu_buffer_size_ = other.gpu_buffer_size_;

        if (other.cuda_stream_) {
            cudaStreamCreate(&cuda_stream_);
        }
    }
#endif
}

void J2ConstellationPropagator::moveFrom(J2ConstellationPropagator&& other) noexcept {
    elements_ = std::move(other.elements_);
    epoch_time_ = other.epoch_time_;
    current_time_ = other.current_time_;
    step_size_ = other.step_size_;
    compute_mode_ = other.compute_mode_;
    sample_interval_ = other.sample_interval_;
    steps_per_sample_ = other.steps_per_sample_;
    device_elements_dirty_ = other.device_elements_dirty_;
    host_elements_dirty_ = other.host_elements_dirty_;

#if defined(HAVE_CUDA_TOOLKIT) && HAVE_CUDA_TOOLKIT
    d_a_ = std::exchange(other.d_a_, nullptr);
    d_e_ = std::exchange(other.d_e_, nullptr);
    d_i_ = std::exchange(other.d_i_, nullptr);
    d_O_ = std::exchange(other.d_O_, nullptr);
    d_w_ = std::exchange(other.d_w_, nullptr);
    d_M_ = std::exchange(other.d_M_, nullptr);
    d_x_ = std::exchange(other.d_x_, nullptr);
    d_y_ = std::exchange(other.d_y_, nullptr);
    d_z_ = std::exchange(other.d_z_, nullptr);
    gpu_buffer_size_ = std::exchange(other.gpu_buffer_size_, 0);
    cuda_stream_ = std::exchange(other.cuda_stream_, nullptr);
    cublas_handle_ = std::exchange(other.cublas_handle_, nullptr);
#endif
}

void J2ConstellationPropagator::swap(J2ConstellationPropagator& other) noexcept {
    using std::swap;

    swap(elements_, other.elements_);
    swap(epoch_time_, other.epoch_time_);
    swap(current_time_, other.current_time_);
    swap(step_size_, other.step_size_);
    swap(compute_mode_, other.compute_mode_);
    swap(sample_interval_, other.sample_interval_);
    swap(steps_per_sample_, other.steps_per_sample_);
    swap(device_elements_dirty_, other.device_elements_dirty_);
    swap(host_elements_dirty_, other.host_elements_dirty_);

#if defined(HAVE_CUDA_TOOLKIT) && HAVE_CUDA_TOOLKIT
    swap(d_a_, other.d_a_);
    swap(d_e_, other.d_e_);
    swap(d_i_, other.d_i_);
    swap(d_O_, other.d_O_);
    swap(d_w_, other.d_w_);
    swap(d_M_, other.d_M_);
    swap(d_x_, other.d_x_);
    swap(d_y_, other.d_y_);
    swap(d_z_, other.d_z_);
    swap(gpu_buffer_size_, other.gpu_buffer_size_);
    swap(cuda_stream_, other.cuda_stream_);
    swap(cublas_handle_, other.cublas_handle_);
#endif
}

void J2ConstellationPropagator::addSatellites(const std::vector<CompactOrbitalElements>& satellites) {
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

void J2ConstellationPropagator::addSatellite(const CompactOrbitalElements& satellite) {
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

void J2ConstellationPropagator::setStepSize(double step) {
    if (step <= 0.0) {
        throw std::invalid_argument("Step size must be positive");
    }
    step_size_ = step;
    recalcSampleStride();
}

void J2ConstellationPropagator::setSampleInterval(double interval) {
    if (interval <= 0.0) {
        throw std::invalid_argument("Sample interval must be positive");
    }
    sample_interval_ = interval;
    recalcSampleStride();
}

void J2ConstellationPropagator::recalcSampleStride() {
    if (step_size_ <= 0.0) {
        throw std::invalid_argument("Step size must be positive");
    }

    if (sample_interval_ < step_size_) {
        steps_per_sample_ = 1;
        sample_interval_ = step_size_;
        return;
    }

    const double ratio = sample_interval_ / step_size_;
    size_t steps = static_cast<size_t>(std::round(ratio));
    if (steps == 0) {
        steps = 1;
    }
    if (std::abs(static_cast<double>(steps) - ratio) > 1e-8) {
        steps = static_cast<size_t>(std::ceil(ratio));
    }

    steps_per_sample_ = steps;
    sample_interval_ = steps_per_sample_ * step_size_;
}

void J2ConstellationPropagator::propagateSamples(size_t sample_count) {
    if (sample_count == 0) {
        return;
    }

    for (size_t s = 0; s < sample_count; ++s) {
        integrateSteps(steps_per_sample_);
        current_time_ += step_size_ * static_cast<double>(steps_per_sample_);
    }
}

void J2ConstellationPropagator::integrateSteps(size_t steps) {
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

void J2ConstellationPropagator::integrateStepsCustom(size_t steps, double dt) {
    if (steps == 0 || dt <= EPSILON) {
        return;
    }

    switch (compute_mode_) {
        case CPU_SCALAR: {
            for (size_t iter = 0; iter < steps; ++iter) {
                propagateScalar(dt);
            }
            break;
        }
        case CPU_SIMD: {
            for (size_t iter = 0; iter < steps; ++iter) {
                propagateSIMD(dt);
            }
            break;
        }
        case GPU_CUDA: {
            if (isCudaAvailable()) {
                propagateCUDA(dt, steps);
            } else {
                std::cerr << "CUDA not available, falling back to SIMD" << std::endl;
                for (size_t iter = 0; iter < steps; ++iter) {
                    propagateSIMD(dt);
                }
            }
            break;
        }
    }
}

void J2ConstellationPropagator::integrateRemainder(double dt) {
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

void J2ConstellationPropagator::propagateConstellationWithStep(double target_time, double integration_step) {
    if (integration_step <= EPSILON) {
        throw std::invalid_argument("integration_step must be positive");
    }

    double dt_total = target_time - current_time_;
    if (dt_total < EPSILON) {
        return;
    }

    size_t steps = 0;
    double remainder = dt_total;
    double raw_steps = std::floor(dt_total / integration_step);
    if (raw_steps > 0.0) {
        steps = static_cast<size_t>(raw_steps);
        remainder = dt_total - integration_step * raw_steps;
    }
    if (remainder < EPSILON) {
        remainder = 0.0;
    } else if (remainder > integration_step - EPSILON) {
        ++steps;
        remainder = 0.0;
    }

    integrateStepsCustom(steps, integration_step);
    integrateRemainder(remainder);

    current_time_ = target_time;
}

void J2ConstellationPropagator::propagateConstellation(double target_time) {
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

void J2ConstellationPropagator::propagateScalarRange(size_t begin, size_t end, double dt) {
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

void J2ConstellationPropagator::propagateScalar(double dt) {
    ensureHostElementsUpToDate();

    propagateScalarRange(0, elements_.size(), dt);

    markDeviceElementsDirty();
}

void J2ConstellationPropagator::propagateSIMD(double dt) {
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

void J2ConstellationPropagator::normalizeAnglesSIMD(std::vector<double, Eigen::aligned_allocator<double>>& angles) {
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

CompactOrbitalElements J2ConstellationPropagator::getSatelliteElements(size_t satellite_id) const {
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

StateVector J2ConstellationPropagator::getSatelliteState(size_t satellite_id) const {
    CompactOrbitalElements elem = getSatelliteElements(satellite_id);
    return elementsToState(elem);
}

StateVector J2ConstellationPropagator::elementsToState(const CompactOrbitalElements& elements) const {
    return CoordinateConverter::elementsToState(elements);
}


CompactOrbitalElements J2ConstellationPropagator::applyImpulseScalar(const CompactOrbitalElements& elements,
                                                                  const Eigen::Vector3d& delta_v, double t) const {
    // 将要素转为状态
    StateVector s = CoordinateConverter::elementsToState(elements);
    // 施加ΔV
    StateVector s_new;
    s_new.r = s.r;
    s_new.v = s.v + delta_v;
    
    // 将状态转回要素
    return CoordinateConverter::stateToCompactElements(s_new, t);
}

void J2ConstellationPropagator::applyImpulseToConstellation(const std::vector<Eigen::Vector3d>& delta_vs, double t) {
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
                CompactOrbitalElements updated = CoordinateConverter::stateToCompactElements(new_state, t);
                elements_.a[idx] = updated.a;
                elements_.e[idx] = updated.e;
                elements_.i[idx] = updated.i;
                elements_.O[idx] = updated.O;
                elements_.w[idx] = updated.w;
                elements_.M[idx] = updated.M;
            }
#else
            // 理论上不会到这里，但为了安全，回退
            applyImpulseSIMD(delta_vs, t);
#endif
            break;
        }
    }
}

void J2ConstellationPropagator::applyImpulseToSatellites(const std::vector<size_t>& satellite_ids,
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
                CompactOrbitalElements updated = CoordinateConverter::stateToCompactElements(new_state, t);
                elements_.a[idx] = updated.a;
                elements_.e[idx] = updated.e;
                elements_.i[idx] = updated.i;
                elements_.O[idx] = updated.O;
                elements_.w[idx] = updated.w;
                elements_.M[idx] = updated.M;
            }
#else
            applyImpulseSubsetSIMD(satellite_ids, delta_vs, t);
#endif
            break;
        }
    }

    markDeviceElementsDirty();
}

void J2ConstellationPropagator::applyImpulseSubsetSIMD(const std::vector<size_t>& satellite_ids,
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

Eigen::MatrixXd J2ConstellationPropagator::getAllPositions() const {
    size_t n = elements_.size();
    Eigen::MatrixXd positions(3, n);
    
    for (size_t i = 0; i < n; ++i) {
        StateVector state = getSatelliteState(i);
        positions.col(i) = state.r;
    }
    
    return positions;
}







bool J2ConstellationPropagator::isCudaAvailable() noexcept {
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

void J2ConstellationPropagator::propagateCUDA(double dt, size_t iterations) {
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

void J2ConstellationPropagator::initializeCUDA() {
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

void J2ConstellationPropagator::cleanupCUDA() {
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

void J2ConstellationPropagator::ensureDeviceElementsUpToDate() {
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

void J2ConstellationPropagator::ensureHostElementsUpToDate() const {
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

void J2ConstellationPropagator::markDeviceElementsDirty() {
#if defined(HAVE_CUDA_TOOLKIT) && HAVE_CUDA_TOOLKIT
    device_elements_dirty_ = true;
    host_elements_dirty_ = false;
#endif
}

void J2ConstellationPropagator::markHostElementsDirty() {
#if defined(HAVE_CUDA_TOOLKIT) && HAVE_CUDA_TOOLKIT
    host_elements_dirty_ = true;
    device_elements_dirty_ = false;
#endif
}

void J2ConstellationPropagator::applyImpulseSIMD(const std::vector<Eigen::Vector3d>& delta_vs, double t) {
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
