#ifndef CONSTELLATION_PROPAGATOR_H
#define CONSTELLATION_PROPAGATOR_H

#include <Eigen/Dense>
#include <vector>
#include <memory>
#include <immintrin.h>  // SIMD intrinsics
#include <cmath>
#include <iostream>
#include "math_constants.h"
#include "common_types.h"

#ifdef __CUDACC__
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#endif

// 压紧的轨道要素结构体 (不包含历元时间)
struct CompactOrbitalElements {
    double a;   // 半长轴 (m)
    double e;   // 偏心率
    double i;   // 倾角 (rad)
    double O;   // 升交点赤经 (rad)
    double w;   // 近地点幅角 (rad)
    double M;   // 平近点角 (rad)
};

// SIMD优化的轨道要素数组 (AoS -> SoA)
struct SIMDOrbitalElements {
    std::vector<double, Eigen::aligned_allocator<double>> a;   // 半长轴数组
    std::vector<double, Eigen::aligned_allocator<double>> e;   // 偏心率数组
    std::vector<double, Eigen::aligned_allocator<double>> i;   // 倾角数组
    std::vector<double, Eigen::aligned_allocator<double>> O;   // 升交点赤经数组
    std::vector<double, Eigen::aligned_allocator<double>> w;   // 近地点幅角数组
    std::vector<double, Eigen::aligned_allocator<double>> M;   // 平近点角数组
    
    void resize(size_t n) {
        a.resize(n); e.resize(n); i.resize(n);
        O.resize(n); w.resize(n); M.resize(n);
    }
    
    size_t size() const { return a.size(); }
};

// 大规模星座外推器
class ConstellationPropagator {
public:
    // 构造函数
    ConstellationPropagator(double epoch_time = 0.0);
    
    // 析构函数
    ~ConstellationPropagator();
    
    // 批量添加卫星
    void addSatellites(const std::vector<CompactOrbitalElements>& satellites);
    
    // 单个添加卫星
    void addSatellite(const CompactOrbitalElements& satellite);
    
    // 批量外推到指定时间
    void propagateConstellation(double target_time);
    
    // 获取指定卫星的当前状态
    StateVector getSatelliteState(size_t satellite_id) const;
    
    // 获取指定卫星的当前轨道要素
    CompactOrbitalElements getSatelliteElements(size_t satellite_id) const;
    
    // 获取所有卫星的位置矢量 (3 x N矩阵)
    Eigen::MatrixXd getAllPositions() const;
    
    // 设置积分步长
    void setStepSize(double step) { step_size_ = step; }
    
    // 启用/禁用自适应步长
    void setAdaptiveStepSize(bool enable) { adaptive_step_size_ = enable; }
    
    // 设置自适应步长参数
    void setAdaptiveParameters(double tolerance = 1e-6, double min_step = 1.0, double max_step = 300.0) {
        tolerance_ = tolerance;
        min_step_size_ = min_step;
        max_step_size_ = max_step;
    }
    
    // 获取卫星数量
    size_t getSatelliteCount() const { return elements_.size(); }
    
    // 设置计算模式
    enum ComputeMode { CPU_SCALAR, CPU_SIMD, GPU_CUDA };
    void setComputeMode(ComputeMode mode) { compute_mode_ = mode; }
    
    // 检查CUDA可用性
    static bool isCudaAvailable();

private:
    SIMDOrbitalElements elements_;      // 星座轨道要素 (SoA格式)
    double epoch_time_;                 // 星座统一历元时间
    double current_time_;               // 当前仿真时间
    double step_size_;                  // 积分步长
    ComputeMode compute_mode_;          // 计算模式
    bool adaptive_step_size_ = false;   // 是否启用自适应步长
    double tolerance_ = 1e-6;           // 误差容忍度
    double min_step_size_ = 1.0;        // 最小步长
    double max_step_size_ = 300.0;      // 最大步长
    
    // CPU标量计算
    void propagateScalar(double dt);
    
    // CPU SIMD计算 (AVX2)
    void propagateSIMD(double dt);
    
    // GPU CUDA计算
    void propagateCUDA(double dt);
    
    // 计算J2摄动导数 (SIMD优化)
    void computeDerivativesSIMD(const SIMDOrbitalElements& elements, 
                               SIMDOrbitalElements& derivatives);
    
    // 四阶龙格-库塔积分 (SIMD优化)
    void rk4IntegrateSIMD(SIMDOrbitalElements& elements, double dt);
    
    // 角度归一化 (SIMD优化)
    void normalizeAnglesSIMD(std::vector<double, Eigen::aligned_allocator<double>>& angles);
    
    // 标量误差估计：用单步dt和两步dt/2比较
    double estimateLocalErrorScalar(const CompactOrbitalElements& elem, double dt);
    
    // SIMD版本的误差估计（针对小批量时退化为标量）
    double estimateLocalErrorSIMD(size_t idx, double dt);
    
    // 单个卫星状态计算
    StateVector elementsToState(const CompactOrbitalElements& elements) const;
    
    // 辅助函数
    double computeEccentricAnomaly(double M, double e) const;
    double computeTrueAnomaly(double E, double e) const;
    double normalizeAngle(double angle) const;
    
#ifdef __CUDACC__
    // CUDA相关成员
    double* d_elements_data_;           // GPU设备内存
    size_t gpu_buffer_size_;
    cublasHandle_t cublas_handle_;
    void initializeCUDA();
    void cleanupCUDA();
#endif
};

// CUDA设备函数声明
#ifdef __CUDACC__
extern "C" {
    void cuda_propagate_j2(double* elements, size_t num_satellites, double dt, 
                          double mu, double re, double j2);
}
#endif

#endif // CONSTELLATION_PROPAGATOR_H