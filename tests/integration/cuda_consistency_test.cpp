#include <gtest/gtest.h>
#include "constellation_propagator.h"
#include <cstdlib>
#include <string>

static double env_or_default(const char* name, double def_val) {
    if (const char* s = std::getenv(name)) {
        try { return std::stod(std::string(s)); } catch (...) { /* ignore parse errors */ }
    }
    return def_val;
}

static const double kPosTol = env_or_default("CUDA_POS_TOL_M", 1.0);
static const double kVelTol = env_or_default("CUDA_VEL_TOL_MPS", 1e-3);

class CUDAConsistencyTest : public ::testing::Test {
protected:
    void SetUp() override {
        // 地球同步轨道参数（单位：米/弧度）
        elements_.a = 42164.169e3; // m
        elements_.e = 0.001;
        elements_.i = 0.017453; // 1度转弧度
        elements_.O = 0.0;
        elements_.w = 0.0;
        elements_.M = 0.0;
    }

    CompactOrbitalElements elements_;
};

// 测试 SCALAR vs CUDA 模式一致性（CUDA 可用时运行，否则跳过）
TEST_F(CUDAConsistencyTest, ScalarVsCUDAConsistency) {
#if HAVE_CUDA_TOOLKIT
    if (!ConstellationPropagator::isCudaAvailable()) {
        GTEST_SKIP() << "[GPU CUDA] CUDA运行时不可用，跳过测试";
    }
#else
    GTEST_SKIP() << "[GPU CUDA] 编译期禁用CUDA，跳过测试";
#endif

    double propagation_time = 3600.0; // 1小时

    // CPU_SCALAR模式
    ConstellationPropagator prop_scalar(0.0);
    prop_scalar.setComputeMode(ConstellationPropagator::CPU_SCALAR);
    prop_scalar.addSatellite(elements_);
    prop_scalar.propagateConstellation(propagation_time);
    StateVector state_scalar = prop_scalar.getSatelliteState(0);

    // GPU_CUDA模式
    ConstellationPropagator prop_cuda(0.0);
    prop_cuda.setComputeMode(ConstellationPropagator::GPU_CUDA);
    prop_cuda.addSatellite(elements_);
    prop_cuda.propagateConstellation(propagation_time);
    StateVector state_cuda = prop_cuda.getSatelliteState(0);

    // 位置与速度一致性
    double pos_diff_x = std::abs(state_scalar.r(0) - state_cuda.r(0));
    double pos_diff_y = std::abs(state_scalar.r(1) - state_cuda.r(1));
    double pos_diff_z = std::abs(state_scalar.r(2) - state_cuda.r(2));

    EXPECT_LT(pos_diff_x, kPosTol) << "X position difference too large (tol=" << kPosTol << "): " << pos_diff_x;
    EXPECT_LT(pos_diff_y, kPosTol) << "Y position difference too large (tol=" << kPosTol << "): " << pos_diff_y;
    EXPECT_LT(pos_diff_z, kPosTol) << "Z position difference too large (tol=" << kPosTol << "): " << pos_diff_z;

    double vel_diff_x = std::abs(state_scalar.v(0) - state_cuda.v(0));
    double vel_diff_y = std::abs(state_scalar.v(1) - state_cuda.v(1));
    double vel_diff_z = std::abs(state_scalar.v(2) - state_cuda.v(2));

    EXPECT_LT(vel_diff_x, kVelTol) << "X velocity difference too large (tol=" << kVelTol << "): " << vel_diff_x;
    EXPECT_LT(vel_diff_y, kVelTol) << "Y velocity difference too large (tol=" << kVelTol << "): " << vel_diff_y;
    EXPECT_LT(vel_diff_z, kVelTol) << "Z velocity difference too large (tol=" << kVelTol << "): " << vel_diff_z;
}

// 测试多卫星场景 SCALAR vs CUDA 一致性（CUDA 可用时运行，否则跳过）
TEST_F(CUDAConsistencyTest, MultiSatelliteConsistency) {
#if HAVE_CUDA_TOOLKIT
    if (!ConstellationPropagator::isCudaAvailable()) {
        GTEST_SKIP() << "[GPU CUDA] CUDA运行时不可用，跳过测试";
    }
#else
    GTEST_SKIP() << "[GPU CUDA] 编译期禁用CUDA，跳过测试";
#endif

    double propagation_time = 1800.0; // 30分钟
    size_t num_satellites = 8; // 覆盖批处理与尾部

    std::vector<CompactOrbitalElements> test_elements;
    test_elements.reserve(num_satellites);
    for (size_t i = 0; i < num_satellites; ++i) {
        CompactOrbitalElements elem = elements_;
        elem.M = i * 0.1; // 平近点角略微扰动
        test_elements.push_back(elem);
    }

    // SCALAR模式
    ConstellationPropagator prop_scalar(0.0);
    prop_scalar.setComputeMode(ConstellationPropagator::CPU_SCALAR);
    for (const auto& elem : test_elements) prop_scalar.addSatellite(elem);
    prop_scalar.propagateConstellation(propagation_time);

    // CUDA模式
    ConstellationPropagator prop_cuda(0.0);
    prop_cuda.setComputeMode(ConstellationPropagator::GPU_CUDA);
    for (const auto& elem : test_elements) prop_cuda.addSatellite(elem);
    prop_cuda.propagateConstellation(propagation_time);

    // 比较每颗卫星状态
    for (size_t i = 0; i < num_satellites; ++i) {
        StateVector s_scalar = prop_scalar.getSatelliteState(i);
        StateVector s_cuda   = prop_cuda.getSatelliteState(i);

        double pos_diff = (s_scalar.r - s_cuda.r).norm();
        double vel_diff = (s_scalar.v - s_cuda.v).norm();

        EXPECT_LT(pos_diff, kPosTol)   << "Satellite " << i << " position difference (tol=" << kPosTol << "): " << pos_diff;
        EXPECT_LT(vel_diff, kVelTol)   << "Satellite " << i << " velocity difference (tol=" << kVelTol << "): " << vel_diff;
    }
}