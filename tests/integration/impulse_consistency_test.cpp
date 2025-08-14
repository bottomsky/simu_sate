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

static const double kPosTol = env_or_default("IMPULSE_POS_TOL_M", 1.0);
static const double kVelTol = env_or_default("IMPULSE_VEL_TOL_MPS", 1e-3);

class ImpulseConsistencyTest : public ::testing::Test {
protected:
    void SetUp() override {
        base_.a = 42164.169e3; // GEO
        base_.e = 0.001;
        base_.i = 1.0 * M_PI / 180.0;
        base_.O = 0.0;
        base_.w = 0.0;
        base_.M = 0.0;
    }

    std::vector<CompactOrbitalElements> makeConstellation(size_t N) const {
        std::vector<CompactOrbitalElements> v; v.reserve(N);
        for (size_t k = 0; k < N; ++k) {
            CompactOrbitalElements e = base_;
            e.M = base_.M + 0.05 * k; // 轻微相位差
            v.push_back(e);
        }
        return v;
    }

    CompactOrbitalElements base_;
};

TEST_F(ImpulseConsistencyTest, ScalarVsSIMD_ImpulseAtT0) {
    const size_t N = 16; // 覆盖批处理
    auto sats = makeConstellation(N);

    // 构造每颗卫星不同的ΔV（在ECI系）
    std::vector<Eigen::Vector3d> dvs; dvs.reserve(N);
    for (size_t i = 0; i < N; ++i) {
        double scale = 0.1 + 0.01 * i; // m/s 量级
        dvs.emplace_back(0.0, scale, -0.5 * scale);
    }

    // SCALAR
    ConstellationPropagator prop_scalar(0.0);
    prop_scalar.setComputeMode(ConstellationPropagator::CPU_SCALAR);
    prop_scalar.addSatellites(sats);
    prop_scalar.applyImpulseToConstellation(dvs, 0.0);

    // SIMD
    ConstellationPropagator prop_simd(0.0);
    prop_simd.setComputeMode(ConstellationPropagator::CPU_SIMD);
    prop_simd.addSatellites(sats);
    prop_simd.applyImpulseToConstellation(dvs, 0.0);

    // 对比更新后的状态
    for (size_t i = 0; i < N; ++i) {
        StateVector s1 = prop_scalar.getSatelliteState(i);
        StateVector s2 = prop_simd.getSatelliteState(i);
        double pos_diff = (s1.r - s2.r).norm();
        double vel_diff = (s1.v - s2.v).norm();
        EXPECT_LT(pos_diff, kPosTol) << "Sat " << i << " position diff too large: " << pos_diff;
        EXPECT_LT(vel_diff, kVelTol) << "Sat " << i << " velocity diff too large: " << vel_diff;
    }
}

TEST_F(ImpulseConsistencyTest, ScalarVsCUDA_ImpulseAtT0) {
#if HAVE_CUDA_TOOLKIT
    if (!ConstellationPropagator::isCudaAvailable()) {
        GTEST_SKIP() << "[GPU CUDA] CUDA运行时不可用，跳过测试";
    }
#else
    GTEST_SKIP() << "[GPU CUDA] 编译期禁用CUDA，跳过测试";
#endif

    const size_t N = 12;
    auto sats = makeConstellation(N);

    std::vector<Eigen::Vector3d> dvs; dvs.reserve(N);
    for (size_t i = 0; i < N; ++i) {
        double scale = 0.08 + 0.008 * i;
        dvs.emplace_back(0.2 * scale, -0.1 * scale, 0.05 * scale);
    }

    // SCALAR
    ConstellationPropagator prop_scalar(0.0);
    prop_scalar.setComputeMode(ConstellationPropagator::CPU_SCALAR);
    prop_scalar.addSatellites(sats);
    prop_scalar.applyImpulseToConstellation(dvs, 0.0);

    // CUDA
    ConstellationPropagator prop_cuda(0.0);
    prop_cuda.setComputeMode(ConstellationPropagator::GPU_CUDA);
    prop_cuda.addSatellites(sats);
    prop_cuda.applyImpulseToConstellation(dvs, 0.0);

    for (size_t i = 0; i < N; ++i) {
        StateVector s1 = prop_scalar.getSatelliteState(i);
        StateVector s2 = prop_cuda.getSatelliteState(i);
        double pos_diff = (s1.r - s2.r).norm();
        double vel_diff = (s1.v - s2.v).norm();
        EXPECT_LT(pos_diff, kPosTol) << "(CUDA) Sat " << i << " position diff too large: " << pos_diff;
        EXPECT_LT(vel_diff, kVelTol) << "(CUDA) Sat " << i << " velocity diff too large: " << vel_diff;
    }
}