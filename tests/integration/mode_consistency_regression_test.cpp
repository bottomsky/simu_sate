#include <gtest/gtest.h>
#include <cmath>
#include "constellation_propagator.h"

// 用于三种计算模式（CPU_SCALAR / CPU_SIMD / GPU_CUDA）的数值一致性回归测试
class ModeConsistencyRegressionTest : public ::testing::Test {
protected:
    void SetUp() override {
        // 使用与已通过测试一致的GEO近似轨道参数，避免LEO下近似引入的差异放大
        base_.a = 42164.169; // km（地球同步轨道半长轴）
        base_.e = 0.001;
        base_.i = 1.0 * M_PI / 180.0; // 1度
        base_.O = 0.0;
        base_.w = 0.0;
        base_.M = 0.0;
    }

    // 生成N颗卫星，平近点角略微错相
    std::vector<CompactOrbitalElements> makeConstellation(size_t N) const {
        std::vector<CompactOrbitalElements> v; v.reserve(N);
        for (size_t k = 0; k < N; ++k) {
            CompactOrbitalElements e = base_;
            e.M = base_.M + 0.03 * k; // 小幅相位差
            v.push_back(e);
        }
        return v;
    }

    // 在指定模式下进行传播，并返回所有卫星的位置矩阵（3xN）及单星状态便于比较
    std::vector<StateVector> propagateInMode(const std::vector<CompactOrbitalElements>& sats,
                                             ConstellationPropagator::ComputeMode mode,
                                             double step, double target_time) const {
        ConstellationPropagator prop(0.0);
        prop.setComputeMode(mode);
        prop.setStepSize(step);
        prop.addSatellites(sats);
        prop.propagateConstellation(target_time);
        std::vector<StateVector> states; states.reserve(sats.size());
        for (size_t i = 0; i < sats.size(); ++i) states.push_back(prop.getSatelliteState(i));
        return states;
    }

    // 计算范数差
    static double posDiff(const StateVector& a, const StateVector& b) { return (a.r - b.r).norm(); }
    static double velDiff(const StateVector& a, const StateVector& b) { return (a.v - b.v).norm(); }

    CompactOrbitalElements base_;
};

TEST_F(ModeConsistencyRegressionTest, SmallConstellation_Step60s_OneHour) {
    const size_t N = 12; // 覆盖SIMD批量与尾部
    auto sats = makeConstellation(N);
    const double step = 60.0; // s
    const double t = 3600.0;  // 1 hour

    // 依次拿三个模式跑
    auto states_scalar = propagateInMode(sats, ConstellationPropagator::CPU_SCALAR, step, t);
    auto states_simd   = propagateInMode(sats, ConstellationPropagator::CPU_SIMD,   step, t);
    std::vector<StateVector> states_cuda;
    bool has_cuda = ConstellationPropagator::isCudaAvailable();
    if (has_cuda) {
        states_cuda = propagateInMode(sats, ConstellationPropagator::GPU_CUDA, step, t);
    }

    // 逐星比较 SCALAR vs SIMD
    for (size_t i = 0; i < N; ++i) {
        double p = posDiff(states_scalar[i], states_simd[i]);
        double v = velDiff(states_scalar[i], states_simd[i]);
        EXPECT_LT(p, 1.0) << "Sat " << i << " pos diff too large: " << p;
        EXPECT_LT(v, 1e-3) << "Sat " << i << " vel diff too large: " << v;
    }

    // 如果CUDA可用，再比较 SCALAR vs CUDA
    if (has_cuda) {
        for (size_t i = 0; i < N; ++i) {
            double p = posDiff(states_scalar[i], states_cuda[i]);
            double v = velDiff(states_scalar[i], states_cuda[i]);
            EXPECT_LT(p, 1.0) << "(CUDA) Sat " << i << " pos diff too large: " << p;
            EXPECT_LT(v, 1e-3) << "(CUDA) Sat " << i << " vel diff too large: " << v;
        }
    }
}

TEST_F(ModeConsistencyRegressionTest, MediumConstellation_Step30s_TwoHours_AdaptiveOff) {
    const size_t N = 64; // 多批次
    auto sats = makeConstellation(N);
    const double step = 30.0; // s
    const double t = 7200.0;  // 2 hours

    auto states_scalar = propagateInMode(sats, ConstellationPropagator::CPU_SCALAR, step, t);
    auto states_simd   = propagateInMode(sats, ConstellationPropagator::CPU_SIMD,   step, t);
    std::vector<StateVector> states_cuda;
    bool has_cuda = ConstellationPropagator::isCudaAvailable();
    if (has_cuda) states_cuda = propagateInMode(sats, ConstellationPropagator::GPU_CUDA, step, t);

    for (size_t i = 0; i < N; ++i) {
        double p = posDiff(states_scalar[i], states_simd[i]);
        double v = velDiff(states_scalar[i], states_simd[i]);
        EXPECT_LT(p, 1.0) << "Sat " << i << " pos diff too large: " << p;
        EXPECT_LT(v, 1e-3) << "Sat " << i << " vel diff too large: " << v;
    }
    if (has_cuda) {
        for (size_t i = 0; i < N; ++i) {
            double p = posDiff(states_scalar[i], states_cuda[i]);
            double v = velDiff(states_scalar[i], states_cuda[i]);
            EXPECT_LT(p, 1.0) << "(CUDA) Sat " << i << " pos diff too large: " << p;
            EXPECT_LT(v, 1e-3) << "(CUDA) Sat " << i << " vel diff too large: " << v;
        }
    }
}

// 自适应步长一致性（仅CPU路径比较，SIMD与SCALAR共享自适应策略）
TEST_F(ModeConsistencyRegressionTest, AdaptiveStep_ScalarVsSIMD) {
    const size_t N = 10;
    auto sats = makeConstellation(N);
    const double t = 5400.0;  // 1.5 hours

    std::vector<StateVector> states_scalar;
    std::vector<StateVector> states_simd;

    // SCALAR 自适应（收紧容差和步长范围）
    {
        ConstellationPropagator prop(0.0);
        prop.setComputeMode(ConstellationPropagator::CPU_SCALAR);
        prop.setAdaptiveStepSize(true);
        prop.setAdaptiveParameters(1e-8, 0.5, 60.0);
        prop.addSatellites(sats);
        prop.propagateConstellation(t);
        states_scalar.reserve(N);
        for (size_t i = 0; i < N; ++i) states_scalar.push_back(prop.getSatelliteState(i));
    }

    // SIMD 自适应（同样参数）
    {
        ConstellationPropagator prop(0.0);
        prop.setComputeMode(ConstellationPropagator::CPU_SIMD);
        prop.setAdaptiveStepSize(true);
        prop.setAdaptiveParameters(1e-8, 0.5, 60.0);
        prop.addSatellites(sats);
        prop.propagateConstellation(t);
        states_simd.reserve(N);
        for (size_t i = 0; i < N; ++i) states_simd.push_back(prop.getSatelliteState(i));
    }

    for (size_t i = 0; i < N; ++i) {
        double p = posDiff(states_scalar[i], states_simd[i]);
        double v = velDiff(states_scalar[i], states_simd[i]);
        EXPECT_LT(p, 1.0) << "Adaptive Sat " << i << " pos diff too large: " << p;
        EXPECT_LT(v, 1e-3) << "Adaptive Sat " << i << " vel diff too large: " << v;
    }
}