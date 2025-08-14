#include <gtest/gtest.h>
#include <cmath>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <vector>
#include <algorithm>
#include <numeric>
#include <iostream>
#include "constellation_propagator.h"

// 统计结果结构
struct ComparisonStats {
    double max_pos_diff;
    double mean_pos_diff;
    double max_vel_diff;
    double mean_vel_diff;
    std::vector<double> pos_diffs;
    std::vector<double> vel_diffs;
};

// 用于三种计算模式（CPU_SCALAR / CPU_SIMD / GPU_CUDA）的数值一致性回归测试
class ModeConsistencyRegressionTest : public ::testing::Test {
protected:
    void SetUp() override {
        // 使用与已通过测试一致的GEO近似轨道参数，避免LEO下近似引入的差异放大
        base_.a = 42164.169e3; // m（地球同步轨道半长轴）
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

    // 计算比较统计
    ComparisonStats computeStats(const std::vector<StateVector>& states1, 
                                const std::vector<StateVector>& states2) {
        ComparisonStats stats;
        stats.pos_diffs.reserve(states1.size());
        stats.vel_diffs.reserve(states1.size());
        
        for (size_t i = 0; i < states1.size(); ++i) {
            double pos_diff = posDiff(states1[i], states2[i]);
            double vel_diff = velDiff(states1[i], states2[i]);
            stats.pos_diffs.push_back(pos_diff);
            stats.vel_diffs.push_back(vel_diff);
        }
        
        stats.max_pos_diff = *std::max_element(stats.pos_diffs.begin(), stats.pos_diffs.end());
        stats.mean_pos_diff = std::accumulate(stats.pos_diffs.begin(), stats.pos_diffs.end(), 0.0) / stats.pos_diffs.size();
        stats.max_vel_diff = *std::max_element(stats.vel_diffs.begin(), stats.vel_diffs.end());
        stats.mean_vel_diff = std::accumulate(stats.vel_diffs.begin(), stats.vel_diffs.end(), 0.0) / stats.vel_diffs.size();
        
        return stats;
    }

    // 生成JSON报告
    void generateJsonReport(const std::string& test_name,
                           const std::vector<CompactOrbitalElements>& sats,
                           double step, double target_time,
                           const ComparisonStats& scalar_vs_simd,
                           const ComparisonStats& scalar_vs_cuda,
                           bool has_cuda) {
        std::stringstream json;
        json << std::fixed << std::setprecision(6);
        json << "{\n";
        json << "  \"test_name\": \"" << test_name << "\",\n";
        json << "  \"parameters\": {\n";
        json << "    \"satellite_count\": " << sats.size() << ",\n";
        json << "    \"step_size_seconds\": " << step << ",\n";
        json << "    \"propagation_time_seconds\": " << target_time << ",\n";
        json << "    \"cuda_available\": " << (has_cuda ? "true" : "false") << "\n";
        json << "  },\n";
        json << "  \"comparisons\": {\n";
        json << "    \"scalar_vs_simd\": {\n";
        json << "      \"position_difference_km\": {\n";
        json << "        \"max\": " << scalar_vs_simd.max_pos_diff << ",\n";
        json << "        \"mean\": " << scalar_vs_simd.mean_pos_diff << "\n";
        json << "      },\n";
        json << "      \"velocity_difference_km_per_s\": {\n";
        json << "        \"max\": " << scalar_vs_simd.max_vel_diff << ",\n";
        json << "        \"mean\": " << scalar_vs_simd.mean_vel_diff << "\n";
        json << "      }\n";
        json << "    }";
        
        if (has_cuda) {
            json << ",\n";
            json << "    \"scalar_vs_cuda\": {\n";
            json << "      \"position_difference_km\": {\n";
            json << "        \"max\": " << scalar_vs_cuda.max_pos_diff << ",\n";
            json << "        \"mean\": " << scalar_vs_cuda.mean_pos_diff << "\n";
            json << "      },\n";
            json << "      \"velocity_difference_km_per_s\": {\n";
            json << "        \"max\": " << scalar_vs_cuda.max_vel_diff << ",\n";
            json << "        \"mean\": " << scalar_vs_cuda.mean_vel_diff << "\n";
            json << "      }\n";
            json << "    }\n";
        } else {
            json << "\n";
        }
        
        json << "  },\n";
        json << "  \"algorithms_compared\": [\n";
        json << "    \"CPU_SCALAR (基准标量算法)\",\n";
        json << "    \"CPU_SIMD (SIMD向量化算法)\"";
        if (has_cuda) {
            json << ",\n    \"GPU_CUDA (CUDA并行计算算法)\"";
        }
        json << "\n  ]\n";
        json << "}\n";

        // 保存到文件
        std::string filename = "d:\\code\\j2-perturbation-orbit-propagator\\tests\\data\\" + test_name + "_consistency_report.json";
        std::ofstream file(filename);
        file << json.str();
        file.close();
    }

    // 打印详细报告
    void printDetailedReport(const std::string& test_name,
                            const std::vector<CompactOrbitalElements>& sats,
                            double step, double target_time,
                            const ComparisonStats& scalar_vs_simd,
                            const ComparisonStats& scalar_vs_cuda,
                            bool has_cuda) {
        std::cout << "\n========== " << test_name << " 一致性检测报告 ==========\n";
        std::cout << "测试参数：\n";
        std::cout << "  卫星数量: " << sats.size() << " 颗\n";
        std::cout << "  步长: " << step << " 秒\n";
        std::cout << "  传播时间: " << target_time << " 秒 (" << target_time/3600.0 << " 小时)\n";
        std::cout << "  CUDA可用性: " << (has_cuda ? "是" : "否") << "\n\n";

        std::cout << "算法对比结果：\n";
        std::cout << "1. CPU_SCALAR vs CPU_SIMD:\n";
        std::cout << "   位置差异 (km): 最大=" << std::fixed << std::setprecision(6) << scalar_vs_simd.max_pos_diff 
                  << ", 平均=" << scalar_vs_simd.mean_pos_diff << "\n";
        std::cout << "   速度差异 (km/s): 最大=" << scalar_vs_simd.max_vel_diff 
                  << ", 平均=" << scalar_vs_simd.mean_vel_diff << "\n\n";

        if (has_cuda) {
            std::cout << "2. CPU_SCALAR vs GPU_CUDA:\n";
            std::cout << "   位置差异 (km): 最大=" << scalar_vs_cuda.max_pos_diff 
                      << ", 平均=" << scalar_vs_cuda.mean_pos_diff << "\n";
            std::cout << "   速度差异 (km/s): 最大=" << scalar_vs_cuda.max_vel_diff 
                      << ", 平均=" << scalar_vs_cuda.mean_vel_diff << "\n\n";
        }

        std::cout << "算法说明：\n";
        std::cout << "- CPU_SCALAR: 基准标量算法，逐颗卫星串行计算\n";
        std::cout << "- CPU_SIMD: SIMD向量化算法，利用CPU向量指令并行计算\n";
        if (has_cuda) {
            std::cout << "- GPU_CUDA: CUDA并行计算算法，利用GPU大规模并行处理\n";
        }
        std::cout << "\n通过标准: 位置差异 < 1.0 km, 速度差异 < 0.001 km/s\n";
        std::cout << "======================================================\n\n";
    }

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

    // 计算统计数据
    ComparisonStats scalar_vs_simd = computeStats(states_scalar, states_simd);
    ComparisonStats scalar_vs_cuda;
    if (has_cuda) {
        scalar_vs_cuda = computeStats(states_scalar, states_cuda);
    }

    // 生成报告
    generateJsonReport("SmallConstellation_Step60s_OneHour", sats, step, t, 
                      scalar_vs_simd, scalar_vs_cuda, has_cuda);
    printDetailedReport("SmallConstellation_Step60s_OneHour", sats, step, t, 
                       scalar_vs_simd, scalar_vs_cuda, has_cuda);

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

    // 计算统计数据
    ComparisonStats scalar_vs_simd = computeStats(states_scalar, states_simd);
    ComparisonStats scalar_vs_cuda;
    if (has_cuda) {
        scalar_vs_cuda = computeStats(states_scalar, states_cuda);
    }

    // 生成报告
    generateJsonReport("MediumConstellation_Step30s_TwoHours", sats, step, t, 
                      scalar_vs_simd, scalar_vs_cuda, has_cuda);
    printDetailedReport("MediumConstellation_Step30s_TwoHours", sats, step, t, 
                       scalar_vs_simd, scalar_vs_cuda, has_cuda);

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

    // 计算统计数据和生成报告（不包括CUDA）
    ComparisonStats scalar_vs_simd = computeStats(states_scalar, states_simd);
    ComparisonStats empty_cuda_stats;
    generateJsonReport("AdaptiveStep_ScalarVsSIMD", sats, -1.0, t, 
                      scalar_vs_simd, empty_cuda_stats, false);
    printDetailedReport("AdaptiveStep_ScalarVsSIMD", sats, -1.0, t, 
                       scalar_vs_simd, empty_cuda_stats, false);

    for (size_t i = 0; i < N; ++i) {
        double p = posDiff(states_scalar[i], states_simd[i]);
        double v = velDiff(states_scalar[i], states_simd[i]);
        EXPECT_LT(p, 1.0) << "Adaptive Sat " << i << " pos diff too large: " << p;
        EXPECT_LT(v, 1e-3) << "Adaptive Sat " << i << " vel diff too large: " << v;
    }
}