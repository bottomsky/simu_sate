#include <gtest/gtest.h>
#include "j2_orbit_propagator.h"
#include "constellation_propagator.h"
#include "math_defs.h"
#include <fstream>
#include <iomanip>
#include <vector>
#include <string>
static const char* OUTPUT_DIR = "d:\\code\\j2-perturbation-orbit-propagator\\tests\\data\\";

class ParameterSweepTest : public ::testing::Test {
protected:
    void SetUp() override {
        // GEO轨道参数
        a = 42164.0e3;  // 半长轴 (m)
        e = 0.001;      // 偏心率
        i = 0.0 * M_PI / 180.0;  // 倾角 (rad)
        omega = 0.0;    // 近地点幅角 (rad)
        Omega = 0.0;    // 升交点赤经 (rad)
        M = 0.0;        // 平近点角 (rad)
    }

    double a, e, i, omega, Omega, M;
    
    // 计算两个状态向量间的位置差
    double positionError(const StateVector& state1, const StateVector& state2) {
        return (state1.r - state2.r).norm();
    }
    
    // 计算轨道半径误差
    double radiusError(const StateVector& state, double expected_radius) {
        return std::abs(state.r.norm() - expected_radius);
    }
};

TEST_F(ParameterSweepTest, StepSizeAccuracyAnalysis) {
    std::vector<double> step_sizes = {1.0, 5.0, 10.0, 30.0, 60.0, 300.0, 600.0, 1800.0, 3600.0};
    std::vector<double> propagation_times = {3600.0, 7200.0, 14400.0, 43200.0, 86164.1}; // 1h, 2h, 4h, 12h, 1天
    
    std::ofstream csv_file(std::string(OUTPUT_DIR) + "step_size_accuracy.csv");
    csv_file << "StepSize(s),PropTime(s),RadiusError(m),PositionError(m),PropTimeHours,StepsCount\n";
    
    std::cout << "\n=== 步长与精度分析 ===\n";
    std::cout << std::fixed << std::setprecision(1);
    
    for (double prop_time : propagation_times) {
        std::cout << "\n外推时间: " << prop_time/3600.0 << " 小时\n";
        std::cout << "步长(s)\t半径误差(m)\t位置误差(m)\t步数\n";
        std::cout << std::string(50, '-') << "\n";
        
        // 参考解：使用最小步长（1秒）
        OrbitalElements elements_ref = {a, e, i, Omega, omega, M, 0.0};
        J2OrbitPropagator ref_propagator(elements_ref);
        ref_propagator.setStepSize(1.0);
        auto ref_elements = ref_propagator.propagate(prop_time);
        auto ref_state = ref_propagator.elementsToState(ref_elements);
        
        for (double step_size : step_sizes) {
            OrbitalElements elements = {a, e, i, Omega, omega, M, 0.0};
            J2OrbitPropagator propagator(elements);
            propagator.setStepSize(step_size);
            
            auto result_elements = propagator.propagate(prop_time);
            auto result_state = propagator.elementsToState(result_elements);
            
            double radius_error = radiusError(result_state, a);
            double position_error = positionError(result_state, ref_state);
            int steps_count = static_cast<int>(prop_time / step_size);
            
            std::cout << step_size << "\t\t" << radius_error << "\t\t" 
                     << position_error << "\t\t" << steps_count << "\n";
            
            csv_file << step_size << "," << prop_time << "," << radius_error << "," 
                    << position_error << "," << prop_time/3600.0 << "," << steps_count << "\n";
        }
    }
    
    csv_file.close();
    std::cout << "\n结果已保存到 " << OUTPUT_DIR << "step_size_accuracy.csv\n";
}

TEST_F(ParameterSweepTest, EccentricityEffectAnalysis) {
    std::vector<double> eccentricities = {0.0, 0.001, 0.005, 0.01, 0.02, 0.05, 0.1};
    double step_size = 60.0;  // 1分钟步长
    double prop_time = 86164.1;  // 一个恒星日
    
    std::ofstream csv_file(std::string(OUTPUT_DIR) + "eccentricity_effect.csv");
    csv_file << "Eccentricity,SemiMajorAxis(km),RadiusError(m),PositionVariation(m),PeriapsisDist(km),ApoapsisDist(km)\n";
    
    std::cout << "\n=== 偏心率对精度的影响 ===\n";
    std::cout << "偏心率\t半径误差(m)\t位置变化(m)\t近地点(km)\t远地点(km)\n";
    std::cout << std::string(60, '-') << "\n";
    
    for (double ecc : eccentricities) {
        OrbitalElements elements = {a, ecc, i, Omega, omega, M, 0.0};
        J2OrbitPropagator propagator(elements);
        propagator.setStepSize(step_size);
        
        // 记录初始状态
        auto initial_state = propagator.elementsToState(elements);
        double initial_radius = initial_state.r.norm();
        
        // 外推一个轨道周期
        auto final_elements = propagator.propagate(prop_time);
        auto final_state = propagator.elementsToState(final_elements);
        
        double radius_error = radiusError(final_state, a);
        double position_variation = positionError(final_state, initial_state);
        
        // 计算近地点和远地点距离
        double periapsis = a * (1.0 - ecc);
        double apoapsis = a * (1.0 + ecc);
        
        std::cout << std::fixed << std::setprecision(3) << ecc << "\t\t" 
                 << std::setprecision(1) << radius_error << "\t\t" 
                 << position_variation << "\t\t" 
                 << periapsis/1000.0 << "\t\t" << apoapsis/1000.0 << "\n";
        
        csv_file << ecc << "," << a/1000.0 << "," << radius_error << "," 
                << position_variation << "," << periapsis/1000.0 << "," 
                << apoapsis/1000.0 << "\n";
    }
    
    csv_file.close();
    std::cout << "\n结果已保存到 " << OUTPUT_DIR << "eccentricity_effect.csv\n";
}

TEST_F(ParameterSweepTest, OptimalParameterRecommendation) {
    std::cout << "\n=== 精度要求与推荐参数 ===\n";
    
    struct AccuracyTarget {
        std::string name;
        double max_error_m;
        double recommended_step_s;
    };
    
    std::vector<AccuracyTarget> targets = {
        {"高精度科学计算", 1.0, 10.0},
        {"工程应用", 10.0, 60.0},
        {"快速评估", 100.0, 300.0},
        {"粗略估算", 1000.0, 1800.0}
    };
    
    double test_time = 86164.1;  // 24小时测试
    
    std::cout << "应用场景\t\t精度要求(m)\t推荐步长(s)\t验证误差(m)\t计算步数\n";
    std::cout << std::string(70, '-') << "\n";
    
    for (const auto& target : targets) {
        OrbitalElements elements = {a, e, i, Omega, omega, M, 0.0};
        J2OrbitPropagator propagator(elements);
        propagator.setStepSize(target.recommended_step_s);
        
        auto initial_state = propagator.elementsToState(elements);
        auto final_elements = propagator.propagate(test_time);
        auto final_state = propagator.elementsToState(final_elements);
        
        double actual_error = radiusError(final_state, a);
        int steps = static_cast<int>(test_time / target.recommended_step_s);
        
        std::cout << std::left << std::setw(16) << target.name << "\t"
                 << std::setw(8) << target.max_error_m << "\t"
                 << std::setw(8) << target.recommended_step_s << "\t\t"
                 << std::setw(8) << std::setprecision(1) << actual_error << "\t\t"
                 << steps << "\n";
    }
    
    std::cout << "\n推荐策略：\n";
    std::cout << "- 对于 1km 精度要求，建议使用 ≤60秒 步长\n";
    std::cout << "- 对于高偏心率轨道 (e>0.01)，建议减小步长至 ≤30秒\n";
    std::cout << "- 长时间外推 (>12小时)，建议使用自适应步长控制\n";
}

TEST_F(ParameterSweepTest, AdaptiveStepSizeDemo) {
    std::cout << "\n=== 自适应步长演示 ===\n";
    
    double target_error = 10.0;  // 目标精度10米
    double prop_time = 86164.1;
    
    // 简单的自适应算法：根据偏心率调整步长
    auto calculateAdaptiveStepSize = [](double eccentricity, double base_step) {
        // 偏心率越大，步长越小
        double factor = 1.0 / (1.0 + 10.0 * eccentricity);
        return base_step * factor;
    };
    
    std::vector<double> test_eccentricities = {0.001, 0.01, 0.05, 0.1};
    double base_step = 60.0;
    
    std::cout << "偏心率\t固定步长误差(m)\t自适应步长(s)\t自适应误差(m)\t改进比例\n";
    std::cout << std::string(70, '-') << "\n";
    
    for (double ecc : test_eccentricities) {
        // 固定步长测试
        OrbitalElements elements1 = {a, ecc, i, Omega, omega, M, 0.0};
        J2OrbitPropagator prop_fixed(elements1);
        prop_fixed.setStepSize(base_step);
        auto final1 = prop_fixed.propagate(prop_time);
        auto state1 = prop_fixed.elementsToState(final1);
        double fixed_error = radiusError(state1, a);
        
        // 自适应步长测试
        double adaptive_step = calculateAdaptiveStepSize(ecc, base_step);
        OrbitalElements elements2 = {a, ecc, i, Omega, omega, M, 0.0};
        J2OrbitPropagator prop_adaptive(elements2);
        prop_adaptive.setStepSize(adaptive_step);
        auto final2 = prop_adaptive.propagate(prop_time);
        auto state2 = prop_adaptive.elementsToState(final2);
        double adaptive_error = radiusError(state2, a);
        
        double improvement = fixed_error / adaptive_error;
        
        std::cout << std::fixed << std::setprecision(3) << ecc << "\t\t"
                 << std::setprecision(1) << fixed_error << "\t\t\t"
                 << std::setprecision(1) << adaptive_step << "\t\t"
                 << adaptive_error << "\t\t"
                 << std::setprecision(2) << improvement << "x\n";
    }
}

TEST_F(ParameterSweepTest, AnalyticalJ2SecularRatesValidation) {
    // 基于 J2 的解析世俗项对 O、w、M 的变化进行验证。
    auto angle_diff = [](double a1, double a2) {
        double d = fmod(a1 - a2, 2.0 * M_PI);
        if (d > M_PI) d -= 2.0 * M_PI;
        if (d < -M_PI) d += 2.0 * M_PI;
        return d;
    };

    struct Case { double a_m; double e; double i_rad; double Omega0; double w0; double M0; double T; double step; };
    std::vector<Case> cases = {
        // GEO 轨道，一天与三天
        {42164e3, 0.001, 0.0 * M_PI/180.0, 0.0, 0.0, 0.0, 86164.1, 1.0},
        {42164e3, 0.001, 0.0 * M_PI/180.0, 0.2, 0.5, 1.0, 3*86164.1, 1.0},
        // 中等倾角、轻微偏心率
        {26560e3, 0.01, 30.0 * M_PI/180.0, 0.3, 1.0, 2.0, 86164.1, 1.0},
        // 近临界倾角 63.4°
        {26560e3, 0.005, 63.4 * M_PI/180.0, 1.0, 2.0, 0.1, 86164.1, 1.0},
    };

    for (const auto& c : cases) {
        // 计算解析世俗率
        double n = std::sqrt(MU / (c.a_m*c.a_m*c.a_m));
        double p = c.a_m * (1.0 - c.e*c.e);
        double factor = 1.5 * n * J2 * std::pow(RE / p, 2);
        double dOdt = -factor * std::cos(c.i_rad);
        double dwdt =  factor * (2.5 * std::sin(c.i_rad)*std::sin(c.i_rad) - 2.0);
        double dMdt =  n - factor * std::sqrt(1.0 - c.e*c.e) * (1.5 * std::sin(c.i_rad)*std::sin(c.i_rad) - 0.5);

        // 解析解（只考虑世俗项）
        double O_pred = c.Omega0 + dOdt * c.T;
        double w_pred = c.w0     + dwdt * c.T;
        double M_pred = c.M0     + dMdt * c.T;

        // 数值解
        OrbitalElements el0{c.a_m, c.e, c.i_rad, c.Omega0, c.w0, c.M0, 0.0};
        J2OrbitPropagator prop(el0);
        prop.setStepSize(c.step);
        auto elT = prop.propagate(c.T);

        // 验证角度误差
        double dO = angle_diff(elT.O, O_pred);
        double dw = angle_diff(elT.w, w_pred);
        double dM = angle_diff(elT.M, M_pred);

        // 阈值：按时长放宽（弧度）
        double scale = c.T / 86164.1; // 相对于一天
        double tol_Ow = 1e-4 * scale + 3e-5; // ~0.006°/day 以内
        double tol_M  = 5e-4 * scale + 1e-4; // M 更敏感，给更宽裕

        EXPECT_LT(std::abs(dO), tol_Ow) << "dO exceed tol for case with a=" << c.a_m << ", i=" << c.i_rad;
        EXPECT_LT(std::abs(dw), tol_Ow) << "dw exceed tol for case with a=" << c.a_m << ", i=" << c.i_rad;
        EXPECT_LT(std::abs(dM), tol_M)  << "dM exceed tol for case with a=" << c.a_m << ", i=" << c.i_rad;
    }
}

TEST_F(ParameterSweepTest, LongTermStabilityValidation) {
    // 使用不同轨道类型进行较长时段外推，验证不变量与解析世俗项的一致性
    struct Case { double a_m; double e; double i_rad; double O0; double w0; double M0; double days; double step; };
    std::vector<Case> cases = {
        // 典型LEO
        {7000e3, 0.001, 45.0 * M_PI/180.0, 0.2, 0.5, 1.0, 7.0, 60.0},
        // MEO (近GPS高度)
        {26560e3, 0.01, 55.0 * M_PI/180.0, 0.1, 1.2, 2.0, 7.0, 60.0},
        // GEO 轻微偏心
        {42164e3, 0.001, 0.0 * M_PI/180.0, 0.0, 0.0, 0.0, 7.0, 60.0},
    };

    auto angle_diff = [](double a1, double a2) {
        double d = fmod(a1 - a2, 2.0 * M_PI);
        if (d > M_PI) d -= 2.0 * M_PI;
        if (d < -M_PI) d += 2.0 * M_PI;
        return d;
    };

    // 输出CSV以便后续绘图分析
    std::ofstream csv(std::string(OUTPUT_DIR) + "long_term_stability.csv");
    csv << "case,a_m(m),e,i(rad),days,step(s),dO(rad),dw(rad),dM(rad)\n";

    for (size_t idx = 0; idx < cases.size(); ++idx) {
        const auto& c = cases[idx];
        double T = c.days * 86164.1; // 使用恒星日
        // 解析世俗率
        double n = std::sqrt(MU / (c.a_m*c.a_m*c.a_m));
        double p = c.a_m * (1.0 - c.e*c.e);
        double factor = 1.5 * n * J2 * std::pow(RE / p, 2);
        double dOdt = -factor * std::cos(c.i_rad);
        double dwdt =  factor * (2.5 * std::sin(c.i_rad)*std::sin(c.i_rad) - 2.0);
        double dMdt =  n - factor * std::sqrt(1.0 - c.e*c.e) * (1.5 * std::sin(c.i_rad)*std::sin(c.i_rad) - 0.5);

        double O_pred = c.O0 + dOdt * T;
        double w_pred = c.w0 + dwdt * T;
        double M_pred = c.M0 + dMdt * T;

        OrbitalElements el0{c.a_m, c.e, c.i_rad, c.O0, c.w0, c.M0, 0.0};
        J2OrbitPropagator prop(el0);
        prop.setStepSize(c.step);
        auto elT = prop.propagate(T);

        // 不变量检查：a, e, i 应保持不变（J2纯摄动模型下导数为0）
        EXPECT_NEAR(elT.a, el0.a, 1e-9) << "a drifted";
        EXPECT_NEAR(elT.e, el0.e, 1e-12) << "e drifted";
        EXPECT_NEAR(elT.i, el0.i, 1e-12) << "i drifted";

        // 角要素与解析世俗项一致性（容差随时间放宽）
        double scale = T / 86164.1; // 天数
        double tol_Ow = 1e-4 * scale + 3e-5; // rad
        double tol_M  = 5e-4 * scale + 1e-4; // rad

        double dO = angle_diff(elT.O, O_pred);
        double dw = angle_diff(elT.w, w_pred);
        double dM = angle_diff(elT.M, M_pred);

        // 写入CSV
        csv << idx << "," << c.a_m << "," << c.e << "," << c.i_rad << "," << c.days << "," << c.step
            << "," << dO << "," << dw << "," << dM << "\n";

        EXPECT_LT(std::abs(dO), tol_Ow) << "Long-term dO exceeds tol";
        EXPECT_LT(std::abs(dw), tol_Ow) << "Long-term dw exceeds tol";
        EXPECT_LT(std::abs(dM), tol_M)  << "Long-term dM exceeds tol";
    }
}

TEST_F(ParameterSweepTest, ExtremeParameterStressTest) {
    // 覆盖高偏心率、大倾角、接近极轨、接近临界倾角、低近地点等边界情况
    struct Case { double a_m; double e; double i_rad; double O0; double w0; double M0; double T; double step; };
    std::vector<Case> cases = {
        // 高偏心率中轨道（保证 p = a(1-e^2) > 0）
        {26560e3, 0.2, 30.0 * M_PI/180.0, 0.0, 1.0, 0.5, 2*3600.0, 10.0},
        // 近极轨
        {7000e3, 0.01, 89.0 * M_PI/180.0, 0.2, 0.4, 1.0, 2*3600.0, 10.0},
        // 临界倾角附近（63.4°）
        {26560e3, 0.005, 63.4 * M_PI/180.0, 0.3, 0.7, 2.0, 3*3600.0, 10.0},
        // 低近地点（近地面200km）
        { (RE + 200e3), 0.0, 51.6 * M_PI/180.0, 1.0, 2.0, 0.0, 2*3600.0, 1.0},
        // 高倾角高偏心
        {15000e3, 0.3, 70.0 * M_PI/180.0, 0.5, 2.5, 3.0, 2*3600.0, 5.0},
    };

    for (const auto& c : cases) {
        // 构造并外推
        OrbitalElements el0{c.a_m, c.e, c.i_rad, c.O0, c.w0, c.M0, 0.0};
        J2OrbitPropagator prop(el0);
        prop.setStepSize(c.step);
        auto elT = prop.propagate(c.T);

        // 有效性检查：元素应为有限数且在合理范围
        auto finite = [](double x){ return std::isfinite(x); };
        EXPECT_TRUE(finite(elT.a) && finite(elT.e) && finite(elT.i) && finite(elT.O) && finite(elT.w) && finite(elT.M));

        // e与i应保持接近初值（J2模型）
        EXPECT_NEAR(elT.e, el0.e, 1e-8) << "e deviated too much under stress";
        EXPECT_NEAR(elT.i, el0.i, 1e-8) << "i deviated too much under stress";

        // 半参数 p = a(1-e^2) 不应接近0（避免奇异）
        double p = elT.a * (1.0 - elT.e * elT.e);
        EXPECT_GT(p, 1.0) << "semilatus rectum too small";

        // 角度应归一化到[-pi, pi]后可对比（这里只检查值有限即可）
        EXPECT_TRUE(finite(elT.O) && finite(elT.w) && finite(elT.M));
    }
}

TEST_F(ParameterSweepTest, StepSizeConvergenceAnalysis) {
    // 在J2纯摄动（世俗项）下，导数为常量，步长对数值解影响应极小。
    // 本用例分析不同步长的误差曲线并验证误差保持在合理范围内。
    OrbitalElements el0{7000e3, 0.01, 30.0 * M_PI/180.0, 0.2, 0.5, 1.0, 0.0};
    double T = 6 * 3600.0; // 6小时

    // 参考解（更细步长）
    J2OrbitPropagator prop_ref(el0);
    prop_ref.setStepSize(0.5);
    auto el_ref = prop_ref.propagate(T);
    StateVector state_ref = prop_ref.elementsToState(el_ref);

    auto pos_err_vs_step = [&](double step){
        J2OrbitPropagator p(el0);
        p.setStepSize(step);
        auto elT = p.propagate(T);
        StateVector sT = p.elementsToState(elT);
        return (sT.r - state_ref.r).norm();
    };

    std::vector<double> steps = {600.0, 300.0, 120.0, 60.0, 30.0, 15.0};
    std::vector<double> errs;
    errs.reserve(steps.size());
    for (double h : steps) errs.push_back(pos_err_vs_step(h));

    // 导出CSV
    std::ofstream csv(std::string(OUTPUT_DIR) + "step_size_convergence.csv");
    csv << "step(s),pos_err(m)\n";
    for (size_t k = 0; k < steps.size(); ++k) {
        csv << steps[k] << "," << errs[k] << "\n";
    }

    // 检查误差上界（位置误差应远小于百米量级）
    double max_err = 0.0;
    for (double e : errs) max_err = std::max(max_err, e);
    EXPECT_LT(max_err, 100.0) << "position error exceeds bound across step sizes";
}

TEST_F(ParameterSweepTest, ThreePathConsistency_AllModes) {
    // 比较 CPU_SCALAR / CPU_SIMD / GPU_CUDA 三种路径的一致性
    // 构造一组卫星
    std::vector<CompactOrbitalElements> sats;
    for (int i = 0; i < 12; ++i) {
        CompactOrbitalElements e1;
        e1.a = 7000e3 + i * 10.0; // 轻微扰动
        e1.e = 0.01 + 0.0001 * i;
        e1.i = (30.0 + i) * M_PI / 180.0;
        e1.O = 0.1 * i;
        e1.w = 0.2 + 0.05 * i;
        e1.M = 0.3 + 0.07 * i;
        sats.push_back(e1);
    }

    double T = 7200.0; // 2小时

    // 标量
    ConstellationPropagator prop_scalar(0.0);
    prop_scalar.setComputeMode(ConstellationPropagator::CPU_SCALAR);
    prop_scalar.setStepSize(30.0);
    prop_scalar.addSatellites(sats);
    prop_scalar.propagateConstellation(T);

    // SIMD
    ConstellationPropagator prop_simd(0.0);
    prop_simd.setComputeMode(ConstellationPropagator::CPU_SIMD);
    prop_simd.setStepSize(30.0);
    prop_simd.addSatellites(sats);
    prop_simd.propagateConstellation(T);

    // CUDA（若可用）
    bool has_cuda = ConstellationPropagator::isCudaAvailable();
    ConstellationPropagator prop_cuda(0.0);
    if (has_cuda) {
        prop_cuda.setComputeMode(ConstellationPropagator::GPU_CUDA);
        prop_cuda.setStepSize(30.0);
        prop_cuda.addSatellites(sats);
        prop_cuda.propagateConstellation(T);
    }

    // 比较结果
    for (size_t i = 0; i < sats.size(); ++i) {
        StateVector s_scalar = prop_scalar.getSatelliteState(i);
        StateVector s_simd   = prop_simd.getSatelliteState(i);

        double pos_diff_ss = (s_scalar.r - s_simd.r).norm();
        double vel_diff_ss = (s_scalar.v - s_simd.v).norm();
        EXPECT_LT(pos_diff_ss, 1.0)   << "Scalar vs SIMD position diff too large for sat " << i;
        EXPECT_LT(vel_diff_ss, 1e-3)  << "Scalar vs SIMD velocity diff too large for sat " << i;

        if (has_cuda) {
            StateVector s_cuda = prop_cuda.getSatelliteState(i);
            double pos_diff_sc = (s_scalar.r - s_cuda.r).norm();
            double vel_diff_sc = (s_scalar.v - s_cuda.v).norm();
            double pos_diff_mc = (s_simd.r   - s_cuda.r).norm();
            double vel_diff_mc = (s_simd.v   - s_cuda.v).norm();

            EXPECT_LT(pos_diff_sc, 1.0)   << "Scalar vs CUDA position diff too large for sat " << i;
            EXPECT_LT(vel_diff_sc, 1e-3)  << "Scalar vs CUDA velocity diff too large for sat " << i;
            EXPECT_LT(pos_diff_mc, 1.0)   << "SIMD vs CUDA position diff too large for sat " << i;
            EXPECT_LT(vel_diff_mc, 1e-3)  << "SIMD vs CUDA velocity diff too large for sat " << i;
        }
    }
}