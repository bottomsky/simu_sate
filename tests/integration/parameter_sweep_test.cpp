#include <gtest/gtest.h>
#include "j2_orbit_propagator.h"
#include "constellation_propagator.h"
#include "math_constants.h"
#include <fstream>
#include <iomanip>
#include <vector>

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
    
    std::ofstream csv_file("step_size_accuracy.csv");
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
    std::cout << "\n结果已保存到 step_size_accuracy.csv\n";
}

TEST_F(ParameterSweepTest, EccentricityEffectAnalysis) {
    std::vector<double> eccentricities = {0.0, 0.001, 0.005, 0.01, 0.02, 0.05, 0.1};
    double step_size = 60.0;  // 1分钟步长
    double prop_time = 86164.1;  // 一个恒星日
    
    std::ofstream csv_file("eccentricity_effect.csv");
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
    std::cout << "\n结果已保存到 eccentricity_effect.csv\n";
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