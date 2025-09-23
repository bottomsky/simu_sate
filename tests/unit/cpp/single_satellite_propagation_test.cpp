#include <gtest/gtest.h>
#include "j2_orbit_propagator.h"
#include "j2_constellation_propagator.h"
#include "math_defs.h"
#include <fstream>
#include <nlohmann/json.hpp>
#include <iomanip>
#include <cmath>
#include <vector>
#include <string>
#ifdef _WIN32
#include <direct.h>
#else
#include <sys/stat.h>
#endif

using json = nlohmann::json;

// Helper function to write orbital elements to a JSON file
void write_orbit_data_to_json(const std::string& filename, const OrbitalElements& initial_elements, const OrbitalElements& propagated_elements, double step_size) {
    json j;
    j["step_size_seconds"] = step_size;
    j["initial_elements"] = {
        {"epoch", initial_elements.t},
        {"a", initial_elements.a},
        {"e", initial_elements.e},
        {"i", initial_elements.i * RAD_TO_DEG},
        {"O", initial_elements.O * RAD_TO_DEG},
        {"w", initial_elements.w * RAD_TO_DEG},
        {"M", initial_elements.M * RAD_TO_DEG}
    };
    j["propagated_elements"] = {
        {"epoch", propagated_elements.t},
        {"a", propagated_elements.a},
        {"e", propagated_elements.e},
        {"i", propagated_elements.i * RAD_TO_DEG},
        {"O", propagated_elements.O * RAD_TO_DEG},
        {"w", propagated_elements.w * RAD_TO_DEG},
        {"M", propagated_elements.M * RAD_TO_DEG}
    };

    // 确保输出目录存在
    size_t last_slash = filename.find_last_of("\\");
    if (last_slash != std::string::npos) {
        std::string dir = filename.substr(0, last_slash);
#ifdef _WIN32
        _mkdir(dir.c_str());
#else
        mkdir(dir.c_str(), 0755);
#endif
    }

    std::ofstream o(filename);
    o << std::setw(4) << j << std::endl;
}

TEST(SingleSatellitePropagation, PropagateAndRecord) {
    // Initial orbital elements for a test satellite (e.g., LEO)
    OrbitalElements initial_elements;
    initial_elements.t = 0.0; // Epoch time in seconds
    initial_elements.a = 7000000.0; // Semi-major axis in meters
    initial_elements.e = 0.001;      // Eccentricity
    initial_elements.i = 1.0;      // Inclination in radians
    initial_elements.O = 0.5;      // RAAN in radians
    initial_elements.w = 0.2;      // Argument of perigee in radians
    initial_elements.M = 0.0;        // Mean anomaly in radians

    // 定义三种不同的步长配置
    struct StepConfig {
        double step_size;
        std::string step_name;
        std::string filename_suffix;
    };
    
    std::vector<StepConfig> step_configs = {
        {1.0, "1秒", "step_1s"},
        {60.0, "60秒", "step_60s"},
        {3600.0, "1小时", "step_1hour"}
    };
    
    int simulation_count = 1000; // 仿真次数：1000次
    double time_interval = 60.0; // 每次仿真的时间间隔：1小时
    
    // 调试输出
    std::cout << "Debug: simulation_count = " << simulation_count << std::endl;
    std::cout << "Running simulations with " << step_configs.size() << " different step sizes" << std::endl;

    // 为每种步长运行仿真
    for (const auto& config : step_configs) {
        std::cout << "Running simulation with step size: " << config.step_name << " (" << config.step_size << "s)" << std::endl;
        
        // 创建传播器
        J2OrbitPropagator propagator(initial_elements);
        propagator.setStepSize(config.step_size);

        // 存储当前步长的仿真结果
        json simulation_results;
        simulation_results["simulation_parameters"] = {
            {"step_size_seconds", config.step_size},
            {"step_size_name", config.step_name},
            {"simulation_count", simulation_count},
            {"time_interval_seconds", time_interval}
        };
        simulation_results["initial_elements"] = {
            {"epoch", initial_elements.t},
            {"a", initial_elements.a},
            {"e", initial_elements.e},
            {"i", initial_elements.i * RAD_TO_DEG},
            {"O", initial_elements.O * RAD_TO_DEG},
            {"w", initial_elements.w * RAD_TO_DEG},
            {"M", initial_elements.M * RAD_TO_DEG}
        };
        simulation_results["simulation_data"] = json::array();

        // 进行多次仿真
        for (int i = 0; i < simulation_count; i++) {
            double target_time = initial_elements.t + (i + 1) * time_interval;
            OrbitalElements propagated_elements = propagator.propagate(target_time);

            // 记录当前仿真结果
            json current_simulation = {
                {"simulation_index", i + 1},
                {"target_time", target_time},
                {"propagated_elements", {
                    {"epoch", propagated_elements.t},
                    {"a", propagated_elements.a},
                    {"e", propagated_elements.e},
                    {"i", propagated_elements.i * RAD_TO_DEG},
                    {"O", propagated_elements.O * RAD_TO_DEG},
                    {"w", propagated_elements.w * RAD_TO_DEG},
                    {"M", propagated_elements.M * RAD_TO_DEG}
                }}
            };
            simulation_results["simulation_data"].push_back(current_simulation);

            // 基本断言检查传播是否发生
            ASSERT_NE(propagated_elements.M, initial_elements.M);
        }

        // 保存当前步长的仿真结果到独立的 JSON 文件
        std::string data_dir = "d:\\code\\j2-perturbation-orbit-propagator\\tests\\data";
        std::string filename = data_dir + "\\multi_simulation_results_" + config.filename_suffix + ".json";
        
        // 确保输出目录存在
        size_t last_slash = filename.find_last_of("\\");
        if (last_slash != std::string::npos) {
            std::string dir = filename.substr(0, last_slash);
#ifdef _WIN32
            _mkdir(dir.c_str());
#else
            mkdir(dir.c_str(), 0755);
#endif
        }

        std::ofstream o(filename);
        o << std::setw(4) << simulation_results << std::endl;
        
        std::cout << "Simulation completed for " << config.step_name << ", results saved to: " << filename << std::endl;
    }
}

namespace {

double angularDifference(double lhs, double rhs) {
    return normalizeAnglePiToPi(lhs - rhs);
}

std::vector<CompactOrbitalElements> createTestConstellation() {
    return {
        {7000000.0, 0.001, 0.9, 0.5, 0.2, 0.0},
        {15000000.0, 0.02, 0.5, 1.0, 0.8, 1.2},
        {26560000.0, 0.1, 1.2, 2.0, 0.4, 2.4}
    };
}

std::vector<CompactOrbitalElements> propagateWithMode(J2ConstellationPropagator::ComputeMode mode,
                                                      double step_size,
                                                      double duration,
                                                      const std::vector<CompactOrbitalElements>& initial) {
    J2ConstellationPropagator propagator(0.0);
    propagator.setComputeMode(mode);
    propagator.setStepSize(step_size);
    propagator.addSatellites(initial);
    propagator.propagateConstellation(duration);

    std::vector<CompactOrbitalElements> result;
    result.reserve(initial.size());
    for (size_t idx = 0; idx < initial.size(); ++idx) {
        result.push_back(propagator.getSatelliteElements(idx));
    }
    return result;
}

void expectElementsClose(const CompactOrbitalElements& actual,
                         const CompactOrbitalElements& reference,
                         double linear_tol,
                         double angular_tol,
                         double mean_anomaly_tol) {
    EXPECT_NEAR(actual.a, reference.a, linear_tol);
    EXPECT_NEAR(actual.e, reference.e, linear_tol);
    EXPECT_NEAR(actual.i, reference.i, angular_tol);
    EXPECT_NEAR(angularDifference(actual.O, reference.O), 0.0, angular_tol);
    EXPECT_NEAR(angularDifference(actual.w, reference.w), 0.0, angular_tol);
    EXPECT_NEAR(angularDifference(actual.M, reference.M), 0.0, mean_anomaly_tol);
}

std::vector<CompactOrbitalElements> collectElements(const J2ConstellationPropagator& propagator) {
    std::vector<CompactOrbitalElements> out;
    const size_t count = propagator.getSatelliteCount();
    out.reserve(count);
    for (size_t idx = 0; idx < count; ++idx) {
        out.push_back(propagator.getSatelliteElements(idx));
    }
    return out;
}

bool elementsDifferBeyond(const CompactOrbitalElements& lhs,
                          const CompactOrbitalElements& rhs,
                          double linear_tol,
                          double angular_tol) {
    if (std::abs(lhs.a - rhs.a) > linear_tol) return true;
    if (std::abs(lhs.e - rhs.e) > linear_tol) return true;
    if (std::abs(lhs.i - rhs.i) > angular_tol) return true;
    if (std::abs(angularDifference(lhs.O, rhs.O)) > angular_tol) return true;
    if (std::abs(angularDifference(lhs.w, rhs.w)) > angular_tol) return true;
    if (std::abs(angularDifference(lhs.M, rhs.M)) > angular_tol) return true;
    return false;
}

}  // namespace

TEST(ConstellationPropagationConsistency, ModesAgreeAcrossStepsAndDurations) {
    const auto initial_elements = createTestConstellation();
    const std::vector<std::pair<double, double>> scenarios = {
        {1.0, 600.0},
        {10.0, 3600.0},
        {60.0, 86400.0}
    };

    const double linear_tol = 1e-6;
    const double angular_tol = 1e-8;
    const double mean_anomaly_tol = 1e-8;

    const bool cuda_available = J2ConstellationPropagator::isCudaAvailable();

    for (const auto& scenario : scenarios) {
        const double step_size = scenario.first;
        const double duration = scenario.second;

        SCOPED_TRACE(testing::Message() << "step=" << step_size << ", duration=" << duration);

        const auto scalar_results = propagateWithMode(J2ConstellationPropagator::CPU_SCALAR,
                                                      step_size,
                                                      duration,
                                                      initial_elements);
        const auto simd_results = propagateWithMode(J2ConstellationPropagator::CPU_SIMD,
                                                    step_size,
                                                    duration,
                                                    initial_elements);

        std::vector<CompactOrbitalElements> cuda_results;
        if (cuda_available) {
            cuda_results = propagateWithMode(J2ConstellationPropagator::GPU_CUDA,
                                             step_size,
                                             duration,
                                             initial_elements);
        } else {
            cuda_results = scalar_results;
        }

        ASSERT_EQ(scalar_results.size(), simd_results.size());
        ASSERT_EQ(scalar_results.size(), cuda_results.size());

        for (size_t idx = 0; idx < scalar_results.size(); ++idx) {
            expectElementsClose(simd_results[idx], scalar_results[idx],
                                linear_tol, angular_tol, mean_anomaly_tol);
            expectElementsClose(cuda_results[idx], scalar_results[idx],
                                linear_tol, angular_tol, mean_anomaly_tol);
        }

    }
}

TEST(ConstellationPropagatorCopySemantics, DeepCopyIndependence) {
    const auto initial_elements = createTestConstellation();
    const bool cuda_available = J2ConstellationPropagator::isCudaAvailable();
    const auto preferred_mode = cuda_available ? J2ConstellationPropagator::GPU_CUDA
                                               : J2ConstellationPropagator::CPU_SIMD;

    J2ConstellationPropagator original(0.0);
    original.setComputeMode(preferred_mode);
    original.setStepSize(10.0);
    original.addSatellites(initial_elements);
    original.propagateConstellation(120.0);

    const double linear_tol = 1e-6;
    const double angular_tol = 1e-8;
    const double mean_anomaly_tol = 1e-8;

    const auto baseline = collectElements(original);
    ASSERT_EQ(baseline.size(), initial_elements.size());

    J2ConstellationPropagator copied(original);
    const auto copied_elements = collectElements(copied);
    ASSERT_EQ(copied_elements.size(), baseline.size());
    for (size_t idx = 0; idx < baseline.size(); ++idx) {
        expectElementsClose(copied_elements[idx], baseline[idx],
                            linear_tol, angular_tol, mean_anomaly_tol);
    }

    original.propagateConstellation(600.0);
    const auto mutated = collectElements(original);
    bool diverged_from_baseline = false;
    for (size_t idx = 0; idx < baseline.size(); ++idx) {
        if (elementsDifferBeyond(mutated[idx], baseline[idx], 1e-3, 1e-5)) {
            diverged_from_baseline = true;
            break;
        }
    }
    EXPECT_TRUE(diverged_from_baseline);

    const auto copy_after_mutation = collectElements(copied);
    for (size_t idx = 0; idx < baseline.size(); ++idx) {
        expectElementsClose(copy_after_mutation[idx], baseline[idx],
                            linear_tol, angular_tol, mean_anomaly_tol);
        EXPECT_TRUE(elementsDifferBeyond(mutated[idx], copy_after_mutation[idx], 1e-3, 1e-5));
    }

    J2ConstellationPropagator assigned(0.0);
    assigned.setComputeMode(preferred_mode);
    assigned = original;
    const auto assigned_elements = collectElements(assigned);
    ASSERT_EQ(assigned_elements.size(), mutated.size());
    for (size_t idx = 0; idx < mutated.size(); ++idx) {
        expectElementsClose(assigned_elements[idx], mutated[idx],
                            linear_tol, angular_tol, mean_anomaly_tol);
    }

    assigned.propagateConstellation(1200.0);
    const auto assigned_after = collectElements(assigned);
    const auto original_after_assignment = collectElements(original);
    for (size_t idx = 0; idx < mutated.size(); ++idx) {
        expectElementsClose(original_after_assignment[idx], mutated[idx],
                            linear_tol, angular_tol, mean_anomaly_tol);
    }

    bool assigned_changed = false;
    for (size_t idx = 0; idx < assigned_after.size(); ++idx) {
        if (elementsDifferBeyond(assigned_after[idx], original_after_assignment[idx], 1e-3, 1e-5)) {
            assigned_changed = true;
            break;
        }
    }
    EXPECT_TRUE(assigned_changed);
}

TEST(ConstellationPropagatorCopySemantics, MoveTransferOwnership) {
    const auto initial_elements = createTestConstellation();
    const bool cuda_available = J2ConstellationPropagator::isCudaAvailable();
    const auto preferred_mode = cuda_available ? J2ConstellationPropagator::GPU_CUDA
                                               : J2ConstellationPropagator::CPU_SIMD;

    J2ConstellationPropagator original(0.0);
    original.setComputeMode(preferred_mode);
    original.setStepSize(20.0);
    original.addSatellites(initial_elements);
    original.propagateConstellation(300.0);

    const double linear_tol = 1e-6;
    const double angular_tol = 1e-8;
    const double mean_anomaly_tol = 1e-8;

    const auto baseline = collectElements(original);
    ASSERT_EQ(baseline.size(), initial_elements.size());

    J2ConstellationPropagator moved(std::move(original));
    const auto moved_elements = collectElements(moved);
    ASSERT_EQ(moved_elements.size(), baseline.size());
    for (size_t idx = 0; idx < baseline.size(); ++idx) {
        expectElementsClose(moved_elements[idx], baseline[idx],
                            linear_tol, angular_tol, mean_anomaly_tol);
    }

    J2ConstellationPropagator target(0.0);
    target.setComputeMode(preferred_mode);
    target = std::move(moved);
    const auto target_elements = collectElements(target);
    ASSERT_EQ(target_elements.size(), baseline.size());
    for (size_t idx = 0; idx < baseline.size(); ++idx) {
        expectElementsClose(target_elements[idx], baseline[idx],
                            linear_tol, angular_tol, mean_anomaly_tol);
    }

    target.propagateConstellation(900.0);
    const auto target_after = collectElements(target);
    bool target_changed = false;
    for (size_t idx = 0; idx < target_after.size(); ++idx) {
        if (elementsDifferBeyond(target_after[idx], target_elements[idx], 1e-3, 1e-5)) {
            target_changed = true;
            break;
        }
    }
    EXPECT_TRUE(target_changed);
}
