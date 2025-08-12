#include <gtest/gtest.h>
#include "j2_orbit_propagator.h"
#include "math_constants.h"
#include <fstream>
#include <nlohmann/json.hpp>
#include <iomanip>
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
    double time_interval = 3600.0; // 每次仿真的时间间隔：1小时
    
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