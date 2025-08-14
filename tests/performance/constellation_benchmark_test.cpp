#include <gtest/gtest.h>
#include <chrono>
#include <iostream>
#include <vector>
#include <iomanip>
#include "constellation_propagator.h"

class ConstellationBenchmarkTest : public ::testing::Test {
protected:
    void SetUp() override {
        // 设置通用的轨道参数
        reference_elements_ = {
            7378137.0,    // 半长轴 (LEO轨道, ~1000km)
            0.001,        // 偏心率 (近圆轨道)
            0.8727,       // 倾角 (50度)
            0.0,          // 升交点赤经
            0.0,          // 近地点幅角  
            0.0           // 平近点角
        };
        
        // 性能测试参数
        propagation_time_ = 10.0;  // 10秒
        frames_ = 1000;            // 1000帧
        step_size_ = 1.0;          // 1秒步长
    }
    
    // 生成指定数量的卫星星座
    std::vector<CompactOrbitalElements> generateConstellation(size_t satellite_count) {
        std::vector<CompactOrbitalElements> constellation;
        constellation.reserve(satellite_count);
        
        for (size_t i = 0; i < satellite_count; ++i) {
            CompactOrbitalElements elem = reference_elements_;
            
            // 为每个卫星添加轻微变化以模拟真实星座
            elem.O += (2.0 * M_PI * i) / satellite_count;  // 升交点均匀分布
            elem.M += (M_PI * i) / satellite_count;         // 平近点角偏移
            elem.a += 1000.0 * (i % 10 - 5);               // 轻微高度变化
            
            constellation.push_back(elem);
        }
        
        return constellation;
    }
    
    // 执行性能基准测试
    double runBenchmark(size_t satellite_count, 
                       ConstellationPropagator::ComputeMode mode,
                       const std::string& mode_name) {
        
        auto constellation = generateConstellation(satellite_count);
        ConstellationPropagator propagator(0.0);
        
        // 设置计算模式和参数
        propagator.setComputeMode(mode);
        propagator.setStepSize(step_size_);
        propagator.addSatellites(constellation);
        
        std::cout << "\n[" << mode_name << "] 测试 " << satellite_count 
                  << " 个卫星，" << frames_ << " 帧..." << std::flush;
        
        // 开始计时
        auto start_time = std::chrono::high_resolution_clock::now();
        
        // 执行指定帧数的传播
        for (int frame = 0; frame < frames_; ++frame) {
            double target_time = frame * propagation_time_;
            propagator.propagateConstellation(target_time);
        }
        
        // 结束计时
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
            end_time - start_time).count();
        
        double total_seconds = duration / 1000000.0;
        double frames_per_second = frames_ / total_seconds;
        
        std::cout << " 完成\n";
        std::cout << "  总耗时: " << std::fixed << std::setprecision(3) 
                  << total_seconds << " 秒\n";
        std::cout << "  平均帧率: " << std::fixed << std::setprecision(1) 
                  << frames_per_second << " FPS\n";
        std::cout << "  每帧耗时: " << std::fixed << std::setprecision(3) 
                  << (total_seconds * 1000.0 / frames_) << " ms\n";
        
        return total_seconds;
    }
    
    CompactOrbitalElements reference_elements_;
    double propagation_time_;
    int frames_;
    double step_size_;
};

// 200卫星性能测试
TEST_F(ConstellationBenchmarkTest, Satellites200_Performance) {
    const size_t satellite_count = 200;
    
    std::cout << "\n=== 200卫星性能基准测试 ===";
    
    // CPU标量模式
    double scalar_time = runBenchmark(satellite_count, 
                                     ConstellationPropagator::CPU_SCALAR, 
                                     "CPU标量");
    
    // CPU SIMD模式  
    double simd_time = runBenchmark(satellite_count,
                                   ConstellationPropagator::CPU_SIMD,
                                   "CPU SIMD");
    
    // GPU CUDA模式 (如果可用)
    double cuda_time = 0.0;
    if (ConstellationPropagator::isCudaAvailable()) {
        cuda_time = runBenchmark(satellite_count,
                                ConstellationPropagator::GPU_CUDA,
                                "GPU CUDA");
    } else {
        std::cout << "\n[GPU CUDA] CUDA不可用，跳过测试\n";
    }
    
    // 输出性能对比
    std::cout << "\n--- 200卫星性能总结 ---\n";
    std::cout << "CPU标量: " << scalar_time << " 秒\n";
    std::cout << "CPU SIMD: " << simd_time << " 秒 (加速比: " 
              << std::fixed << std::setprecision(2) << scalar_time/simd_time << "x)\n";
    if (cuda_time > 0) {
        std::cout << "GPU CUDA: " << cuda_time << " 秒 (加速比: " 
                  << std::fixed << std::setprecision(2) << scalar_time/cuda_time << "x)\n";
    }
    
    // 基本验证：所有模式都应该在合理时间内完成
    EXPECT_LT(scalar_time, 60.0);  // 标量模式应在60秒内完成
    EXPECT_LT(simd_time, scalar_time);  // SIMD应比标量更快
}

// 1000卫星性能测试
TEST_F(ConstellationBenchmarkTest, Satellites1000_Performance) {
    const size_t satellite_count = 1000;
    
    std::cout << "\n=== 1000卫星性能基准测试 ===";
    
    // CPU标量模式
    double scalar_time = runBenchmark(satellite_count, 
                                     ConstellationPropagator::CPU_SCALAR, 
                                     "CPU标量");
    
    // CPU SIMD模式  
    double simd_time = runBenchmark(satellite_count,
                                   ConstellationPropagator::CPU_SIMD,
                                   "CPU SIMD");
    
    // GPU CUDA模式 (如果可用)
    double cuda_time = 0.0;
    if (ConstellationPropagator::isCudaAvailable()) {
        cuda_time = runBenchmark(satellite_count,
                                ConstellationPropagator::GPU_CUDA,
                                "GPU CUDA");
    } else {
        std::cout << "\n[GPU CUDA] CUDA不可用，跳过测试\n";
    }
    
    // 输出性能对比
    std::cout << "\n--- 1000卫星性能总结 ---\n";
    std::cout << "CPU标量: " << scalar_time << " 秒\n";
    std::cout << "CPU SIMD: " << simd_time << " 秒 (加速比: " 
              << std::fixed << std::setprecision(2) << scalar_time/simd_time << "x)\n";
    if (cuda_time > 0) {
        std::cout << "GPU CUDA: " << cuda_time << " 秒 (加速比: " 
                  << std::fixed << std::setprecision(2) << scalar_time/cuda_time << "x)\n";
    }
    
    // 基本验证
    EXPECT_LT(scalar_time, 300.0);  // 标量模式应在5分钟内完成
    EXPECT_LT(simd_time, scalar_time);  // SIMD应比标量更快
}

// 5000卫星性能测试
TEST_F(ConstellationBenchmarkTest, Satellites5000_Performance) {
    const size_t satellite_count = 5000;
    
    std::cout << "\n=== 5000卫星性能基准测试 ===";
    
    // CPU标量模式
    double scalar_time = runBenchmark(satellite_count, 
                                     ConstellationPropagator::CPU_SCALAR, 
                                     "CPU标量");
    
    // CPU SIMD模式  
    double simd_time = runBenchmark(satellite_count,
                                   ConstellationPropagator::CPU_SIMD,
                                   "CPU SIMD");
    
    // GPU CUDA模式 (如果可用)
    double cuda_time = 0.0;
    if (ConstellationPropagator::isCudaAvailable()) {
        cuda_time = runBenchmark(satellite_count,
                                ConstellationPropagator::GPU_CUDA,
                                "GPU CUDA");
    } else {
        std::cout << "\n[GPU CUDA] CUDA不可用，跳过测试\n";
    }
    
    // 输出性能对比
    std::cout << "\n--- 5000卫星性能总结 ---\n";
    std::cout << "CPU标量: " << scalar_time << " 秒\n";
    std::cout << "CPU SIMD: " << simd_time << " 秒 (加速比: " 
              << std::fixed << std::setprecision(2) << scalar_time/simd_time << "x)\n";
    if (cuda_time > 0) {
        std::cout << "GPU CUDA: " << cuda_time << " 秒 (加速比: " 
                  << std::fixed << std::setprecision(2) << scalar_time/cuda_time << "x)\n";
    }
    
    // 基本验证
    EXPECT_LT(scalar_time, 1500.0);  // 标量模式应在25分钟内完成
    EXPECT_LT(simd_time, scalar_time);  // SIMD应比标量更快
}

// 新增：10000卫星性能测试
TEST_F(ConstellationBenchmarkTest, Satellites10000_Performance) {
    const size_t satellite_count = 10000;

    std::cout << "\n=== 10000卫星性能基准测试 ===";

    double scalar_time = runBenchmark(satellite_count,
                                     ConstellationPropagator::CPU_SCALAR,
                                     "CPU标量");

    double simd_time = runBenchmark(satellite_count,
                                   ConstellationPropagator::CPU_SIMD,
                                   "CPU SIMD");

    double cuda_time = 0.0;
    if (ConstellationPropagator::isCudaAvailable()) {
        cuda_time = runBenchmark(satellite_count,
                                ConstellationPropagator::GPU_CUDA,
                                "GPU CUDA");
    } else {
        std::cout << "\n[GPU CUDA] CUDA不可用，跳过测试\n";
    }

    std::cout << "\n--- 10000卫星性能总结 ---\n";
    std::cout << "CPU标量: " << scalar_time << " 秒\n";
    std::cout << "CPU SIMD: " << simd_time << " 秒 (加速比: "
              << std::fixed << std::setprecision(2) << scalar_time / simd_time << "x)\n";
    if (cuda_time > 0) {
        std::cout << "GPU CUDA: " << cuda_time << " 秒 (加速比: "
                  << std::fixed << std::setprecision(2) << scalar_time / cuda_time << "x)\n";
    }

    EXPECT_LT(simd_time, scalar_time);
    EXPECT_LT(scalar_time, 3600.0); // 宽松上限
}

// 新增：20000卫星性能测试
TEST_F(ConstellationBenchmarkTest, Satellites20000_Performance) {
    const size_t satellite_count = 20000;

    std::cout << "\n=== 20000卫星性能基准测试 ===";

    double scalar_time = runBenchmark(satellite_count,
                                     ConstellationPropagator::CPU_SCALAR,
                                     "CPU标量");

    double simd_time = runBenchmark(satellite_count,
                                   ConstellationPropagator::CPU_SIMD,
                                   "CPU SIMD");

    double cuda_time = 0.0;
    if (ConstellationPropagator::isCudaAvailable()) {
        cuda_time = runBenchmark(satellite_count,
                                ConstellationPropagator::GPU_CUDA,
                                "GPU CUDA");
    } else {
        std::cout << "\n[GPU CUDA] CUDA不可用，跳过测试\n";
    }

    std::cout << "\n--- 20000卫星性能总结 ---\n";
    std::cout << "CPU标量: " << scalar_time << " 秒\n";
    std::cout << "CPU SIMD: " << simd_time << " 秒 (加速比: "
              << std::fixed << std::setprecision(2) << scalar_time / simd_time << "x)\n";
    if (cuda_time > 0) {
        std::cout << "GPU CUDA: " << cuda_time << " 秒 (加速比: "
                  << std::fixed << std::setprecision(2) << scalar_time / cuda_time << "x)\n";
    }

    EXPECT_LT(simd_time, scalar_time);
    EXPECT_LT(scalar_time, 3600.0); // 宽松上限
}

// 综合性能对比测试
TEST_F(ConstellationBenchmarkTest, Scalability_Analysis) {
    std::vector<size_t> satellite_counts = {200, 1000, 5000, 10000, 20000};
    
    std::cout << "\n=== 可扩展性分析 ===\n";
    std::cout << std::setw(12) << "卫星数量" 
              << std::setw(12) << "CPU标量(s)" 
              << std::setw(12) << "CPU SIMD(s)"
              << std::setw(12) << "GPU CUDA(s)" << "\n";
    std::cout << std::string(48, '-') << "\n";

    for (size_t count : satellite_counts) {
        auto constellation = generateConstellation(count);
        
        // 短时间测试以快速评估可扩展性
        int short_frames = 100;  // 减少帧数以加快测试
        ConstellationPropagator prop_scalar(0.0), prop_simd(0.0), prop_cuda(0.0);
        
        prop_scalar.setComputeMode(ConstellationPropagator::CPU_SCALAR);
        prop_scalar.setStepSize(step_size_);
        prop_scalar.addSatellites(constellation);
        
        prop_simd.setComputeMode(ConstellationPropagator::CPU_SIMD);
        prop_simd.setStepSize(step_size_);
        prop_simd.addSatellites(constellation);
        
        // 测量CPU标量性能
        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < short_frames; ++i) {
            prop_scalar.propagateConstellation(i * propagation_time_);
        }
        auto end = std::chrono::high_resolution_clock::now();
        double scalar_time = std::chrono::duration_cast<std::chrono::microseconds>(
            end - start).count() / 1000000.0;
        
        // 测量CPU SIMD性能  
        start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < short_frames; ++i) {
            prop_simd.propagateConstellation(i * propagation_time_);
        }
        end = std::chrono::high_resolution_clock::now();
        double simd_time = std::chrono::duration_cast<std::chrono::microseconds>(
            end - start).count() / 1000000.0;
        
        // 测量GPU CUDA性能（如可用）
        double cuda_time = -1.0;
        if (ConstellationPropagator::isCudaAvailable()) {
            ConstellationPropagator prop_cuda_run(0.0);
            prop_cuda_run.setComputeMode(ConstellationPropagator::GPU_CUDA);
            prop_cuda_run.setStepSize(step_size_);
            prop_cuda_run.addSatellites(constellation);
            
            start = std::chrono::high_resolution_clock::now();
            for (int i = 0; i < short_frames; ++i) {
                prop_cuda_run.propagateConstellation(i * propagation_time_);
            }
            end = std::chrono::high_resolution_clock::now();
            cuda_time = std::chrono::duration_cast<std::chrono::microseconds>(
                end - start).count() / 1000000.0;
        } else {
            std::cerr << "CUDA not available, falling back to SIMD" << std::endl;
        }
        
        std::cout << std::setw(12) << count
                  << std::setw(12) << std::fixed << std::setprecision(3) << scalar_time
                  << std::setw(12) << std::fixed << std::setprecision(3) << simd_time
                  << std::setw(12) << std::fixed << std::setprecision(3) << (cuda_time < 0 ? simd_time : cuda_time)
                  << "\n";
    }
    std::cout << "\n注：GPU CUDA显示-1表示CUDA不可用\n";
}
    
