#include "constellation_propagator.h"
#include "math_defs.h"
#include <chrono>
#include <iomanip>
#include <random>

void generateRandomConstellation(std::vector<CompactOrbitalElements>& satellites, size_t count) {
    std::random_device rd;
    std::mt19937 gen(rd());
    
    // LEO卫星参数范围
    std::uniform_real_distribution<double> alt_dist(400e3, 1200e3);  // 高度 400-1200km
    std::uniform_real_distribution<double> inc_dist(0.0, M_PI);      // 倾角 0-180度
    std::uniform_real_distribution<double> angle_dist(0.0, 2*M_PI);  // 角度 0-360度
    std::uniform_real_distribution<double> ecc_dist(0.0001, 0.02);   // 偏心率 0.0001-0.02
    
    satellites.reserve(count);
    
    for (size_t i = 0; i < count; ++i) {
        CompactOrbitalElements elem;
        elem.a = RE + alt_dist(gen);      // 半长轴
        elem.e = ecc_dist(gen);           // 偏心率
        elem.i = inc_dist(gen);           // 倾角
        elem.O = angle_dist(gen);         // 升交点赤经
        elem.w = angle_dist(gen);         // 近地点幅角
        elem.M = angle_dist(gen);         // 平近点角
        
        satellites.push_back(elem);
    }
}

void benchmarkPropagation(ConstellationPropagator& propagator, 
                         const std::string& mode_name, 
                         double propagation_time) {
    auto start = std::chrono::high_resolution_clock::now();
    
    propagator.propagateConstellation(propagation_time);
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    double elapsed_ms = duration.count() / 1000.0;
    size_t sat_count = propagator.getSatelliteCount();
    double sats_per_ms = sat_count / elapsed_ms;
    
    std::cout << std::fixed << std::setprecision(2);
    std::cout << mode_name << " 模式:" << std::endl;
    std::cout << "  卫星数量: " << sat_count << std::endl;
    std::cout << "  计算时间: " << elapsed_ms << " ms" << std::endl;
    std::cout << "  处理速度: " << sats_per_ms << " sats/ms" << std::endl;
    std::cout << "  每颗卫星: " << elapsed_ms/sat_count << " ms/sat" << std::endl;
    std::cout << std::endl;
}

void printMemoryUsage(const ConstellationPropagator& propagator) {
    size_t sat_count = propagator.getSatelliteCount();
    
    // 计算内存使用量
    size_t compact_size = sizeof(CompactOrbitalElements) * sat_count;
    size_t original_size = sizeof(double) * 7 * sat_count;  // 原始结构包含历元时间
    size_t simd_overhead = sizeof(std::vector<double>) * 6;  // vector开销
    
    std::cout << "内存使用分析:" << std::endl;
    std::cout << "  卫星数量: " << sat_count << std::endl;
    std::cout << "  优化前 (含历元时间): " << original_size/1024.0 << " KB" << std::endl;
    std::cout << "  优化后 (无历元时间): " << compact_size/1024.0 << " KB" << std::endl;
    std::cout << "  节省内存: " << (original_size - compact_size)/1024.0 << " KB" << std::endl;
    std::cout << "  节省比例: " << (1.0 - (double)compact_size/original_size)*100 << "%" << std::endl;
    std::cout << std::endl;
}

int main() {
    std::cout << "=== J2轨道外推器大规模星座演示 ===" << std::endl;
    std::cout << std::endl;
    
    // CUDA 运行检测提示
    bool cudaAvailable = ConstellationPropagator::isCudaAvailable();
    std::cout << "CUDA 可用性检测: " << (cudaAvailable ? "可用" : "不可用") << std::endl;
    if (cudaAvailable) {
        std::cout << "提示: 创建的 ConstellationPropagator 将自动优先使用 CUDA 加速计算" << std::endl;
    } else {
        std::cout << "提示: 将使用 CPU SIMD 模式进行计算" << std::endl;
    }
    std::cout << std::endl;
    
    // 测试不同规模的星座
    std::vector<size_t> test_sizes = {1000, 5000, 10000, 20000};
    double propagation_time = 3600.0;  // 外推1小时
    
    for (size_t sat_count : test_sizes) {
        std::cout << "=== 测试 " << sat_count << " 颗卫星 ===" << std::endl;
        
        // 生成随机星座
        std::vector<CompactOrbitalElements> satellites;
        generateRandomConstellation(satellites, sat_count);
        
        // 创建传播器并添加卫星（会自动检测并使用最佳计算模式）
        ConstellationPropagator propagator(0.0);
        propagator.addSatellites(satellites);
        propagator.setStepSize(60.0);  // 60秒步长
        
        // 显示自动选择的计算模式
        std::cout << "自动选择的计算模式: " << (cudaAvailable ? "GPU CUDA (自动启用)" : "CPU SIMD (默认)") << std::endl;
        std::cout << std::endl;
        
        // 打印内存使用情况
        printMemoryUsage(propagator);
        
        // 测试不同计算模式的性能
        std::cout << "性能测试 (外推时间: " << propagation_time << "s):" << std::endl;
        
        // CPU标量模式
        ConstellationPropagator prop_scalar = propagator;
        prop_scalar.setComputeMode(ConstellationPropagator::CPU_SCALAR);
        benchmarkPropagation(prop_scalar, "CPU标量", propagation_time);
        
        // CPU SIMD模式
        ConstellationPropagator prop_simd = propagator;
        prop_simd.setComputeMode(ConstellationPropagator::CPU_SIMD);
        benchmarkPropagation(prop_simd, "CPU SIMD", propagation_time);
        
        // CUDA模式 (如果可用)
        if (ConstellationPropagator::isCudaAvailable()) {
            ConstellationPropagator prop_cuda = propagator;
            prop_cuda.setComputeMode(ConstellationPropagator::GPU_CUDA);
            benchmarkPropagation(prop_cuda, "GPU CUDA", propagation_time);
        } else {
            std::cout << "CUDA 不可用，跳过GPU测试" << std::endl;
        }
        
        // 验证结果一致性 (抽样检查)
        if (sat_count <= 1000) {
            std::cout << "结果验证 (前5颗卫星位置):" << std::endl;
            for (size_t i = 0; i < std::min(size_t(5), sat_count); ++i) {
                StateVector state = prop_simd.getSatelliteState(i);
                std::cout << "  卫星 " << i << ": (" 
                         << std::scientific << std::setprecision(3)
                         << state.r[0] << ", " << state.r[1] << ", " << state.r[2] 
                         << ") m" << std::endl;
            }
        }
        
        std::cout << std::string(50, '-') << std::endl;
    }
    
    return 0;
}