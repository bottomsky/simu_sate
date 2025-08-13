#include <gtest/gtest.h>
#include "constellation_propagator.h"

class SIMDConsistencyTest : public ::testing::Test {
protected:
    void SetUp() override {
        // 地球同步轨道参数
        elements_.a = 42164.169; // km
        elements_.e = 0.001;
        elements_.i = 0.017453; // 1度转弧度
        elements_.O = 0.0;
        elements_.w = 0.0;
        elements_.M = 0.0;
    }

    CompactOrbitalElements elements_;
};

// 测试SCALAR vs SIMD模式的一致性
TEST_F(SIMDConsistencyTest, ScalarVsSIMDConsistency) {
    double propagation_time = 3600.0; // 1小时
    
    // CPU_SCALAR模式
    ConstellationPropagator prop_scalar(0.0);
    prop_scalar.setComputeMode(ConstellationPropagator::CPU_SCALAR);
    prop_scalar.addSatellite(elements_);
    prop_scalar.propagateConstellation(propagation_time);
    StateVector state_scalar = prop_scalar.getSatelliteState(0);
    
    // CPU_SIMD模式
    ConstellationPropagator prop_simd(0.0);
    prop_simd.setComputeMode(ConstellationPropagator::CPU_SIMD);
    prop_simd.addSatellite(elements_);
    prop_simd.propagateConstellation(propagation_time);
    StateVector state_simd = prop_simd.getSatelliteState(0);
    
    // 验证位置一致性 (误差应该很小)
    double pos_diff_x = std::abs(state_scalar.r(0) - state_simd.r(0));
    double pos_diff_y = std::abs(state_scalar.r(1) - state_simd.r(1)); 
    double pos_diff_z = std::abs(state_scalar.r(2) - state_simd.r(2));
    
    EXPECT_LT(pos_diff_x, 1.0) << "X position difference too large: " << pos_diff_x;
    EXPECT_LT(pos_diff_y, 1.0) << "Y position difference too large: " << pos_diff_y;
    EXPECT_LT(pos_diff_z, 1.0) << "Z position difference too large: " << pos_diff_z;
    
    // 验证速度一致性
    double vel_diff_x = std::abs(state_scalar.v(0) - state_simd.v(0));
    double vel_diff_y = std::abs(state_scalar.v(1) - state_simd.v(1));
    double vel_diff_z = std::abs(state_scalar.v(2) - state_simd.v(2));
    
    EXPECT_LT(vel_diff_x, 0.001) << "X velocity difference too large: " << vel_diff_x;
    EXPECT_LT(vel_diff_y, 0.001) << "Y velocity difference too large: " << vel_diff_y;
    EXPECT_LT(vel_diff_z, 0.001) << "Z velocity difference too large: " << vel_diff_z;
}

// 测试多卫星场景下的一致性
TEST_F(SIMDConsistencyTest, MultiSatelliteConsistency) {
    double propagation_time = 1800.0; // 30分钟
    size_t num_satellites = 8; // 确保有SIMD批处理和标量尾部
    
    std::vector<CompactOrbitalElements> test_elements;
    for (size_t i = 0; i < num_satellites; ++i) {
        CompactOrbitalElements elem = elements_;
        elem.M = i * 0.1; // 稍微不同的平近点角
        test_elements.push_back(elem);
    }
    
    // SCALAR模式
    ConstellationPropagator prop_scalar(0.0);
    prop_scalar.setComputeMode(ConstellationPropagator::CPU_SCALAR);
    for (const auto& elem : test_elements) {
        prop_scalar.addSatellite(elem);
    }
    prop_scalar.propagateConstellation(propagation_time);
    
    // SIMD模式
    ConstellationPropagator prop_simd(0.0);
    prop_simd.setComputeMode(ConstellationPropagator::CPU_SIMD);
    for (const auto& elem : test_elements) {
        prop_simd.addSatellite(elem);
    }
    prop_simd.propagateConstellation(propagation_time);
    
    // 比较每颗卫星的状态
    for (size_t i = 0; i < num_satellites; ++i) {
        StateVector state_scalar = prop_scalar.getSatelliteState(i);
        StateVector state_simd = prop_simd.getSatelliteState(i);
        
        double pos_diff = (state_scalar.r - state_simd.r).norm();
        double vel_diff = (state_scalar.v - state_simd.v).norm();
        
        EXPECT_LT(pos_diff, 1.0) << "Satellite " << i << " position difference: " << pos_diff;
        EXPECT_LT(vel_diff, 0.001) << "Satellite " << i << " velocity difference: " << vel_diff;
    }
}