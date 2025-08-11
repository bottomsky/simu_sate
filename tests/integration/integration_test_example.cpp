#include <gtest/gtest.h>
#include "j2_orbit_propagator.h"
#include "constellation_propagator.h"
#include "math_constants.h"

// 集成测试：验证 J2 轨道传播器与星座传播器之间的一致性
class IntegrationTest : public ::testing::Test {
protected:
    void SetUp() override {
        // 地球同步轨道参数 (GEO)
        a = 42164.0e3;  // 半长轴 (m)
        e = 0.001;      // 偏心率
        i = 0.0 * M_PI / 180.0;  // 倾角 (rad)
        omega = 0.0;    // 近地点幅角 (rad)
        Omega = 0.0;    // 升交点赤经 (rad)
        M = 0.0;        // 平近点角 (rad)
        
        stepSize = 60.0;  // 1分钟步长
    }

    double a, e, i, omega, Omega, M;
    double stepSize;
};

TEST_F(IntegrationTest, SingleVsConstellationConsistency) {
    // 使用单卫星传播器
    OrbitalElements elements = {a, e, i, Omega, omega, M, 0.0};
    J2OrbitPropagator singlePropagator(elements);
    singlePropagator.setStepSize(stepSize);
    
    double t1 = 3600.0; // 1小时后
    auto singleElements = singlePropagator.propagate(t1);
    auto singleState = singlePropagator.elementsToState(singleElements);
    
    // 使用星座传播器（单颗卫星）
    ConstellationPropagator constellationPropagator;
    constellationPropagator.setStepSize(stepSize);
    CompactOrbitalElements compactElements = {a, e, i, Omega, omega, M};
    constellationPropagator.addSatellite(compactElements);
    
    constellationPropagator.propagateConstellation(t1);
    auto positions = constellationPropagator.getAllPositions();
    
    // 验证位置一致性（1米精度）
    ASSERT_EQ(positions.cols(), 1);
    const double px = positions(0, 0);
    const double py = positions(1, 0);
    const double pz = positions(2, 0);
    EXPECT_NEAR(px, singleState.r(0), 1.0);
    EXPECT_NEAR(py, singleState.r(1), 1.0);
    EXPECT_NEAR(pz, singleState.r(2), 1.0);
}

TEST_F(IntegrationTest, MultipleOrbitalPropagation) {
    ConstellationPropagator propagator;
    propagator.setStepSize(stepSize);
    
    // 添加3颗不同轨道的卫星
    CompactOrbitalElements geoSat = {a, e, i, Omega, omega, M};
    CompactOrbitalElements meoSat = {7000.0e3, 0.01, 45.0*M_PI/180.0, 0, 0, 0};
    CompactOrbitalElements leoSat = {6700.0e3, 0.0, 98.0*M_PI/180.0, 0, 0, 0};
    
    propagator.addSatellite(geoSat);
    propagator.addSatellite(meoSat);
    propagator.addSatellite(leoSat);
    
    double t = 1800.0; // 30分钟
    propagator.propagateConstellation(t);
    
    auto positions = propagator.getAllPositions();
    ASSERT_EQ(positions.cols(), 3);
    
    // 验证位置合理性（距离地心范围）
    for (int i = 0; i < positions.cols(); ++i) {
        const double r = positions.col(i).norm();
        EXPECT_GT(r, 6.3e6);  // 大于地球半径
        EXPECT_LT(r, 5.0e7);  // 小于50000公里
    }
}

TEST_F(IntegrationTest, LargeTimeStepPropagation) {
    ConstellationPropagator propagator;
    propagator.setStepSize(3600.0);  // 1小时步长
    CompactOrbitalElements geoSat = {a, e, i, Omega, omega, M};
    propagator.addSatellite(geoSat);
    
    // 传播24小时（一个恒星日）
    double oneDay = 86164.1; // 恒星日秒数
    propagator.propagateConstellation(oneDay);
    
    auto positions = propagator.getAllPositions();
    ASSERT_EQ(positions.cols(), 1);
    
    // GEO卫星应该回到接近初始位置（考虑J2摄动的小偏差）
    const double finalR = positions.col(0).norm();
    
    // 验证轨道高度保持稳定（GEO轨道半径，允许J2与积分误差导致的≈±50 km波动）
    EXPECT_NEAR(finalR, a, 50000.0);  // 50km精度
}