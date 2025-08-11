#ifndef J2_ORBIT_PROPAGATOR_H
#define J2_ORBIT_PROPAGATOR_H

#include <Eigen/Dense>
#include <cmath>
#include <iostream>

// 物理常数定义
const double MU = 3.986004418e14;       // 地球引力常数 (m^3/s^2)
const double RE = 6378137.0;           // 地球平均半径 (m)
const double J2 = 1.08263e-3;          // 地球二阶带谐系数
const double EPSILON = 1e-12;          // 数值计算精度

// 轨道要素结构体
struct OrbitalElements {
    double a;   // 半长轴 (m)
    double e;   // 偏心率
    double i;   // 倾角 (rad)
    double O;   // 升交点赤经 (rad)
    double w;   // 近地点幅角 (rad)
    double M;   // 平近点角 (rad)
    double t;   // 历元时间 (s)
};

// 位置速度结构体
struct StateVector {
    Eigen::Vector3d r;  // 位置矢量 (m)
    Eigen::Vector3d v;  // 速度矢量 (m/s)
};

class J2OrbitPropagator {
public:
    // 构造函数
    J2OrbitPropagator(const OrbitalElements& initial_elements);
    
    // 析构函数
    ~J2OrbitPropagator() = default;
    
    // 外推轨道到指定时间
    OrbitalElements propagate(double t);
    
    // 从轨道要素计算位置速度
    StateVector elementsToState(const OrbitalElements& elements);
    
    // 从位置速度计算轨道要素
    OrbitalElements stateToElements(const StateVector& state, double t);
    
    // 设置积分步长
    void setStepSize(double step) { step_size_ = step; }
    
    // 获取当前积分步长
    double getStepSize() const { return step_size_; }

private:
    OrbitalElements current_elements_;  // 当前轨道要素
    double step_size_;                  // 积分步长 (s)
    
    // 计算J2摄动引起的轨道要素变化率
    Eigen::VectorXd computeDerivatives(const OrbitalElements& elements);
    
    // 四阶龙格-库塔积分
    OrbitalElements rk4Integrate(const OrbitalElements& elements, double dt);
    
    // 计算偏近点角
    double computeEccentricAnomaly(double M, double e);
    
    // 计算真近点角
    double computeTrueAnomaly(double E, double e);
    
    // 角度归一化到[0, 2π)
    double normalizeAngle(double angle);
};

#endif // J2_ORBIT_PROPAGATOR_H
