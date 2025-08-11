#ifndef J2_ORBIT_PROPAGATOR_H
#define J2_ORBIT_PROPAGATOR_H

#include <Eigen/Dense>
#include <cmath>
#include <iostream>
#include "math_constants.h"
#include "common_types.h"

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
    
    // 启用/禁用自适应步长
    void setAdaptiveStepSize(bool enable) { adaptive_step_size_ = enable; }
    
    // 设置自适应步长参数
    void setAdaptiveParameters(double tolerance = 1e-6, double min_step = 1.0, double max_step = 300.0) {
        tolerance_ = tolerance;
        min_step_size_ = min_step;
        max_step_size_ = max_step;
    }

private:
    OrbitalElements current_elements_;  // 当前轨道要素
    double step_size_;                  // 积分步长 (s)
    bool adaptive_step_size_ = false;   // 是否启用自适应步长
    double tolerance_ = 1e-6;           // 误差容忍度（用于步长控制）
    double min_step_size_ = 1.0;        // 最小步长
    double max_step_size_ = 300.0;      // 最大步长
    
    // 计算J2摄动引起的轨道要素变化率
    Eigen::VectorXd computeDerivatives(const OrbitalElements& elements);
    
    // 四阶龙格-库塔积分
    OrbitalElements rk4Integrate(const OrbitalElements& elements, double dt);
    
    // 自适应步长误差估计（嵌套RK）
    double estimateLocalError(const OrbitalElements& elements, double dt);
    
    // 计算偏近点角
    double computeEccentricAnomaly(double M, double e);
    
    // 计算真近点角
    double computeTrueAnomaly(double E, double e);
    
    // 角度归一化
    double normalizeAngle(double angle);
};

#endif // J2_ORBIT_PROPAGATOR_H
