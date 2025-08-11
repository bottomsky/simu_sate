#ifndef COMMON_TYPES_H
#define COMMON_TYPES_H

#include <Eigen/Dense>

// 位置速度结构体
struct StateVector {
    Eigen::Vector3d r;  // 位置矢量 (m)
    Eigen::Vector3d v;  // 速度矢量 (m/s)
};

#endif // COMMON_TYPES_H