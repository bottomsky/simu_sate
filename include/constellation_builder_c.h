#ifndef CONSTELLATION_BUILDER_C_H
#define CONSTELLATION_BUILDER_C_H

#ifdef __cplusplus
extern "C" {
#endif

#include "j2_orbit_propagator_c.h"           // J2Status, CStateVector, J2_API
#include "j2_constellation_propagator_c.h"   // CCompactOrbitalElements

// Walker-Delta 星座配置（C 兼容，全部使用 SI 单位：长度 m，角度 rad，时间 s）
typedef struct {
    int plane_count;                 // 轨道平面数 (A)
    int satellites_per_plane;        // 每平面卫星数 (B)
    int relative_phasing;            // Walker F 参数
    double altitude_m;               // 高度 (m)
    double inclination_rad;          // 倾角 (rad)
    double eccentricity;             // 偏心率
    double argument_of_perigee_rad;  // 近地点幅角 (rad)
    double mean_anomaly_offset_rad;  // 平近点角全局偏移 (rad)
    double raan_offset_rad;          // 升交点赤经参考偏移 (rad)
    double epoch_seconds;            // 历元时间 (s)
} CWalkerDeltaConfig;

// 计算 Walker-Delta 星座的卫星总数（plane_count * satellites_per_plane）
J2_API int j2_constellation_walker_delta_count(const CWalkerDeltaConfig* config);

// 生成 Walker-Delta 星座的紧凑轨道要素（config 需使用 SI 单位；elements 需预分配 max_count 个元素）
J2_API J2Status j2_constellation_create_walker_delta(
    const CWalkerDeltaConfig* config,
    CCompactOrbitalElements* elements,
    int max_count,
    int* actual_count
);

#ifdef __cplusplus
}
#endif

#endif // CONSTELLATION_BUILDER_C_H
