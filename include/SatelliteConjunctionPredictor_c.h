#ifndef SATELLITE_CONJUNCTION_PREDICTOR_C_H
#define SATELLITE_CONJUNCTION_PREDICTOR_C_H

#ifdef __cplusplus
extern "C" {
#endif

#include "j2_orbit_propagator_c.h"         // J2Status, CStateVector, J2_API
#include "j2_constellation_propagator_c.h" // J2ConstellationPropagatorHandle

// 事件阶段
typedef enum {
    CONJUNCTION_PHASE_START = 0,
    CONJUNCTION_PHASE_CLOSEST = 1,
    CONJUNCTION_PHASE_END = 2
} CConjunctionPhase;

// C 事件结构（单位：时间秒，距离米）
typedef struct {
    int satellite1_id;
    int satellite2_id;
    CConjunctionPhase phase;
    double time_s;           // 事件时间（秒）
    double miss_distance_m;  // 最近距离（米）
    CStateVector state_i;    // 卫星 i 在事件时刻的 ECI 状态
    CStateVector state_j;    // 卫星 j 在事件时刻的 ECI 状态
} CConjunctionEvent;

// 预测器配置（与 C++ 配置一致，单位：秒/米）
typedef struct {
    double horizon_s;    // 预测窗口（秒）
    double coarse_dt_s;  // 粗采样步长（秒）
    double refine_dt_s;  // 细采样步长（秒）
    double threshold_m;  // 检测距离阈值（米）
    double cell_size_m;  // 网格单元大小（米），通常等于阈值
} CConjunctionPredictorConfig;

// 不透明句柄
typedef struct CConjunctionPredictor CConjunctionPredictor;

// 创建/销毁
J2_API CConjunctionPredictor* j2_conjunction_predictor_create(const CConjunctionPredictorConfig* config);
J2_API void j2_conjunction_predictor_destroy(CConjunctionPredictor* predictor);

// 运行预测（基于星座传播器当前状态）
J2_API J2Status j2_conjunction_predictor_predict(CConjunctionPredictor* predictor,
                                                 J2ConstellationPropagatorHandle propagator);

// 查询事件
J2_API int j2_conjunction_predictor_get_event_count(const CConjunctionPredictor* predictor);
J2_API J2Status j2_conjunction_predictor_get_events(const CConjunctionPredictor* predictor,
                                                    CConjunctionEvent* events,
                                                    int max_count,
                                                    int* actual_count);

// 清理结果
J2_API J2Status j2_conjunction_predictor_clear(CConjunctionPredictor* predictor);

#ifdef __cplusplus
}
#endif

#endif // SATELLITE_CONJUNCTION_PREDICTOR_C_H
