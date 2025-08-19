#ifndef CONSTELLATION_PROPAGATOR_C_H
#define CONSTELLATION_PROPAGATOR_C_H

#include <stddef.h>  // 确保 size_t 类型定义正确

#ifdef __cplusplus
extern "C" {
#endif

// 导出宏定义（与j2_orbit_propagator_c.h保持一致）
#ifndef J2_API
  #if defined(_WIN32) || defined(_WIN64)
    #if defined(J2_BUILD_STATIC)
      // 静态库不需要导入/导出修饰
      #define J2_API
    #elif defined(J2_BUILD_DLL)
      #define J2_API __declspec(dllexport)
    #else
      #define J2_API __declspec(dllimport)
    #endif
  #else
    // 非 Windows 平台：静态库与共享库统一使用默认可见性
    #define J2_API __attribute__((visibility("default")))
  #endif
#endif

// C格式的压紧轨道要素结构体（不包含历元时间）
typedef struct {
    double a;   // 半长轴 (m)
    double e;   // 偏心率
    double i;   // 倾角 (rad)
    double O;   // 升交点赤经 (rad)
    double w;   // 近地点幅角 (rad)
    double M;   // 平近点角 (rad)
} CCompactOrbitalElements;

// 包含共用的C状态向量定义
#include "j2_orbit_propagator_c.h"

// 计算模式枚举
typedef enum {
    COMPUTE_MODE_CPU_SCALAR = 0,
    COMPUTE_MODE_CPU_SIMD = 1,
    COMPUTE_MODE_GPU_CUDA = 2
} ComputeMode;

// 不透明指针，用于隐藏ConstellationPropagator类的实现细节
typedef void* ConstellationPropagatorHandle;

// === 创建和销毁函数 ===

/**
 * @brief 创建星座传播器实例
 * @param epoch_time 星座统一历元时间 (s)
 * @return 传播器句柄，失败时返回NULL
 */
J2_API ConstellationPropagatorHandle constellation_propagator_create(double epoch_time);

/**
 * @brief 销毁星座传播器实例
 * @param handle 传播器句柄
 */
J2_API void constellation_propagator_destroy(ConstellationPropagatorHandle handle);

// === 卫星管理函数 ===

/**
 * @brief 批量添加卫星到星座
 * @param handle 传播器句柄
 * @param satellites 卫星轨道要素数组
 * @param count 卫星数量
 * @return 0表示成功，非0表示失败
 */
J2_API int constellation_propagator_add_satellites(ConstellationPropagatorHandle handle, 
                                                  const CCompactOrbitalElements* satellites, 
                                                  size_t count);

/**
 * @brief 添加单个卫星到星座
 * @param handle 传播器句柄
 * @param satellite 卫星轨道要素
 * @return 0表示成功，非0表示失败
 */
J2_API int constellation_propagator_add_satellite(ConstellationPropagatorHandle handle, 
                                                 const CCompactOrbitalElements* satellite);

/**
 * @brief 获取星座中的卫星数量
 * @param handle 传播器句柄
 * @param count 输出的卫星数量
 * @return 0表示成功，非0表示失败
 */
J2_API int constellation_propagator_get_satellite_count(ConstellationPropagatorHandle handle, 
                                                       size_t* count);

// === 轨道传播函数 ===

/**
 * @brief 将整个星座传播到指定时间
 * @param handle 传播器句柄
 * @param target_time 目标时间 (s)
 * @return 0表示成功，非0表示失败
 */
J2_API int constellation_propagator_propagate(ConstellationPropagatorHandle handle, 
                                             double target_time);

/**
 * @brief 获取指定卫星的当前轨道要素
 * @param handle 传播器句柄
 * @param satellite_id 卫星索引
 * @param elements 输出的轨道要素
 * @return 0表示成功，非0表示失败
 */
J2_API int constellation_propagator_get_satellite_elements(ConstellationPropagatorHandle handle, 
                                                          size_t satellite_id, 
                                                          CCompactOrbitalElements* elements);

/**
 * @brief 获取指定卫星的当前状态向量
 * @param handle 传播器句柄
 * @param satellite_id 卫星索引
 * @param state 输出的状态向量
 * @return 0表示成功，非0表示失败
 */
J2_API int constellation_propagator_get_satellite_state(ConstellationPropagatorHandle handle, 
                                                       size_t satellite_id, 
                                                       CStateVector* state);

/**
 * @brief 获取所有卫星的位置 (平铺为一维数组)
 * @param handle 传播器句柄
 * @param positions 输出的位置数组[x1,y1,z1,x2,y2,z2,...]，需要预分配 3*count 个元素
 * @param count 输入：数组容量(卫星数量)，输出：实际填充的卫星数量
 * @return 0表示成功，非0表示失败
 */
J2_API int constellation_propagator_get_all_positions(ConstellationPropagatorHandle handle, 
                                                     double* positions, 
                                                     size_t* count);

// === 脉冲施加函数 ===

/**
 * @brief 对整个星座施加脉冲
 * @param handle 传播器句柄
 * @param delta_vs 速度增量数组[dvx1,dvy1,dvz1,dvx2,dvy2,dvz2,...]，需要 3*count 个元素
 * @param count 卫星数量
 * @param impulse_time 脉冲施加时间 (s)
 * @return 0表示成功，非0表示失败
 */
J2_API int constellation_propagator_apply_impulse_to_constellation(ConstellationPropagatorHandle handle, 
                                                                  const double* delta_vs, 
                                                                  size_t count, 
                                                                  double impulse_time);

/**
 * @brief 对指定卫星子集施加脉冲
 * @param handle 传播器句柄
 * @param satellite_ids 卫星索引数组
 * @param delta_vs 速度增量数组[dvx1,dvy1,dvz1,dvx2,dvy2,dvz2,...]
 * @param count 卫星数量（satellite_ids和delta_vs数组的长度）
 * @param impulse_time 脉冲施加时间 (s)
 * @return 0表示成功，非0表示失败
 */
J2_API int constellation_propagator_apply_impulse_to_satellites(ConstellationPropagatorHandle handle, 
                                                               const size_t* satellite_ids, 
                                                               const double* delta_vs, 
                                                               size_t count, 
                                                               double impulse_time);

// === 参数设置函数 ===

/**
 * @brief 设置积分步长
 * @param handle 传播器句柄
 * @param step_size 步长 (s)
 * @return 0表示成功，非0表示失败
 */
J2_API int constellation_propagator_set_step_size(ConstellationPropagatorHandle handle, 
                                                 double step_size);

/**
 * @brief 设置计算模式
 * @param handle 传播器句柄
 * @param mode 计算模式
 * @return 0表示成功，非0表示失败
 */
J2_API int constellation_propagator_set_compute_mode(ConstellationPropagatorHandle handle, 
                                                    ComputeMode mode);

/**
 * @brief 启用或禁用自适应步长
 * @param handle 传播器句柄
 * @param enable 1启用，0禁用
 * @return 0表示成功，非0表示失败
 */
J2_API int constellation_propagator_set_adaptive_step_size(ConstellationPropagatorHandle handle, 
                                                          int enable);

/**
 * @brief 设置自适应步长参数
 * @param handle 传播器句柄
 * @param tolerance 误差容忍度
 * @param min_step 最小步长 (s)
 * @param max_step 最大步长 (s)
 * @return 0表示成功，非0表示失败
 */
J2_API int constellation_propagator_set_adaptive_parameters(ConstellationPropagatorHandle handle, 
                                                           double tolerance, 
                                                           double min_step, 
                                                           double max_step);

/**
 * @brief 检查CUDA可用性
 * @return 1表示CUDA可用，0表示不可用
 */
J2_API int constellation_propagator_is_cuda_available(void);

#ifdef __cplusplus
}
#endif

#endif // CONSTELLATION_PROPAGATOR_C_H