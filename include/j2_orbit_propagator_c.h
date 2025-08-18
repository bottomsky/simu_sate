#ifndef J2_ORBIT_PROPAGATOR_C_H
#define J2_ORBIT_PROPAGATOR_C_H

#ifdef __cplusplus
extern "C" {
#endif

// 导出宏定义（跨平台）
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

// C格式的轨道要素结构体
typedef struct {
    double a;   // 半长轴 (m)
    double e;   // 偏心率
    double i;   // 倾角 (rad)
    double O;   // 升交点赤经 (rad)
    double w;   // 近地点幅角 (rad)
    double M;   // 平近点角 (rad)
    double t;   // 历元时间 (s)
} COrbitalElements;

// C格式的状态向量结构体
typedef struct {
    double r[3];  // 位置矢量 (m) [x, y, z]
    double v[3];  // 速度矢量 (m/s) [vx, vy, vz]
} CStateVector;

// 不透明指针，用于隐藏C++类的实现细节
typedef void* J2PropagatorHandle;

// === 创建和销毁函数 ===

/**
 * @brief 创建J2轨道传播器实例
 * @param initial_elements 初始轨道要素
 * @return 传播器句柄，失败时返回NULL
 */
J2_API J2PropagatorHandle j2_propagator_create(const COrbitalElements* initial_elements);

/**
 * @brief 销毁J2轨道传播器实例
 * @param handle 传播器句柄
 */
J2_API void j2_propagator_destroy(J2PropagatorHandle handle);

// === 轨道传播函数 ===

/**
 * @brief 将轨道传播到指定时间
 * @param handle 传播器句柄
 * @param target_time 目标时间 (s)
 * @param result 输出的轨道要素
 * @return 0表示成功，非0表示失败
 */
J2_API int j2_propagator_propagate(J2PropagatorHandle handle, double target_time, COrbitalElements* result);

/**
 * @brief 从轨道要素计算状态向量
 * @param handle 传播器句柄
 * @param elements 轨道要素
 * @param state 输出的状态向量
 * @return 0表示成功，非0表示失败
 */
J2_API int j2_propagator_elements_to_state(J2PropagatorHandle handle, const COrbitalElements* elements, CStateVector* state);

/**
 * @brief 从状态向量计算轨道要素
 * @param handle 传播器句柄
 * @param state 状态向量
 * @param time 对应的时间 (s)
 * @param elements 输出的轨道要素
 * @return 0表示成功，非0表示失败
 */
J2_API int j2_propagator_state_to_elements(J2PropagatorHandle handle, const CStateVector* state, double time, COrbitalElements* elements);

/**
 * @brief 在ECI系施加速度增量(脉冲)，返回更新后的轨道要素
 * @param handle 传播器句柄
 * @param elements 施加脉冲前的轨道要素
 * @param delta_v 速度增量向量 (m/s) [dvx, dvy, dvz]
 * @param t 脉冲施加时刻（新要素的历元，单位s）
 * @param result 输出的更新后轨道要素
 * @return 0表示成功，非0表示失败
 */
J2_API int j2_propagator_apply_impulse(J2PropagatorHandle handle, const COrbitalElements* elements, const double delta_v[3], double t, COrbitalElements* result);

// === 参数设置函数 ===

/**
 * @brief 设置积分步长
 * @param handle 传播器句柄
 * @param step_size 步长 (s)
 * @return 0表示成功，非0表示失败
 */
J2_API int j2_propagator_set_step_size(J2PropagatorHandle handle, double step_size);

/**
 * @brief 获取当前积分步长
 * @param handle 传播器句柄
 * @param step_size 输出的步长 (s)
 * @return 0表示成功，非0表示失败
 */
J2_API int j2_propagator_get_step_size(J2PropagatorHandle handle, double* step_size);

/**
 * @brief 启用或禁用自适应步长
 * @param handle 传播器句柄
 * @param enable 1启用，0禁用
 * @return 0表示成功，非0表示失败
 */
J2_API int j2_propagator_set_adaptive_step_size(J2PropagatorHandle handle, int enable);

/**
 * @brief 设置自适应步长参数
 * @param handle 传播器句柄
 * @param tolerance 误差容忍度
 * @param min_step 最小步长 (s)
 * @param max_step 最大步长 (s)
 * @return 0表示成功，非0表示失败
 */
J2_API int j2_propagator_set_adaptive_parameters(J2PropagatorHandle handle, double tolerance, double min_step, double max_step);

// === 坐标转换函数 ===

/**
 * @brief ECI到ECEF坐标转换
 * @param eci_position ECI位置向量 [x, y, z] (m)
 * @param utc_seconds UTC时间 (秒，从某个参考时刻开始)
 * @param ecef_position 输出的ECEF位置向量 [x, y, z] (m)
 * @return 0表示成功，非0表示失败
 */
J2_API int j2_eci_to_ecef_position(const double eci_position[3], double utc_seconds, double ecef_position[3]);

/**
 * @brief ECEF到ECI坐标转换
 * @param ecef_position ECEF位置向量 [x, y, z] (m)
 * @param utc_seconds UTC时间 (秒，从某个参考时刻开始)
 * @param eci_position 输出的ECI位置向量 [x, y, z] (m)
 * @return 0表示成功，非0表示失败
 */
J2_API int j2_ecef_to_eci_position(const double ecef_position[3], double utc_seconds, double eci_position[3]);

/**
 * @brief ECI到ECEF速度转换
 * @param eci_position ECI位置向量 [x, y, z] (m)
 * @param eci_velocity ECI速度向量 [vx, vy, vz] (m/s)
 * @param utc_seconds UTC时间 (秒，从某个参考时刻开始)
 * @param ecef_velocity 输出的ECEF速度向量 [vx, y, z] (m/s)
 * @return 0表示成功，非0表示失败
 */
J2_API int j2_eci_to_ecef_velocity(const double eci_position[3], const double eci_velocity[3], double utc_seconds, double ecef_velocity[3]);

/**
 * @brief ECEF到ECI速度转换
 * @param ecef_position ECEF位置向量 [x, y, z] (m)
 * @param ecef_velocity ECEF速度向量 [vx, vy, vz] (m/s)
 * @param utc_seconds UTC时间 (秒，从某个参考时刻开始)
 * @param eci_velocity 输出的ECI速度向量 [vx, vy, vz] (m/s)
 * @return 0表示成功，非0表示失败
 */
J2_API int j2_ecef_to_eci_velocity(const double ecef_position[3], const double ecef_velocity[3], double utc_seconds, double eci_velocity[3]);

// === 工具函数 ===

/**
 * @brief 计算格林威治平恒星时
 * @param utc_seconds UTC时间 (秒，从某个参考时刻开始)
 * @param gmst 输出的GMST (弧度)
 * @return 0表示成功，非0表示失败
 */
J2_API int j2_compute_gmst(double utc_seconds, double* gmst);

/**
 * @brief 角度归一化到[0, 2π)范围
 * @param angle 输入角度 (弧度)
 * @return 归一化后的角度 (弧度)
 */
J2_API double j2_normalize_angle(double angle);

#ifdef __cplusplus
}
#endif

#endif // J2_ORBIT_PROPAGATOR_C_H