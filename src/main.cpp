#include "j2_orbit_propagator.h"
#include <iomanip>

int main() {
    // 初始化轨道要素 (国际空间站近似轨道)
    OrbitalElements initial_elements;
    initial_elements.a = 6778137.0;   // 半长轴 (m)
    initial_elements.e = 0.0001;      // 偏心率
    initial_elements.i = 51.6 * M_PI / 180.0;  // 倾角 (rad)
    initial_elements.O = 0.0;         // 升交点赤经 (rad)
    initial_elements.w = 0.0;         // 近地点幅角 (rad)
    initial_elements.M = 0.0;         // 平近点角 (rad)
    initial_elements.t = 0.0;         // 历元时间 (s)
    
    // 创建轨道外推器实例
    J2OrbitPropagator propagator(initial_elements);
    propagator.setStepSize(60.0);  // 设置积分步长为60秒
    
    // 输出初始状态
    StateVector initial_state = propagator.elementsToState(initial_elements);
    std::cout << "初始状态 (t=0s):" << std::endl;
    std::cout << "位置: (" << initial_state.r[0] << ", " << initial_state.r[1] << ", " << initial_state.r[2] << ") m" << std::endl;
    std::cout << "速度: (" << initial_state.v[0] << ", " << initial_state.v[1] << ", " << initial_state.v[2] << ") m/s" << std::endl;
    std::cout << std::endl;
    
    // 外推轨道到不同时间点
    double times[] = {3600.0, 7200.0, 14400.0, 86400.0};  // 1小时, 2小时, 4小时, 1天
    int num_times = sizeof(times) / sizeof(times[0]);
    
    for (int i = 0; i < num_times; i++) {
        double t = times[i];
        OrbitalElements propagated = propagator.propagate(t);
        StateVector state = propagator.elementsToState(propagated);
        
        std::cout << "外推到 " << t << " 秒后:" << std::endl;
        std::cout << "半长轴: " << propagated.a << " m" << std::endl;
        std::cout << "偏心率: " << propagated.e << std::endl;
        std::cout << "倾角: " << propagated.i * 180.0 / M_PI << " 度" << std::endl;
        std::cout << "升交点赤经: " << propagated.O * 180.0 / M_PI << " 度" << std::endl;
        std::cout << "近地点幅角: " << propagated.w * 180.0 / M_PI << " 度" << std::endl;
        std::cout << "平近点角: " << propagated.M * 180.0 / M_PI << " 度" << std::endl;
        std::cout << "位置: (" << state.r[0] << ", " << state.r[1] << ", " << state.r[2] << ") m" << std::endl;
        std::cout << std::endl;
    }
    
    return 0;
}
