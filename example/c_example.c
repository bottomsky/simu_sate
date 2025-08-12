/**
 * J2轨道传播器C语言接口使用示例
 * 
 * 该示例展示了如何直接使用C接口调用J2轨道传播器的功能，
 * 包括轨道传播、坐标转换等基本操作。
 * 
 * 编译命令 (Windows):
 *   gcc -o c_example c_example.c -L. -lj2_orbit_propagator
 * 
 * 编译命令 (Linux/macOS):
 *   gcc -o c_example c_example.c -L. -lj2_orbit_propagator -lm
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// 包含C接口头文件
#include "../j2_orbit_propagator_c.h"

// 辅助函数：打印轨道要素
void print_orbital_elements(const COrbitalElements* elements, const char* title) {
    printf("%s:\n", title);
    printf("  a: %.6f m\n", elements->a);
    printf("  e: %.6f\n", elements->e);
    printf("  i: %.6f rad (%.2f°)\n", elements->i, elements->i * 180.0 / M_PI);
    printf("  O: %.6f rad (%.2f°)\n", elements->O, elements->O * 180.0 / M_PI);
    printf("  w: %.6f rad (%.2f°)\n", elements->w, elements->w * 180.0 / M_PI);
    printf("  M: %.6f rad (%.2f°)\n", elements->M, elements->M * 180.0 / M_PI);
    printf("  t: %.6f s\n", elements->t);
}

// 辅助函数：打印状态向量
void print_state_vector(const CStateVector* state, const char* title) {
    printf("%s:\n", title);
    printf("  位置 (m): [%.3f, %.3f, %.3f]\n", 
           state->r[0], state->r[1], state->r[2]);
    printf("  速度 (m/s): [%.3f, %.3f, %.3f]\n", 
           state->v[0], state->v[1], state->v[2]);
}

// 辅助函数：打印3D向量
void print_vector3d(const double* vec, const char* title, const char* unit) {
    printf("%s: [%.3f, %.3f, %.3f] %s\n", 
           title, vec[0], vec[1], vec[2], unit);
}

int main() {
    printf("J2轨道传播器C语言接口示例\n");
    printf("================================\n\n");
    
    // 定义初始轨道要素 (类似ISS轨道)
    COrbitalElements initial_elements = {
        .a = 6780000.0,     // 半长轴 (m)
        .e = 0.0001,        // 偏心率
        .i = 51.6 * M_PI / 180.0,  // 倾角 (rad)
        .O = 0.0,           // 升交点赤经 (rad)
        .w = 0.0,           // 近地点幅角 (rad)
        .M = 0.0,           // 平近点角 (rad)
        .t = 0.0            // 历元时间 (s)
    };
    
    print_orbital_elements(&initial_elements, "初始轨道要素");
    
    // 创建传播器实例
    void* propagator = j2_propagator_create(&initial_elements);
    if (!propagator) {
        fprintf(stderr, "错误：无法创建传播器实例\n");
        return 1;
    }
    
    printf("\n传播器创建成功\n");
    
    // 设置积分步长
    double step_size = 60.0;  // 60秒
    if (j2_propagator_set_step_size(propagator, step_size) != 0) {
        fprintf(stderr, "错误：设置步长失败\n");
        j2_propagator_destroy(propagator);
        return 1;
    }
    
    printf("积分步长设置为: %.1f 秒\n\n", step_size);
    
    // 轨道传播到1小时后
    double target_time = 3600.0;  // 1小时
    COrbitalElements propagated_elements;
    
    if (j2_propagator_propagate(propagator, target_time, &propagated_elements) != 0) {
        fprintf(stderr, "错误：轨道传播失败\n");
        j2_propagator_destroy(propagator);
        return 1;
    }
    
    printf("传播到 t=%.1f 秒后:\n", target_time);
    print_orbital_elements(&propagated_elements, "传播后轨道要素");
    
    // 轨道要素转换为状态向量
    CStateVector state;
    if (j2_propagator_elements_to_state(propagator, &propagated_elements, &state) != 0) {
        fprintf(stderr, "错误：轨道要素转状态向量失败\n");
        j2_propagator_destroy(propagator);
        return 1;
    }
    
    printf("\n");
    print_state_vector(&state, "对应的状态向量");
    
    // 状态向量转换回轨道要素
    COrbitalElements recovered_elements;
    if (j2_propagator_state_to_elements(propagator, &state, target_time, &recovered_elements) != 0) {
        fprintf(stderr, "错误：状态向量转轨道要素失败\n");
        j2_propagator_destroy(propagator);
        return 1;
    }
    
    printf("\n");
    print_orbital_elements(&recovered_elements, "恢复的轨道要素");
    
    // ECI/ECEF坐标转换示例
    printf("\n坐标转换示例:\n");
    printf("================\n");
    
    double eci_position[3] = {state.r[0], state.r[1], state.r[2]};
    double ecef_position[3];
    double recovered_eci[3];
    
    print_vector3d(eci_position, "ECI位置", "m");
    
    // ECI转ECEF
    if (j2_eci_to_ecef_position(eci_position, target_time, ecef_position) != 0) {
        fprintf(stderr, "错误：ECI转ECEF失败\n");
    } else {
        print_vector3d(ecef_position, "ECEF位置", "m");
    }
    
    // ECEF转回ECI
    if (j2_ecef_to_eci_position(ecef_position, target_time, recovered_eci) != 0) {
        fprintf(stderr, "错误：ECEF转ECI失败\n");
    } else {
        print_vector3d(recovered_eci, "恢复ECI位置", "m");
    }
    
    // 计算GMST
    double gmst;
    if (j2_compute_gmst(target_time, &gmst) == 0) {
        printf("GMST: %.6f rad (%.2f°)\n", gmst, gmst * 180.0 / M_PI);
    }
    
    // 角度归一化示例
    double test_angle = 7.5;  // 超过2π的角度
    double normalized = j2_normalize_angle(test_angle);
    printf("角度归一化: %.3f → %.3f rad\n", test_angle, normalized);
    
    // 清理资源
    j2_propagator_destroy(propagator);
    
    printf("\n示例运行成功！\n");
    return 0;
}