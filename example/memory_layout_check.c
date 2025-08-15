#include <stdio.h>
#include <stddef.h>
#include "../include/j2_orbit_propagator_c.h"
#include "../include/constellation_propagator_c.h"

int main() {
    printf("C Structure Layout Analysis\n");
    printf("===========================\n");
    
    // COrbitalElements分析
    printf("COrbitalElements:\n");
    printf("  Size: %zu bytes\n", sizeof(COrbitalElements));
    printf("  Field offsets:\n");
    printf("    a: %zu\n", offsetof(COrbitalElements, a));
    printf("    e: %zu\n", offsetof(COrbitalElements, e));
    printf("    i: %zu\n", offsetof(COrbitalElements, i));
    printf("    O: %zu\n", offsetof(COrbitalElements, O));
    printf("    w: %zu\n", offsetof(COrbitalElements, w));
    printf("    M: %zu\n", offsetof(COrbitalElements, M));
    printf("    t: %zu\n", offsetof(COrbitalElements, t));
    printf("  Expected total: %zu (7 doubles)\n", sizeof(double) * 7);
    printf("\n");
    
    // CCompactOrbitalElements分析
    printf("CCompactOrbitalElements:\n");
    printf("  Size: %zu bytes\n", sizeof(CCompactOrbitalElements));
    printf("  Field offsets:\n");
    printf("    a: %zu\n", offsetof(CCompactOrbitalElements, a));
    printf("    e: %zu\n", offsetof(CCompactOrbitalElements, e));
    printf("    i: %zu\n", offsetof(CCompactOrbitalElements, i));
    printf("    O: %zu\n", offsetof(CCompactOrbitalElements, O));
    printf("    w: %zu\n", offsetof(CCompactOrbitalElements, w));
    printf("    M: %zu\n", offsetof(CCompactOrbitalElements, M));
    printf("  Expected total: %zu (6 doubles)\n", sizeof(double) * 6);
    printf("\n");
    
    // CStateVector分析
    printf("CStateVector:\n");
    printf("  Size: %zu bytes\n", sizeof(CStateVector));
    printf("  Field offsets:\n");
    printf("    r: %zu\n", offsetof(CStateVector, r));
    printf("    v: %zu\n", offsetof(CStateVector, v));
    printf("  Expected total: %zu (6 doubles: 3+3)\n", sizeof(double) * 6);
    printf("\n");
    
    // 验证对齐需求
    printf("Alignment Analysis:\n");
    printf("  double alignment: %zu\n", _Alignof(double));
    printf("  COrbitalElements alignment: %zu\n", _Alignof(COrbitalElements));
    printf("  CCompactOrbitalElements alignment: %zu\n", _Alignof(CCompactOrbitalElements));
    printf("  CStateVector alignment: %zu\n", _Alignof(CStateVector));
    printf("\n");
    
    // 数组内存布局测试
    printf("Array Layout Test:\n");
    double test_r[3] = {1.0, 2.0, 3.0};
    double test_v[3] = {4.0, 5.0, 6.0};
    CStateVector test_state = { {test_r[0], test_r[1], test_r[2]}, {test_v[0], test_v[1], test_v[2]} };
    printf("  test_state.r address: %p\n", (void*)test_state.r);
    printf("  test_state.v address: %p\n", (void*)test_state.v);
    printf("  Difference: %td bytes\n", (char*)test_state.v - (char*)test_state.r);
    printf("  Expected: %zu bytes (3 doubles)\n", sizeof(double) * 3);
    
    return 0;
}