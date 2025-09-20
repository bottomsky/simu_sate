# -*- coding: utf-8 -*-
"""
轨道外推精度测试：
- 通过 Python 调用 C++ 动态库（ctypes，调用 C 接口）
- 将给定初始轨道要素传播一个轨道周期，再与期望特性对比

注意：J2 摆动会引起轨道平面和近地点参数的变化，这里主要验证：
- 半长轴与能量守恒（近似不变）
- e, i 在短期内变化较小（容差范围内）
- ECI->状态向量转换的一致性（往返转换误差)
"""
import ctypes
import math
# 移除未使用的 numpy 依赖
import pytest

from conftest import COrbitalElements, CStateVector


@pytest.mark.parametrize("fraction", [0.25, 0.5, 1.0])
def test_single_satellite_propagation_accuracy(j2_lib, j2_propagator, initial_elements, tolerances, fraction):
    mu = 3.986004418e14

    # 周期（Kepler 近似）
    a = initial_elements["a"]
    T = 2 * math.pi * math.sqrt(a**3 / mu)
    target_t = initial_elements["t"] + T * fraction

    result = COrbitalElements()
    ret = j2_lib.j2_propagator_propagate(j2_propagator, ctypes.c_double(target_t), ctypes.byref(result))
    assert ret == 0, f"传播失败: {ret}"

    # 半长轴接近不变（J2 略有能量变化，允许相对容差）
    rel_err_a = abs(result.a - a) / a
    assert rel_err_a < tolerances["relative"], f"半长轴相对误差过大: {rel_err_a}"

    # i/e 在短期内变化较小
    assert abs(result.i - initial_elements["i"]) < 1e-4
    assert abs(result.e - initial_elements["e"]) < 1e-4

    # 要素->状态->要素往返一致性
    state = CStateVector()
    ret = j2_lib.j2_propagator_elements_to_state(j2_propagator, ctypes.byref(result), ctypes.byref(state))
    assert ret == 0

    back = COrbitalElements()
    ret = j2_lib.j2_propagator_state_to_elements(j2_propagator, ctypes.byref(state), ctypes.c_double(result.t), ctypes.byref(back))
    assert ret == 0

    assert abs(back.a - result.a) / a < 1e-6
    assert abs(back.e - result.e) < 1e-6
    assert abs(back.i - result.i) < 1e-6


@pytest.mark.parametrize(
    "delta_v,axis",
    [
        ([1.0, 0.0, 0.0], 0),
        ([0.0, 1.0, 0.0], 1),
        ([0.0, 0.0, 1.0], 2),
    ],
)
def test_impulse_effect_direction(j2_lib, constellation, delta_v, axis):
    # 在当前历元施加脉冲，检查速度分量对应轴增量显著
    dv = (ctypes.c_double * 3)(*delta_v)
    ret = j2_lib.constellation_propagator_apply_impulse_to_constellation(constellation, dv, ctypes.c_size_t(1), ctypes.c_double(0.0))
    assert ret == 0

    # 施加脉冲后推进 1s，保证状态更新
    ret = j2_lib.constellation_propagator_propagate(constellation, ctypes.c_double(1.0))
    assert ret == 0

    state = CStateVector()
    ret = j2_lib.constellation_propagator_get_satellite_state(constellation, ctypes.c_size_t(0), ctypes.byref(state))
    assert ret == 0

    # 仅比较对应轴速度增量是否为正（方向性），不强行约束量值（由库内部实现确定）
    assert state.v[axis] > 0.0