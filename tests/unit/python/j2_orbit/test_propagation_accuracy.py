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


def _j2_long_term_rates(mu, a, e, inc):
    J2 = 1.08263e-3
    RE = 6378137.0

    n = math.sqrt(mu / (a**3))
    p = a * (1.0 - e * e)
    factor = 1.5 * J2 * n * (RE / p) ** 2
    cos_i = math.cos(inc)
    cos_i_sq = cos_i * cos_i
    sqrt_term = math.sqrt(max(0.0, 1.0 - e * e))

    return {
        "O": -factor * cos_i,
        "w": 0.5 * factor * (5.0 * cos_i_sq - 1.0),
        "M": n + 0.5 * factor * sqrt_term * (3.0 * cos_i_sq - 1.0),
    }


def _angle_diff(a, b):
    diff = abs(a - b) % (2.0 * math.pi)
    return min(diff, 2.0 * math.pi - diff)


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

    # 验证 J2 长期平均漂移方向与量值
    rates = _j2_long_term_rates(mu, a, initial_elements["e"], initial_elements["i"])
    elapsed = T * fraction
    expected_O = (initial_elements["O"] + rates["O"] * elapsed) % (2.0 * math.pi)
    expected_w = (initial_elements["w"] + rates["w"] * elapsed) % (2.0 * math.pi)
    expected_M = (initial_elements["M"] + rates["M"] * elapsed) % (2.0 * math.pi)

    assert _angle_diff(result.O, expected_O) < 5e-5
    assert _angle_diff(result.w, expected_w) < 5e-5
    assert _angle_diff(result.M, expected_M) < 5e-5

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
    # 在当前历元施加脉冲前记录速度分量
    base_state = CStateVector()
    ret = j2_lib.constellation_propagator_get_satellite_state(constellation, ctypes.c_size_t(0), ctypes.byref(base_state))
    assert ret == 0
    base_component = float(base_state.v[axis])

    dv = (ctypes.c_double * 3)(*delta_v)
    ret = j2_lib.constellation_propagator_apply_impulse_to_constellation(constellation, dv, ctypes.c_size_t(1), ctypes.c_double(0.0))
    assert ret == 0

    state = CStateVector()
    ret = j2_lib.constellation_propagator_get_satellite_state(constellation, ctypes.c_size_t(0), ctypes.byref(state))
    assert ret == 0

    # 对应轴速度增量应与给定脉冲方向一致
    delta_component = float(state.v[axis]) - base_component
    assert delta_component > 0.0
