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

from conftest import CCompactOrbitalElements, COrbitalElements, CStateVector


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


def _norm3(vec):
    return math.sqrt(sum(float(v) * float(v) for v in vec))


def _diff_norm(a, b):
    return math.sqrt(sum((float(a[i]) - float(b[i])) ** 2 for i in range(3)))


def _propagate_constellation_state(j2_lib, initial_elements, step_size, target_time, compute_mode):
    handle = j2_lib.j2_constellation_propagator_create(ctypes.c_double(initial_elements["t"]))
    assert handle, "无法创建星座传播器"
    try:
        elem = CCompactOrbitalElements(
            a=initial_elements["a"],
            e=initial_elements["e"],
            i=initial_elements["i"],
            O=initial_elements["O"],
            w=initial_elements["w"],
            M=initial_elements["M"],
        )
        ret = j2_lib.j2_constellation_propagator_add_satellite(handle, ctypes.byref(elem))
        assert ret == 0

        ret = j2_lib.j2_constellation_propagator_set_step_size(handle, ctypes.c_double(step_size))
        assert ret == 0
        ret = j2_lib.j2_constellation_propagator_set_adaptive_step_size(handle, ctypes.c_int(0))
        assert ret == 0
        ret = j2_lib.j2_constellation_propagator_set_compute_mode(handle, ctypes.c_int(compute_mode))
        assert ret == 0

        ret = j2_lib.j2_constellation_propagator_propagate(handle, ctypes.c_double(target_time))
        assert ret == 0

        state = CStateVector()
        ret = j2_lib.j2_constellation_propagator_get_satellite_state(handle, 0, ctypes.byref(state))
        assert ret == 0
        return state
    finally:
        j2_lib.j2_constellation_propagator_destroy(handle)


def _apply_impulse_and_propagate(j2_lib, initial_elements, step_size, delta_v, target_time, compute_mode):
    handle = j2_lib.j2_constellation_propagator_create(ctypes.c_double(initial_elements["t"]))
    assert handle, "无法创建星座传播器"
    try:
        elem = CCompactOrbitalElements(
            a=initial_elements["a"],
            e=initial_elements["e"],
            i=initial_elements["i"],
            O=initial_elements["O"],
            w=initial_elements["w"],
            M=initial_elements["M"],
        )
        ret = j2_lib.j2_constellation_propagator_add_satellite(handle, ctypes.byref(elem))
        assert ret == 0

        ret = j2_lib.j2_constellation_propagator_set_step_size(handle, ctypes.c_double(step_size))
        assert ret == 0
        ret = j2_lib.j2_constellation_propagator_set_adaptive_step_size(handle, ctypes.c_int(0))
        assert ret == 0
        ret = j2_lib.j2_constellation_propagator_set_compute_mode(handle, ctypes.c_int(compute_mode))
        assert ret == 0

        delta_array = (ctypes.c_double * 3)(*delta_v)
        ret = j2_lib.j2_constellation_propagator_apply_impulse_to_constellation(
            handle,
            delta_array,
            ctypes.c_size_t(1),
            ctypes.c_double(0.0),
        )
        assert ret == 0

        ret = j2_lib.j2_constellation_propagator_propagate(handle, ctypes.c_double(target_time))
        assert ret == 0

        state = CStateVector()
        ret = j2_lib.j2_constellation_propagator_get_satellite_state(handle, ctypes.c_size_t(0), ctypes.byref(state))
        assert ret == 0
        return state
    finally:
        j2_lib.j2_constellation_propagator_destroy(handle)


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
    "step_size,target_seconds,delta_v",
    [
        (30.0, 600.0, [0.0, 0.2, 0.05]),
        (60.0, 1800.0, [0.15, -0.1, 0.08]),
    ],
)
def test_constellation_compute_modes_impulse_consistency(j2_lib, initial_elements, step_size, target_seconds, delta_v):
    modes = [("CPU_SCALAR", 0), ("CPU_SIMD", 1)]
    if bool(j2_lib.j2_constellation_propagator_is_cuda_available()):
        modes.append(("GPU_CUDA", 2))

    states = {}
    for mode_name, mode_value in modes:
        state = _apply_impulse_and_propagate(
            j2_lib,
            initial_elements,
            step_size,
            delta_v,
            target_seconds,
            mode_value,
        )
        pos = [float(state.r[i]) for i in range(3)]
        vel = [float(state.v[i]) for i in range(3)]
        states[mode_name] = (pos, vel)

    ref_pos, ref_vel = states["CPU_SCALAR"]
    ref_pos_norm = _norm3(ref_pos)
    ref_vel_norm = _norm3(ref_vel)
    rel_tol = 1e-8

    for mode_name, (pos, vel) in states.items():
        pos_diff = _diff_norm(pos, ref_pos)
        vel_diff = _diff_norm(vel, ref_vel)
        rel_pos_err = pos_diff / max(ref_pos_norm, 1.0)
        rel_vel_err = vel_diff / max(ref_vel_norm, 1.0)

        assert rel_pos_err <= rel_tol, (
            f"position mismatch for {mode_name} vs CPU_SCALAR after impulse (step={step_size}, T={target_seconds}) -> "
            f"rel_pos_err={rel_pos_err}, abs_pos_diff={pos_diff}"
        )
        assert rel_vel_err <= rel_tol, (
            f"velocity mismatch for {mode_name} vs CPU_SCALAR after impulse (step={step_size}, T={target_seconds}) -> "
            f"rel_vel_err={rel_vel_err}, abs_vel_diff={vel_diff}"
        )

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
    ret = j2_lib.j2_constellation_propagator_get_satellite_state(constellation, ctypes.c_size_t(0), ctypes.byref(base_state))
    assert ret == 0
    base_component = float(base_state.v[axis])

    dv = (ctypes.c_double * 3)(*delta_v)
    ret = j2_lib.j2_constellation_propagator_apply_impulse_to_constellation(constellation, dv, ctypes.c_size_t(1), ctypes.c_double(0.0))
    assert ret == 0

    state = CStateVector()
    ret = j2_lib.j2_constellation_propagator_get_satellite_state(constellation, ctypes.c_size_t(0), ctypes.byref(state))
    assert ret == 0

    # 对应轴速度增量应与给定脉冲方向一致
    delta_component = float(state.v[axis]) - base_component
    assert delta_component > 0.0


@pytest.mark.parametrize(
    "step_size,target_seconds",
    [
        (10.0, 600.0),
        (30.0, 3600.0),
        (60.0, 7200.0),
    ],
)
def test_constellation_compute_modes_accuracy(j2_lib, initial_elements, step_size, target_seconds):
    # 使用更小步长的单星传播器作为参考解
    baseline_step = max(1.0, step_size / 5.0)
    reference_state = _propagate_constellation_state(
        j2_lib,
        initial_elements,
        baseline_step,
        target_seconds,
        0,
    )
    ref_pos = [float(reference_state.r[i]) for i in range(3)]
    ref_vel = [float(reference_state.v[i]) for i in range(3)]
    ref_pos_norm = _norm3(ref_pos)
    ref_vel_norm = _norm3(ref_vel)

    modes = [("CPU_SCALAR", 0), ("CPU_SIMD", 1)]
    if bool(j2_lib.j2_constellation_propagator_is_cuda_available()):
        modes.append(("GPU_CUDA", 2))

    states = {}
    for mode_name, mode_value in modes:
        state = _propagate_constellation_state(j2_lib, initial_elements, step_size, target_seconds, mode_value)
        states[mode_name] = state

    rel_tol_baseline = 1e-9
    rel_tol_cross_mode = 1e-9

    for mode_name, state in states.items():
        pos = [float(state.r[i]) for i in range(3)]
        vel = [float(state.v[i]) for i in range(3)]
        pos_diff = _diff_norm(pos, ref_pos)
        vel_diff = _diff_norm(vel, ref_vel)
        rel_pos_err = pos_diff / max(ref_pos_norm, 1.0)
        rel_vel_err = vel_diff / max(ref_vel_norm, 1.0)

        assert rel_pos_err <= rel_tol_baseline, (
            f"baseline drift too large for {mode_name} (step={step_size}, T={target_seconds}) -> "
            f"rel_pos_err={rel_pos_err}, abs_pos_diff={pos_diff}"
        )
        assert rel_vel_err <= rel_tol_baseline, (
            f"baseline drift too large for {mode_name} (step={step_size}, T={target_seconds}) -> "
            f"rel_vel_err={rel_vel_err}, abs_vel_diff={vel_diff}"
        )

    scalar_state = states["CPU_SCALAR"]
    scalar_pos = [float(scalar_state.r[i]) for i in range(3)]
    scalar_vel = [float(scalar_state.v[i]) for i in range(3)]

    for mode_name, state in states.items():
        if mode_name == "CPU_SCALAR":
            continue
        pos = [float(state.r[i]) for i in range(3)]
        vel = [float(state.v[i]) for i in range(3)]
        pos_diff = _diff_norm(pos, scalar_pos)
        vel_diff = _diff_norm(vel, scalar_vel)
        rel_pos_err = pos_diff / max(ref_pos_norm, 1.0)
        rel_vel_err = vel_diff / max(ref_vel_norm, 1.0)

        assert rel_pos_err <= rel_tol_cross_mode, (
            f"compute-mode mismatch for {mode_name} vs CPU_SCALAR (step={step_size}, T={target_seconds}) -> "
            f"rel_pos_err={rel_pos_err}, abs_pos_diff={pos_diff}"
        )
        assert rel_vel_err <= rel_tol_cross_mode, (
            f"compute-mode mismatch for {mode_name} vs CPU_SCALAR (step={step_size}, T={target_seconds}) -> "
            f"rel_vel_err={rel_vel_err}, abs_vel_diff={vel_diff}"
        )
