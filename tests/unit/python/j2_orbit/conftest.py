# -*- coding: utf-8 -*-
"""
Pytest 公用夹具：
- 加载 j2_orbit_propagator 动态库（支持 Linux/Windows/macOS）
- 定义 ctypes 结构体与函数原型（J2 单星接口 + Constellation 星座接口）
- 提供常用初始轨道要素与容差
"""
import ctypes
import os
import platform
from ctypes import Structure, c_double, c_void_p, c_int, c_size_t, POINTER
import pytest
from pathlib import Path

# 地球引力常数 (m^3/s^2)
EARTH_MU = 3.986004418e14


def _resolve_library_path():
    system = platform.system()
    if system == "Windows":
        lib_name = "j2_orbit_propagator.dll"
    elif system == "Darwin":
        lib_name = "libj2_orbit_propagator.dylib"
    else:
        lib_name = "libj2_orbit_propagator.so"

    # ENV 优先
    env_path = os.environ.get("J2_LIB_PATH")
    if env_path and os.path.exists(env_path):
        return env_path

    here = Path(__file__).resolve()
    repo_root = here.parents[4]  # tests/unit/python/j2_orbit -> 上溯 4 层到 simu_sate 目录

    candidates = [
        str(repo_root / "build" / "Release" / lib_name),
        str(repo_root / "build" / "Debug" / lib_name),
        str(repo_root / "build" / lib_name),
        str((repo_root / "example").resolve() / lib_name),
        lib_name,  # 系统路径
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return lib_name


# === ctypes 结构体 ===
class COrbitalElements(Structure):
    _fields_ = [
        ("a", c_double),
        ("e", c_double),
        ("i", c_double),
        ("O", c_double),
        ("w", c_double),
        ("M", c_double),
        ("t", c_double),
    ]


class CStateVector(Structure):
    _fields_ = [("r", c_double * 3), ("v", c_double * 3)]


DoubleArray3 = c_double * 3
DoubleArray9 = c_double * 9


class CCompactOrbitalElements(Structure):
    _fields_ = [
        ("a", c_double),
        ("e", c_double),
        ("i", c_double),
        ("O", c_double),
        ("w", c_double),
        ("M", c_double),
    ]


@pytest.fixture(scope="session")
def j2_lib():
    lib_path = _resolve_library_path()
    try:
        lib = ctypes.CDLL(lib_path)
    except OSError as e:
        pytest.skip(f"未找到或无法加载 J2 动态库: {lib_path} ({e})", allow_module_level=True)

    # J2 单星接口
    lib.j2_propagator_create.argtypes = [POINTER(COrbitalElements)]
    lib.j2_propagator_create.restype = c_void_p

    lib.j2_propagator_destroy.argtypes = [c_void_p]
    lib.j2_propagator_destroy.restype = None

    lib.j2_propagator_propagate.argtypes = [c_void_p, c_double, POINTER(COrbitalElements)]
    lib.j2_propagator_propagate.restype = c_int

    lib.j2_propagator_elements_to_state.argtypes = [
        c_void_p,
        POINTER(COrbitalElements),
        POINTER(CStateVector),
    ]
    lib.j2_propagator_elements_to_state.restype = c_int

    lib.j2_propagator_state_to_elements.argtypes = [
        c_void_p,
        POINTER(CStateVector),
        c_double,
        POINTER(COrbitalElements),
    ]
    lib.j2_propagator_state_to_elements.restype = c_int

    lib.j2_propagator_set_step_size.argtypes = [c_void_p, c_double]
    lib.j2_propagator_set_step_size.restype = c_int

    lib.j2_propagator_set_adaptive_step_size.argtypes = [c_void_p, c_int]
    lib.j2_propagator_set_adaptive_step_size.restype = c_int

    # ECI/ECEF 工具
    lib.j2_eci_to_ecef_position.argtypes = [POINTER(c_double), c_double, POINTER(c_double)]
    lib.j2_eci_to_ecef_position.restype = c_int
    lib.j2_ecef_to_eci_position.argtypes = [POINTER(c_double), c_double, POINTER(c_double)]
    lib.j2_ecef_to_eci_position.restype = c_int
    lib.j2_eci_to_ecef_velocity.argtypes = [POINTER(c_double), POINTER(c_double), c_double, POINTER(c_double)]
    lib.j2_eci_to_ecef_velocity.restype = c_int
    lib.j2_ecef_to_eci_velocity.argtypes = [POINTER(c_double), POINTER(c_double), c_double, POINTER(c_double)]
    lib.j2_ecef_to_eci_velocity.restype = c_int
    lib.j2_ecef_to_geodetic.argtypes = [POINTER(c_double), POINTER(c_double)]
    lib.j2_ecef_to_geodetic.restype = c_int
    lib.j2_geodetic_to_ecef.argtypes = [POINTER(c_double), POINTER(c_double)]
    lib.j2_geodetic_to_ecef.restype = c_int
    lib.j2_eci_to_geodetic.argtypes = [POINTER(c_double), c_double, POINTER(c_double)]
    lib.j2_eci_to_geodetic.restype = c_int
    lib.j2_geodetic_to_eci.argtypes = [POINTER(c_double), c_double, POINTER(c_double)]
    lib.j2_geodetic_to_eci.restype = c_int
    lib.j2_rtn_to_eci_rotation.argtypes = [POINTER(c_double), POINTER(c_double), POINTER(c_double)]
    lib.j2_rtn_to_eci_rotation.restype = c_int
    lib.j2_eci_to_rtn_rotation.argtypes = [POINTER(c_double), POINTER(c_double), POINTER(c_double)]
    lib.j2_eci_to_rtn_rotation.restype = c_int
    lib.j2_eci_to_rtn_vector.argtypes = [POINTER(c_double), POINTER(c_double), POINTER(c_double), POINTER(c_double)]
    lib.j2_eci_to_rtn_vector.restype = c_int
    lib.j2_rtn_to_eci_vector.argtypes = [POINTER(c_double), POINTER(c_double), POINTER(c_double), POINTER(c_double)]
    lib.j2_rtn_to_eci_vector.restype = c_int
    lib.j2_compute_gmst.argtypes = [c_double, POINTER(c_double)]
    lib.j2_compute_gmst.restype = c_int
    lib.j2_normalize_angle.argtypes = [c_double]
    lib.j2_normalize_angle.restype = c_double

    # 星座接口
    lib.constellation_propagator_create.argtypes = [c_double]
    lib.constellation_propagator_create.restype = c_void_p

    lib.constellation_propagator_destroy.argtypes = [c_void_p]
    lib.constellation_propagator_destroy.restype = None

    lib.constellation_propagator_add_satellite.argtypes = [c_void_p, POINTER(CCompactOrbitalElements)]
    lib.constellation_propagator_add_satellite.restype = c_int

    lib.constellation_propagator_get_satellite_state.argtypes = [
        c_void_p,
        c_size_t,
        POINTER(CStateVector),
    ]
    lib.constellation_propagator_get_satellite_state.restype = c_int

    lib.constellation_propagator_apply_impulse_to_constellation.argtypes = [
        c_void_p,
        POINTER(c_double),
        c_size_t,
        c_double,
    ]
    lib.constellation_propagator_apply_impulse_to_constellation.restype = c_int

    lib.constellation_propagator_propagate.argtypes = [c_void_p, c_double]
    lib.constellation_propagator_propagate.restype = c_int

    return lib


@pytest.fixture(scope="session")
def tolerances():
    return {
        "angle": 1e-6,  # rad
        "length": 1e-3,  # m
        "velocity": 1e-3,  # m/s
        "relative": 1e-6,
    }


@pytest.fixture()
def initial_elements():
    # 以典型 LEO 圆轨道为例：a=7000km, e≈0, i=98°
    return {
        "a": 7000e3,
        "e": 1e-3,
        "i": 98.0 * 3.141592653589793 / 180.0,
        "O": 0.0,
        "w": 0.0,
        "M": 0.0,
        "t": 0.0,
    }


@pytest.fixture()
def j2_propagator(j2_lib, initial_elements):
    c_elems = COrbitalElements(
        a=initial_elements["a"],
        e=initial_elements["e"],
        i=initial_elements["i"],
        O=initial_elements["O"],
        w=initial_elements["w"],
        M=initial_elements["M"],
        t=initial_elements["t"],
    )
    handle = j2_lib.j2_propagator_create(ctypes.byref(c_elems))
    assert handle, "无法创建 J2 传播器"
    try:
        yield handle
    finally:
        j2_lib.j2_propagator_destroy(handle)


@pytest.fixture()
def constellation(j2_lib, initial_elements):
    epoch = initial_elements["t"]
    handle = j2_lib.constellation_propagator_create(epoch)
    assert handle, "无法创建 星座传播器"

    c_elem = CCompactOrbitalElements(
        a=initial_elements["a"],
        e=initial_elements["e"],
        i=initial_elements["i"],
        O=initial_elements["O"],
        w=initial_elements["w"],
        M=initial_elements["M"],
    )
    # 添加 1 颗卫星，便于验证
    ret = j2_lib.constellation_propagator_add_satellite(handle, ctypes.byref(c_elem))
    assert ret == 0, f"添加卫星失败: {ret}"
    try:
        yield handle
    finally:
        j2_lib.constellation_propagator_destroy(handle)


@pytest.fixture(scope="session")
def earth_mu():
    return EARTH_MU