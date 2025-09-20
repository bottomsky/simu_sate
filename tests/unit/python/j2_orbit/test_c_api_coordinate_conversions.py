# -*- coding: utf-8 -*-
"""验证 C 语言接口的坐标/速度转换与 RTN 旋转实现。"""
import ctypes
import math
import sys
from pathlib import Path

import pytest

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

import conftest as j2_conf

COrbitalElements = j2_conf.COrbitalElements
CStateVector = j2_conf.CStateVector
DoubleArray3 = j2_conf.DoubleArray3
DoubleArray9 = j2_conf.DoubleArray9


def _elements_from_dict(data: dict) -> COrbitalElements:
    return COrbitalElements(
        a=data["a"],
        e=data["e"],
        i=data["i"],
        O=data["O"],
        w=data["w"],
        M=data["M"],
        t=data["t"],
    )


def _state_from_propagator(j2_lib, handle, elements: dict) -> CStateVector:
    c_elems = _elements_from_dict(elements)
    state = CStateVector()
    rc = j2_lib.j2_propagator_elements_to_state(
        handle,
        ctypes.byref(c_elems),
        ctypes.byref(state),
    )
    assert rc == 0
    return state


def _as_arrays(state: CStateVector) -> tuple[DoubleArray3, DoubleArray3]:
    return DoubleArray3(*state.r), DoubleArray3(*state.v)


def _matmul(a: list[list[float]], b: list[list[float]]) -> list[list[float]]:
    return [
        [sum(a[i][k] * b[k][j] for k in range(3)) for j in range(3)]
        for i in range(3)
    ]


def _transpose(m: list[list[float]]) -> list[list[float]]:
    return [[m[j][i] for j in range(3)] for i in range(3)]


@pytest.mark.parametrize("utc_seconds", [0.0, 1234.5])
def test_eci_ecef_roundtrip(j2_lib, j2_propagator, initial_elements, tolerances, utc_seconds):
    state = _state_from_propagator(j2_lib, j2_propagator, initial_elements)
    eci_pos, eci_vel = _as_arrays(state)

    ecef_pos = DoubleArray3()
    assert j2_lib.j2_eci_to_ecef_position(eci_pos, utc_seconds, ecef_pos) == 0

    eci_pos_back = DoubleArray3()
    assert j2_lib.j2_ecef_to_eci_position(ecef_pos, utc_seconds, eci_pos_back) == 0
    assert list(eci_pos_back) == pytest.approx(
        list(eci_pos), rel=tolerances["relative"], abs=tolerances["length"]
    )

    ecef_vel = DoubleArray3()
    assert j2_lib.j2_eci_to_ecef_velocity(eci_pos, eci_vel, utc_seconds, ecef_vel) == 0

    eci_vel_back = DoubleArray3()
    assert (
        j2_lib.j2_ecef_to_eci_velocity(ecef_pos, ecef_vel, utc_seconds, eci_vel_back)
        == 0
    )
    assert list(eci_vel_back) == pytest.approx(
        list(eci_vel), rel=tolerances["relative"], abs=tolerances["velocity"]
    )


def test_rtn_vector_roundtrip(j2_lib, j2_propagator, initial_elements, tolerances):
    state = _state_from_propagator(j2_lib, j2_propagator, initial_elements)
    eci_pos, eci_vel = _as_arrays(state)
    test_vec = DoubleArray3(1.2, -0.4, 0.8)

    vec_rtn = DoubleArray3()
    assert j2_lib.j2_eci_to_rtn_vector(eci_pos, eci_vel, test_vec, vec_rtn) == 0

    vec_eci = DoubleArray3()
    assert j2_lib.j2_rtn_to_eci_vector(eci_pos, eci_vel, vec_rtn, vec_eci) == 0

    assert list(vec_eci) == pytest.approx(
        list(test_vec), rel=tolerances["relative"], abs=1e-10
    )


def test_rtn_rotation_matrices_are_inverses(j2_lib, j2_propagator, initial_elements):
    state = _state_from_propagator(j2_lib, j2_propagator, initial_elements)
    eci_pos, eci_vel = _as_arrays(state)

    rtn_matrix = DoubleArray9()
    assert j2_lib.j2_eci_to_rtn_rotation(eci_pos, eci_vel, rtn_matrix) == 0

    eci_matrix = DoubleArray9()
    assert j2_lib.j2_rtn_to_eci_rotation(eci_pos, eci_vel, eci_matrix) == 0

    rtn_rows = [list(rtn_matrix[i : i + 3]) for i in range(0, 9, 3)]
    eci_rows = [list(eci_matrix[i : i + 3]) for i in range(0, 9, 3)]

    identity = [[1.0 if i == j else 0.0 for j in range(3)] for i in range(3)]
    product = _matmul(rtn_rows, eci_rows)
    for row, expected in zip(product, identity):
        assert row == pytest.approx(expected, rel=1e-9, abs=1e-9)

    transpose = _transpose(rtn_rows)
    for row, expected in zip(eci_rows, transpose):
        assert row == pytest.approx(expected, rel=1e-9, abs=1e-9)

    for row in rtn_rows:
        norm = math.sqrt(sum(x * x for x in row))
        assert math.isclose(norm, 1.0, rel_tol=1e-9, abs_tol=1e-9)
