# -*- coding: utf-8 -*-
"""
测试目标：
1) 验证 apply_impulse 接口：
   - 零脉冲应与原结果一致（等价性）
   - 非零脉冲应与原结果不同（差异性，作为烟囱测试）
2) 验证 set_adaptive_parameters 接口：
   - 在符号可用时，可成功设置（返回不抛异常）
   - 在符号不可用时，合理跳过测试
注意：
- 当底层 DLL 未导出相关符号时，python_binding_example 中已做可选绑定，本用例会根据 hasattr(j2_lib, symbol) 进行 skip。
- 运行方式建议通过 scripts/tests/run_python_tests.ps1 执行。
"""

import os
import sys
import json
import math
import pytest

# 动态调整 sys.path 以便直接导入 example 下的绑定示例
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
EXAMPLE_DIR = os.path.join(ROOT, "example")
if EXAMPLE_DIR not in sys.path:
    sys.path.insert(0, EXAMPLE_DIR)

from python_binding_example import J2OrbitPropagator, j2_lib  # type: ignore


def _save_json(path: str, data: dict) -> None:
    """保存字典为 JSON 文件（UTF-8）。
    参数:
      - path: 输出文件完整路径
      - data: 待保存的 Python 字典
    返回值:
      - 无
    异常:
      - 可能抛出文件写入相关异常（由调用方处理或由 pytest 报错）
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _angle_diff(a: float, b: float) -> float:
    """计算两个角度的环域差异，范围在 [-pi, pi]。
    参数:
      - a: 弧度制角
      - b: 弧度制角
    返回值:
      - 最小环域差值 a-b 归一化到 [-pi, pi]
    """
    two_pi = 2.0 * math.pi
    d = (a - b) % two_pi
    if d > math.pi:
        d -= two_pi
    return d


def _almost_equal_elements(e1: dict, e2: dict) -> bool:
    """按域专用公差比较轨道要素近似相等（忽略 t 字段）。
    参数:
      - e1: 轨道要素字典，包含 a,e,i,O,w,M(,t)
      - e2: 轨道要素字典，包含 a,e,i,O,w,M(,t)
    返回值:
      - True 若认为两组要素在容差内等价；否则 False
    说明:
      - 角度 i, O, w, M 使用环域差异（[-pi, pi]）进行比较
      - 忽略 t 字段，因为 apply_impulse 返回的 t 语义依赖底层实现（可能为脉冲施加时刻），与 base.t 不同
    """
    tol_abs = {
        "a": 1e-5,   # 米
        "e": 1e-12,  # 无量纲
        "i": 1e-8,   # 弧度
        "O": 1e-8,
        "w": 1e-8,
        "M": 1e-8,
    }
    rtol = 1e-8

    angle_keys = {"i", "O", "w", "M"}

    for k in ["a", "e", "i", "O", "w", "M"]:
        v1 = float(e1[k])
        v2 = float(e2[k])
        if k in angle_keys:
            if abs(_angle_diff(v1, v2)) > tol_abs[k]:
                return False
        else:
            if not math.isclose(v1, v2, rel_tol=rtol, abs_tol=tol_abs[k]):
                return False
    return True


@pytest.mark.parametrize("dv, expect_equal", [([0.0, 0.0, 0.0], True), ([0.1, 0.0, 0.0], False)])
def test_apply_impulse_smoke(dv, expect_equal, tmp_path):
    """烟囱测试：验证 apply_impulse 的零/非零脉冲行为。
    步骤：
      1. 生成初始根数并传播到 t1
      2. 若符号缺失则 skip
      3. 在 t=0.0 处施加脉冲，比较几何要素等价性（忽略 t）
      4. 持久化对比数据到 test 目录
    """
    # 准备初始要素与传播
    initial = {"a": 6.78e6, "e": 0.0001, "i": 0.9006, "O": 0.0, "w": 0.0, "M": 0.0, "t": 0.0}
    prop = J2OrbitPropagator(initial)
    prop.set_step_size(60.0)
    t1 = 3600.0
    base = prop.propagate(t1)

    # 若符号不可用则跳过
    if not hasattr(j2_lib, "j2_propagator_apply_impulse"):
        pytest.skip("动态库未导出 j2_propagator_apply_impulse，跳过测试")

    # 应用脉冲（选择在 t=0.0 施加）
    after = prop.apply_impulse(base, dv, 0.0)

    # 断言
    if expect_equal:
        eq = _almost_equal_elements(base, after)
        if not eq:
            # 打印差异帮助定位失败原因
            keys = ["a", "e", "i", "O", "w", "M"]
            diffs = {k: float(base[k]) - float(after[k]) for k in keys}
            print("elements diff:", diffs)
        assert eq
    else:
        assert not _almost_equal_elements(base, after)

    # 保存测试数据（正式测试数据路径）
    out_dir = os.path.join(ROOT, "test", "unit", "j2_orbit", "python_binding", "apply_impulse_smoke")
    _save_json(os.path.join(out_dir, f"dv_{'eq' if expect_equal else 'neq'}.json"), {"base": base, "after": after, "dv": dv})


def test_set_adaptive_parameters_smoke():
    """烟囱测试：验证 set_adaptive_parameters 可调用性（不抛异常）。"""
    initial = {"a": 6.78e6, "e": 0.0001, "i": 0.9006, "O": 0.0, "w": 0.0, "M": 0.0, "t": 0.0}
    prop = J2OrbitPropagator(initial)

    if not hasattr(j2_lib, "j2_propagator_set_adaptive_parameters"):
        pytest.skip("动态库未导出 j2_propagator_set_adaptive_parameters，跳过测试")

    # 仅验证不抛异常（烟囱）
    prop.set_adaptive_parameters(1e-6, 0.1, 120.0)