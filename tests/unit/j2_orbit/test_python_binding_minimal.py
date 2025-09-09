# -*- coding: utf-8 -*-
# 最小闭环单元测试：Python 通过 ctypes 调用 C++ J2 轨道传播 DLL
#
# 用例目标：
# - 验证传播、要素/状态互转与 ECI/ECEF 位置往返转换的基础闭环可用性。
# - 产出基础测试数据到 test/unit/j2_orbit/propagate/basic_case_001 目录。
#
# 运行方式：
# - 使用项目提供的 PowerShell 脚本 scripts/tests/run_python_tests.ps1
# - 或在虚拟环境中执行： .\.venv\Scripts\pytest tests\unit\j2_orbit\test_python_binding_minimal.py -q

import os
import sys
import json
from pathlib import Path
import math
import pytest

# 解析项目根目录与示例目录
THIS_FILE = Path(__file__).resolve()
REPO_ROOT = THIS_FILE.parents[3]  # tests/unit/j2_orbit/ -> up 3 levels
EXAMPLE_DIR = REPO_ROOT / "example"

# 将 example 加入 import 路径，导入 Python 绑定示例
sys.path.insert(0, str(EXAMPLE_DIR))
from python_binding_example import (
    J2OrbitPropagator,
    eci_to_ecef_position,
    ecef_to_eci_position,
)

# 输出测试数据的目标目录（正式测试数据目录）
OUTPUT_DIR = REPO_ROOT / "test" / "unit" / "j2_orbit" / "propagate" / "basic_case_001"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def _save_json(name: str, data: dict) -> None:
    # 保存字典为 JSON 文件到 OUTPUT_DIR。
    # 参数: name (str) 文件名（不含路径）；data (dict) 要保存的对象
    # 返回: None
    # 异常: IOError 写入失败时抛出
    path = OUTPUT_DIR / name
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _vec_err(a, b):
    # 计算两个 3 元向量的误差度量（最大绝对误差与相对误差）。
    # 参数: a, b 为长度为3的列表
    # 返回: (max_abs_err, rel_err)
    # 异常: ValueError 当输入长度不为3
    if len(a) != 3 or len(b) != 3:
        raise ValueError("向量长度必须为3")
    abs_errs = [abs(a[i] - b[i]) for i in range(3)]
    max_abs = max(abs_errs)
    norm_b = math.sqrt(sum(x * x for x in b)) or 1.0
    rel = math.sqrt(sum((a[i] - b[i]) ** 2 for i in range(3))) / norm_b
    return max_abs, rel


@pytest.mark.order(1)
def test_minimal_end_to_end():
    # 最小闭环端到端测试。
    # 步骤:
    #   1) 创建传播器并将步长设为 60s；
    #   2) 将初始轨道传播 3600s；
    #   3) 要素->状态->要素往返，检查关键要素保持一致（在合理容差内）；
    #   4) 取传播后的 ECI 位置，做 ECI->ECEF->ECI 往返，检查向量误差在容差内；
    #   5) 保存过程数据到正式测试数据目录。
    # 断言:
    #   - 传播后 t 与目标时间相等；
    #   - 往返后的 a/e/i 与输入近似一致；
    #   - 位置往返误差较小（相对误差 < 1e-6 或绝对误差 < 1e-3 米）。

    # 初始要素（示例：近似 LEO）
    initial = {
        "a": 6.78e6,
        "e": 1.0e-4,
        "i": 0.9006,
        "O": 0.0,
        "w": 0.0,
        "M": 0.0,
        "t": 0.0,
    }

    # 1) 创建传播器并设置步长
    prop = J2OrbitPropagator(initial)
    prop.set_step_size(60.0)
    assert prop.get_step_size() == pytest.approx(60.0, rel=1e-12, abs=0)

    # 2) 传播 3600s
    target_time = 3600.0
    propagated = prop.propagate(target_time)
    assert propagated["t"] == pytest.approx(target_time, rel=0, abs=1e-12)

    # 3) 要素->状态->要素 往返
    state = prop.elements_to_state(propagated)
    recovered = prop.state_to_elements(state, target_time)

    # a/e/i 的一致性（适当容差）
    assert recovered["a"] == pytest.approx(propagated["a"], rel=1e-8)
    assert recovered["e"] == pytest.approx(propagated["e"], rel=1e-8)
    assert recovered["i"] == pytest.approx(propagated["i"], rel=1e-8)

    # 4) 位置往返：ECI -> ECEF -> ECI
    eci_pos = state["r"]
    ecef_pos = eci_to_ecef_position(eci_pos, target_time)
    eci_pos_rec = ecef_to_eci_position(ecef_pos, target_time)

    max_abs, rel = _vec_err(eci_pos_rec, eci_pos)
    assert (rel < 1e-6) or (max_abs < 1e-3)

    # 5) 保存过程数据
    _save_json(
        "summary.json",
        {
            "initial": initial,
            "target_time": target_time,
            "propagated": propagated,
            "state": state,
            "eci_to_ecef": {
                "eci_pos": eci_pos,
                "ecef_pos": ecef_pos,
                "eci_pos_rec": eci_pos_rec,
                "max_abs_err_m": max_abs,
                "rel_err": rel,
            },
        },
    )

    # 额外输出：便于快速查看
    print(f"Roundtrip position error: max_abs={max_abs:.6e} m, rel={rel:.6e}")