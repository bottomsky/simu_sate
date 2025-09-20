# -*- coding: utf-8 -*-
"""
全局 PyTest 夹具：
- project_root: 仓库根路径 Path
- output_base: 默认输出目录（优先 tmp_path_factory，如设置 TEST_OUTPUT_DIR 则使用该目录）
- ensure_output_subdir: 基于 suite/case 创建子目录并返回 Path
"""
from __future__ import annotations
import os
from pathlib import Path
import pytest

@pytest.fixture(scope="session")
def project_root() -> Path:
    # 本文件位于 simu_sate/tests/conftest.py，向上两级即仓库根
    return Path(__file__).resolve().parents[2]

@pytest.fixture(scope="session")
def output_base(tmp_path_factory: pytest.TempPathFactory) -> Path:
    override = os.environ.get("TEST_OUTPUT_DIR")
    if override:
        p = Path(override)
        p.mkdir(parents=True, exist_ok=True)
        return p
    # 默认使用临时目录，保证测试可重复、无副作用
    return tmp_path_factory.mktemp("test-output")

@pytest.fixture()
def ensure_output_subdir(output_base: Path):
    def _make(*subpaths: str) -> Path:
        p = output_base.joinpath(*subpaths)
        p.mkdir(parents=True, exist_ok=True)
        return p
    return _make