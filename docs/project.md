# 项目文档（j2-perturbation-orbit-propagator）

## 本次实施概览（方案1：最小改动）
- 修复/增强测试：tests/unit/j2_orbit/test_python_binding_apply_impulse_and_adaptive.py
  - 新增角度环域差函数 _angle_diff(a,b)
  - 重构 _almost_equal_elements：
    - 按域专用公差比较（a/e 绝对与相对容差，i/O/w/M 角度容差）
    - 忽略 t 字段（零脉冲等价性与时间戳语义无关）
    - 失败时打印详细差异便于诊断
  - 移除未知 pytest 标记（order）以消除告警
  - 在测试结束保存正式测试数据到 test/unit/j2_orbit/python_binding/apply_impulse_smoke
- 运行测试：该测试文件全部用例通过，无未知标记告警

## 运行说明（Windows）
- 使用提供脚本：scripts/tests/run_python_tests.ps1
  - 示例：
    - 运行整个文件：
      powershell -NoProfile -ExecutionPolicy Bypass -File scripts/tests/run_python_tests.ps1 -TestPattern "tests/unit/j2_orbit/test_python_binding_apply_impulse_and_adaptive.py"
    - 运行单用例：
      powershell -NoProfile -ExecutionPolicy Bypass -File scripts/tests/run_python_tests.ps1 -TestPattern "tests/unit/j2_orbit/test_python_binding_apply_impulse_and_adaptive.py::test_apply_impulse_smoke[dv0-True]"

## 目录约定
- 正式测试数据：test/ 下按分类存放（unit/performance/integration/system/stress/regression/other）
  - 本次新增：test/unit/j2_orbit/python_binding/apply_impulse_smoke
- 临时数据：temp/ 下（应从 git 忽略）
- 脚本：scripts/
- 配置：configs/

## 后续可选增强（概述）
- 方案2：
  - 更丰富的诊断与基线管理；脚本便捷支持 -k/标记过滤
- 方案3：
  - 抽取 tests/utils 公共比较/保存工具；脚本支持 nodeid/-k/标记/并行；补充更多边界用例